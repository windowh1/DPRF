## built-in
import math,logging,json,random,functools,os
import types
os.environ["TOKENIZERS_PARALLELISM"]='true'
os.environ["WANDB_IGNORE_GLOBS"]='*.bin' ## not upload ckpt to wandb cloud

## third-party
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs
import transformers
from transformers import (
    BertTokenizer,
    BertModel,
    AutoModel,
    DPRQuestionEncoder, 
    DPRContextEncoder
)
transformers.logging.set_verbosity_error()
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from tqdm import tqdm

## own
from utils import (
    get_yaml_file,
    set_seed,
    get_linear_scheduler,
    normalize_query,
    normalize_document,
)

logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    ## adding args here for more control from CLI is possible
    parser.add_argument("--config_file",default='config/train_dpr_nq.yaml')
    args = parser.parse_args()

    yaml_config = get_yaml_file(args.config_file)
    args_dict = {k:v for k,v in vars(args).items() if v is not None}
    yaml_config.update(args_dict)
    args = types.SimpleNamespace(**yaml_config)
    return args

class DualEncoder(nn.Module):
    def __init__(self,query_encoder,doc_encoder):
        super().__init__()
        self.query_encoder = query_encoder
        self.doc_encoder = doc_encoder

    def forward(
        self,
        query_input_ids, # [bs,seq_len]
        query_attention_mask, # [bs,seq_len]
        query_token_type_ids, # [bs,seq_len],
        doc_input_ids, # [bs*n_doc,seq_len]
        doc_attention_mask, # [bs*n_doc,seq_len]
        doc_token_type_ids, # [bs*n_doc,seq_len]
    ):  
        CLS_POS = 0
        ## [bs,n_dim]
        query_embedding = self.query_encoder(
            input_ids=query_input_ids,
            attention_mask = query_attention_mask,
            token_type_ids = query_token_type_ids,
            ).last_hidden_state[:,CLS_POS,:]
        
        ## [bs * n_doc,n_dim]
        doc_embedding = self.doc_encoder(
            input_ids = doc_input_ids,
            attention_mask = doc_attention_mask,
            token_type_ids = doc_token_type_ids,
            ).last_hidden_state[:,CLS_POS,:]
        
        return query_embedding,doc_embedding 

def calculate_dpr_loss(matching_score,labels):
    return F.nll_loss(input=F.log_softmax(matching_score,dim=1),target=labels)

def calculate_kd_loss(matching_score, large_score, labels, temperature=5.0):
    soft_prob = F.log_softmax(matching_score / temperature, dim=1)
    soft_tgt = F.log_softmax(large_score / temperature, dim=1)
    
    soft_loss = nn.KLDivLoss(reduction='batchmean',log_target=True)(soft_prob, soft_tgt)

    orig_prob = F.log_softmax(matching_score, dim=1)
    orig_loss = F.nll_loss(input=orig_prob, target=labels)

    loss = soft_loss + orig_loss / (temperature**2)
    return loss

class SparEncoder(nn.Module):
    def __init__(self, dpr_query_encoder, dpr_doc_encoder, lex_query_encoder, lex_doc_encoder):
        super().__init__()
        self.dpr_query_encoder = dpr_query_encoder
        self.dpr_doc_encoder = dpr_doc_encoder
        self.lex_query_encoder = lex_query_encoder
        self.lex_doc_encoder = lex_doc_encoder

    def forward(
            self,
            query_input_ids, # [bs,seq_len]
            query_attention_mask, # [bs,seq_len]
            query_token_type_ids, # [bs,seq_len],
            doc_input_ids, # [bs*n_doc,seq_len]
            doc_attention_mask, # [bs*n_doc,seq_len]
            doc_token_type_ids, # [bs*n_doc,seq_len]
    ):
        with torch.no_grad():
            CLS_POS = 0
            dpr_query_emb = self.dpr_query_encoder(input_ids=query_input_ids,
                                                attention_mask=query_attention_mask, 
                                                token_type_ids=query_token_type_ids).pooler_output
                # [batch_size, hidden_size] = [16, 768]
            dpr_doc_emb = self.dpr_doc_encoder(input_ids=doc_input_ids, 
                                                attention_mask=doc_attention_mask, 
                                                token_type_ids=doc_token_type_ids).pooler_output
                # [batch_size, hidden_size] = [32, 768]      
            lex_query_emb = self.lex_query_encoder(input_ids=query_input_ids,
                                                attention_mask=query_attention_mask, 
                                                token_type_ids=query_token_type_ids).last_hidden_state[:,CLS_POS,:]
                # [batch_size, hidden_size] = [16, 768]
            lex_doc_emb = self.lex_doc_encoder(input_ids=doc_input_ids, 
                                                attention_mask=doc_attention_mask, 
                                                token_type_ids=doc_token_type_ids).last_hidden_state[:,CLS_POS,:]
                # [batch_size, hidden_size] = [32, 768]      
        return dpr_query_emb, dpr_doc_emb, lex_query_emb, lex_doc_emb

def get_large_score(dpr_query_emb, dpr_doc_emb, lex_query_emb, lex_doc_emb, concat_weight=0.7):
    with torch.no_grad():
        dpr_score = torch.matmul(dpr_query_emb, dpr_doc_emb.permute(1,0))
        lex_score = torch.matmul(lex_query_emb, lex_doc_emb.permute(1,0)) # [16, 32]
        large_score = dpr_score + concat_weight * lex_score

    return large_score

def calculate_hit_cnt(matching_score,labels):
    _, max_ids = torch.max(matching_score,1)
    return (max_ids == labels).sum()

def calculate_average_rank(matching_score,labels):
    _,indices = torch.sort(matching_score,dim=1,descending=True)
    ranks = []
    for idx,label in enumerate(labels):
        rank = ((indices[idx] == label).nonzero()).item() + 1  ##  rank starts from 1
        ranks.append(rank)
    return ranks


class QADataset(torch.utils.data.Dataset):
    def __init__(self,file_path):
        self.data = json.load(open(file_path))
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        return self.data[idx]

    @staticmethod
    def collate_fn(samples,tokenizer,args,stage):
        
        # prepare query input
        queries = [normalize_query(x['question']) for x in samples]
        query_inputs = tokenizer(queries,max_length=256,padding=True,truncation=True,return_tensors='pt')
        
        # prepare document input
        ## select the first positive document
        ## passage = title + document
        positive_passages = [x['positive_ctxs'][0] for x in samples]
        positive_titles = [x['title'] for x in positive_passages]
        positive_docs = [x['text'] for x in positive_passages]

        negative_passages = [random.choice(x['hard_negative_ctxs']) 
                                if len(x['hard_negative_ctxs']) != 0  
                                else random.choice(x['negative_ctxs']) 
                                for x in samples ]

        negative_titles = [x["title"] for x in negative_passages]
        negative_docs = [x["text"] for x in negative_passages]
        titles = positive_titles + negative_titles
        docs = positive_docs + negative_docs

        doc_inputs = tokenizer(titles,docs,max_length=256,padding=True,truncation=True,return_tensors='pt')

        return {
            'query_input_ids':query_inputs.input_ids,
            'query_attention_mask':query_inputs.attention_mask,
            'query_token_type_ids':query_inputs.token_type_ids,

            "doc_input_ids":doc_inputs.input_ids,
            "doc_attention_mask":doc_inputs.attention_mask,
            "doc_token_type_ids":doc_inputs.token_type_ids,
        }

def validate(model, dataloader, accelerator):
    model.eval()
    query_embeddings = []
    positive_doc_embeddings = []
    negative_doc_embeddings = []
    for batch in dataloader:
        with torch.no_grad():
            query_embedding,doc_embedding = model(**batch)
        query_num,_ = query_embedding.shape
        query_embeddings.append(query_embedding.cpu())
        positive_doc_embeddings.append(doc_embedding[:query_num,:].cpu())
        negative_doc_embeddings.append(doc_embedding[query_num:,:].cpu())

    query_embeddings = torch.cat(query_embeddings,dim=0) # bs, emb_dim
    doc_embeddings = torch.cat(positive_doc_embeddings+negative_doc_embeddings,dim=0)  # num_pos+num_neg, emb_dim
    matching_score = torch.matmul(query_embeddings,doc_embeddings.permute(1,0)) # bs, num_pos+num_neg
    labels = torch.arange(query_embeddings.shape[0],dtype=torch.int64).to(matching_score.device) # gold label for each queries
    loss = calculate_dpr_loss(matching_score,labels=labels).item()
    ranks = calculate_average_rank(matching_score,labels=labels)
    
    if accelerator.use_distributed and accelerator.num_processes>1:
        ranks_from_all_gpus = [None for _ in range(accelerator.num_processes)] 
        dist.all_gather_object(ranks_from_all_gpus,ranks)
        ranks = [x for y in ranks_from_all_gpus for x in y]

        loss_from_all_gpus = [None for _ in range(accelerator.num_processes)] 
        dist.all_gather_object(loss_from_all_gpus,loss)
        loss = sum(loss_from_all_gpus)/len(loss_from_all_gpus)
    
    return sum(ranks)/len(ranks),loss

def main():
    args = parse_args()
    set_seed(args.seed)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        device_placement=True,
        mixed_precision='fp16',
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with='wandb',
        kwargs_handlers=[kwargs]
    )

    accelerator.init_trackers(
        project_name="Knowledge Distillation", 
        config=args,
    )
    if accelerator.is_local_main_process:
        wandb_tracker = accelerator.get_tracker("wandb")
        LOG_DIR = wandb_tracker.run.dir

    tokenizer = BertTokenizer.from_pretrained(args.base_model)
    query_encoder = BertModel.from_pretrained(args.base_model,add_pooling_layer=False) # cpu
    doc_encoder = BertModel.from_pretrained(args.base_model,add_pooling_layer=False) # cpu

    dual_encoder = DualEncoder(query_encoder,doc_encoder)
    dual_encoder.train()
    
    dpr_query_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-multiset-base")
    dpr_doc_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-multiset-base")
    lex_query_encoder = AutoModel.from_pretrained('facebook/spar-wiki-bm25-lexmodel-query-encoder')
    lex_doc_encoder = AutoModel.from_pretrained('facebook/spar-wiki-bm25-lexmodel-context-encoder')

    spar_encoder = SparEncoder(dpr_query_encoder, dpr_doc_encoder, lex_query_encoder, lex_doc_encoder)
    spar_encoder.eval()

    train_dataset = QADataset(args.train_file)
    train_collate_fn = functools.partial(QADataset.collate_fn,
                                         tokenizer=tokenizer,
                                         stage='train',
                                         args=args,)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.per_device_train_batch_size,
                                                   shuffle=True,
                                                   collate_fn=train_collate_fn,
                                                   num_workers=args.num_workers,
                                                   pin_memory=True,)
    
    dev_dataset = QADataset(args.dev_file)
    dev_collate_fn = functools.partial(QADataset.collate_fn,
                                       tokenizer=tokenizer,
                                       stage='dev',
                                       args=args,)
    dev_dataloader = torch.utils.data.DataLoader(dev_dataset,
                                                 batch_size=args.per_device_eval_batch_size,
                                                 shuffle=False,
                                                 collate_fn=dev_collate_fn,
                                                 num_workers=args.num_workers,
                                                 pin_memory=True,)
    
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in dual_encoder.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in dual_encoder.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters,lr=args.lr, eps=args.adam_eps)
    
    dual_encoder, spar_encoder, optimizer, train_dataloader, dev_dataloader = accelerator.prepare(
        dual_encoder, spar_encoder, optimizer, train_dataloader, dev_dataloader,)
        # prepare all objects for distributed training and mixed precision
    
    NUM_UPDATES_PER_EPOCH = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps) # 920 / 1 = 920 = 58880 / 128
        # i.e., len(train_dataset) = 58880를 batch_size = 32 * 4 = 128 로 돌면 한 epoch 당 460개의 batch가 있는데, 매 batch마다 gradient update 
    MAX_TRAIN_STEPS = NUM_UPDATES_PER_EPOCH * args.max_train_epochs # 460 * 40 = 18400: # of total batch (total epoch)
    MAX_TRAIN_EPOCHS = math.ceil(MAX_TRAIN_STEPS / NUM_UPDATES_PER_EPOCH) # 40
    TOTAL_TRAIN_BATCH_SIZE = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps # 32 * 4 * 1 = 128
    EVAL_STEPS = args.val_check_interval if isinstance(args.val_check_interval,int) else int(args.val_check_interval * NUM_UPDATES_PER_EPOCH) # 1
        # i.e., gradient update 시마다 eval
    lr_scheduler = get_linear_scheduler(optimizer,warmup_steps=args.warmup_steps,total_training_steps=MAX_TRAIN_STEPS)

    logger.info("***** Running training *****")
    logger.info(f"  Num train examples = {len(train_dataset)}") # 58880
    logger.info(f"  Num dev examples = {len(dev_dataset)}") # 6515
    logger.info(f"  Num Epochs = {MAX_TRAIN_EPOCHS}") # 40
    logger.info(f"  Per device train batch size = {args.per_device_train_batch_size}") # 32
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {TOTAL_TRAIN_BATCH_SIZE}") # 128
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}") # 1
    logger.info(f"  Total optimization steps = {MAX_TRAIN_STEPS}") # 18400
    logger.info(f"  Per device eval batch size = {args.per_device_eval_batch_size}") # 32
    completed_steps = 0
    progress_bar = tqdm(range(MAX_TRAIN_STEPS), disable=not accelerator.is_local_main_process,ncols=100)

    for epoch in range(MAX_TRAIN_EPOCHS):
        set_seed(args.seed+epoch)
        progress_bar.set_description(f"epoch: {epoch+1}/{MAX_TRAIN_EPOCHS}")
        for step,batch in enumerate(train_dataloader):
            with accelerator.accumulate(dual_encoder): # synchronize gradient only when the optimizer.step() is called to avoid overhead
                with accelerator.autocast(): # automatic mixed precision
                    query_embedding,doc_embedding  = dual_encoder(**batch)
                    dpr_query_embedding, dpr_doc_embedding, lex_query_embedding, lex_doc_embedding = spar_encoder(**batch)
        
                    single_device_query_num,_ = query_embedding.shape
                    single_device_doc_num,_ = doc_embedding.shape
                    if accelerator.use_distributed:
                        query_list = [torch.zeros_like(query_embedding) for _ in range(accelerator.num_processes)] # accelerator.num_processes = # of gpus
                        dist.all_gather(tensor_list=query_list, tensor=query_embedding.contiguous()) # 모든 프로세스에서 tensor를 수집해 query_list에 저장
                        query_list[dist.get_rank()] = query_embedding
                        query_embedding = torch.cat(query_list, dim=0)

                        doc_list = [torch.zeros_like(doc_embedding) for _ in range(accelerator.num_processes)]
                        dist.all_gather(tensor_list=doc_list, tensor=doc_embedding.contiguous())
                        doc_list[dist.get_rank()] = doc_embedding
                        doc_embedding = torch.cat(doc_list, dim=0)

                        dpr_query_list = [torch.zeros_like(dpr_query_embedding) for _ in range(accelerator.num_processes)]
                        dist.all_gather(tensor_list=dpr_query_list, tensor=dpr_query_embedding.contiguous())
                        dpr_query_list[dist.get_rank()] = dpr_query_embedding
                        dpr_query_embedding = torch.cat(dpr_query_list, dim=0)

                        dpr_doc_list = [torch.zeros_like(dpr_doc_embedding) for _ in range(accelerator.num_processes)]
                        dist.all_gather(tensor_list=dpr_doc_list, tensor=dpr_doc_embedding.contiguous())
                        dpr_doc_list[dist.get_rank()] = dpr_doc_embedding
                        dpr_doc_embedding = torch.cat(dpr_doc_list, dim=0)

                        lex_query_list = [torch.zeros_like(lex_query_embedding) for _ in range(accelerator.num_processes)]
                        dist.all_gather(tensor_list=lex_query_list, tensor=lex_query_embedding.contiguous())
                        lex_query_list[dist.get_rank()] = lex_query_embedding
                        lex_query_embedding = torch.cat(lex_query_list, dim=0)

                        lex_doc_list = [torch.zeros_like(lex_doc_embedding) for _ in range(accelerator.num_processes)]
                        dist.all_gather(tensor_list=lex_doc_list, tensor=lex_doc_embedding.contiguous())
                        lex_doc_list[dist.get_rank()] = lex_doc_embedding
                        lex_doc_embedding = torch.cat(lex_doc_list, dim=0)

                    matching_score = torch.matmul(query_embedding,doc_embedding.permute(1,0))
                    labels = torch.cat([torch.arange(single_device_query_num) + gpu_index * single_device_doc_num for gpu_index in range(accelerator.num_processes)],dim=0).to(matching_score.device)
                    large_score = get_large_score(dpr_query_embedding, dpr_doc_embedding, lex_query_embedding, lex_doc_embedding).to(matching_score.device)

                    loss = calculate_kd_loss(matching_score, large_score, labels=labels)

                accelerator.backward(loss)

                ## one optimization step
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    progress_bar.set_postfix(loss=f"{loss:.4f}",lr=f"{lr_scheduler.get_last_lr()[0]:6f}")
                    completed_steps += 1
                    accelerator.clip_grad_norm_(dual_encoder.parameters(), args.max_grad_norm)
                    if not accelerator.optimizer_step_was_skipped:
                        lr_scheduler.step()
                    accelerator.log({"training_loss": loss}, step=completed_steps)
                    accelerator.log({"lr": lr_scheduler.get_last_lr()[0]}, step=completed_steps)
                    
                    if completed_steps % EVAL_STEPS == 0:
                        avg_rank,loss = validate(dual_encoder,dev_dataloader,accelerator)
                        dual_encoder.train()
                        accelerator.log({"avg_rank": avg_rank, "loss":loss}, step=completed_steps)
                        accelerator.wait_for_everyone() # 모든 process가 이 지점에 도달할 때까지 대기
                        if accelerator.is_local_main_process: # main process인 경우에만 모델 저장 (여러 process가 실행되지만 모델 저장은 한 번만 필요)
                            unwrapped_model = accelerator.unwrap_model(dual_encoder) # 모델 저장을 위한 전처리
                            unwrapped_model.query_encoder.save_pretrained(os.path.join(LOG_DIR,f"step-{completed_steps}/query_encoder"))
                            tokenizer.save_pretrained(os.path.join(LOG_DIR,f"step-{completed_steps}/query_encoder"))
                            
                            unwrapped_model.doc_encoder.save_pretrained(os.path.join(LOG_DIR,f"step-{completed_steps}/doc_encoder"))
                            tokenizer.save_pretrained(os.path.join(LOG_DIR,f"step-{completed_steps}/doc_encoder"))

                        accelerator.wait_for_everyone()
                
                optimizer.step() # gradient update
                optimizer.zero_grad()
    
    if accelerator.is_local_main_process:wandb_tracker.finish()
    accelerator.end_training()

if __name__ == '__main__':
    main()