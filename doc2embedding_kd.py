import csv
from tqdm import tqdm
import os
import time
import transformers
transformers.logging.set_verbosity_error()
from transformers import (
    DPRContextEncoder,
    DPRContextEncoderTokenizer,
    BertTokenizer,
    BertModel,
    AutoTokenizer,
    AutoModel,
    )
import torch
import numpy as np
import wandb
from accelerate import PartialState

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--wikipedia_path",default="downloads/data/wikipedia_split/psgs_w100.tsv")
    parser.add_argument("--num_docs",type=int,default=21015324)
    parser.add_argument("--encoding_batch_size",type=int,default=1024)
    parser.add_argument("--concat_weight",type=float,default=0.7)
    parser.add_argument("--model_type",required=True)
    parser.add_argument("--pretrained_model_path",default=None)
    parser.add_argument("--output_dir",required=True)
    args = parser.parse_args()
    
    distributed_state = PartialState()
    device = distributed_state.device

    if distributed_state.is_main_process:
        wandb.init(project="knowledge-distillation",job_type="Inference",name="doc2embedding_"+args.model_type)
        start_time = time.time()

    ## load encoder
    if args.model_type == 'spar':
        dpr_doc_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-multiset-base")
        dpr_doc_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-multiset-base")
        dpr_doc_encoder.eval()
        dpr_doc_encoder.to(device)

        lex_tokenizer = AutoTokenizer.from_pretrained('facebook/spar-wiki-bm25-lexmodel-query-encoder')
        lex_doc_encoder = AutoModel.from_pretrained('facebook/spar-wiki-bm25-lexmodel-context-encoder')
        lex_doc_encoder.eval()
        lex_doc_encoder.to(device)

    elif args.model_type == 'dpr-nq-base':
        tokenizer = DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
        doc_encoder = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
        doc_encoder.eval()
        doc_encoder.to(device)

    else:
        tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_path)
        doc_encoder = BertModel.from_pretrained(args.pretrained_model_path,add_pooling_layer=False)
        doc_encoder.eval()
        doc_encoder.to(device)

    ## load wikipedia passages
    progress_bar = tqdm(total=args.num_docs, disable=not distributed_state.is_main_process,ncols=100,desc='loading wikipedia...')
    id_col,text_col,title_col=0,1,2
    wikipedia = []
    with open(args.wikipedia_path) as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if row[id_col] == "id":continue
            wikipedia.append(
                [row[title_col],row[text_col].strip('"')]
            )
            progress_bar.update(1)

    with distributed_state.split_between_processes(wikipedia) as sharded_wikipedia:
        
        sharded_wikipedia = [sharded_wikipedia[idx:idx+args.encoding_batch_size] for idx in range(0,len(sharded_wikipedia),args.encoding_batch_size)]
        encoding_progress_bar = tqdm(total=len(sharded_wikipedia), disable=not distributed_state.is_main_process,ncols=100,desc='encoding wikipedia...')
        doc_embeddings = []
        
        for data in sharded_wikipedia:
            title = [x[0] for x in data]
            passage = [x[1] for x in data]            
            CLS_POS = 0
            with torch.no_grad():
                if args.model_type == 'spar':
                    dpr_doc_input = dpr_doc_tokenizer(title,passage,padding=True,return_tensors='pt',truncation=True).to(device)
                    dpr_doc_emb = dpr_doc_encoder(**dpr_doc_input).pooler_output
                    lex_doc_input = lex_tokenizer(title,passage,padding=True,return_tensors='pt',truncation=True).to(device)                    
                    lex_doc_emb = lex_doc_encoder(**lex_doc_input).last_hidden_state[:,CLS_POS,:]
                    output = torch.cat([dpr_doc_emb, args.concat_weight * lex_doc_emb], dim=1)
                else:
                    model_input = tokenizer(title,passage,max_length=256,padding='max_length',return_tensors='pt',truncation=True).to(device)
                    if isinstance(doc_encoder,BertModel):
                        output = doc_encoder(**model_input).last_hidden_state[:,CLS_POS,:].cpu().numpy()
                    else:
                        output = doc_encoder(**model_input).pooler_output.cpu().numpy()
            doc_embeddings.append(output)
            encoding_progress_bar.update(1)
        doc_embeddings = np.concatenate(doc_embeddings,axis=0)
        os.makedirs(args.output_dir,exist_ok=True)
        np.save(f'{args.output_dir}/wikipedia_shard_{distributed_state.process_index}.npy',doc_embeddings)
    
    if distributed_state.is_main_process:
        wandb.log({"doc2embedding_duration": time.time()-start_time})
        wandb.finish()
