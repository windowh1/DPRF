from utils import normalize_query
import csv
import faiss,pickle
import json        
import numpy as np 
from tqdm import tqdm
from transformers import (
    DPRQuestionEncoder,
    DPRQuestionEncoderTokenizer,
    BertModel,
    BertTokenizer,
    AutoTokenizer,
    AutoModel,
    )
import torch
from utils.tokenizers import SimpleTokenizer
import unicodedata
import time
import transformers
transformers.logging.set_verbosity_error()
import wandb

def normalize(text):
    return unicodedata.normalize("NFD", text)

def has_answer(answers,doc):
    tokenizer = SimpleTokenizer()
    doc = tokenizer.tokenize(normalize(doc)).words(uncased=True)
    for answer in answers:
        answer = tokenizer.tokenize(normalize(answer)).words(uncased=True)
        for i in range(0, len(doc) - len(answer) + 1):
                if answer == doc[i : i + len(answer)]:
                    return True
    return False

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--wikipedia_path",default="downloads/data/wikipedia_split/psgs_w100.tsv")
    parser.add_argument("--nq_test_file",default="downloads/data/retriever/qas/nq-test.csv")
    parser.add_argument("--top_k",default=100)
    parser.add_argument("--model_type",required=True)
    parser.add_argument("--pyserini_index",default="downloads/lucene-index-wikipedia-dpr-100w-20210120-d1b9e6/")
    parser.add_argument("--pretrained_model_path",default=None)
    parser.add_argument("--doc_embedding_dir",default=None)
    parser.add_argument("--encoding_batch_size",type=int,default=32)
    parser.add_argument("--num_shards",type=int,default=36)
    parser.add_argument("--num_docs",type=int,default=21015324)
    parser.add_argument("--java_home", default=None)
    args = parser.parse_args()

    wandb.init(project="knowledge-distillation", job_type="Inference",name="test_"+args.model_type)

    ## load QA dataset
    query_col,answers_col=0,1
    queries,answers = [],[]
    with open(args.nq_test_file) as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            queries.append(normalize_query(row[query_col]))
            answers.append(eval(row[answers_col]))
    
    if args.model_type == "bm25":
        import os
        os.environ['JAVA_HOME'] = args.java_home
        os.environ['PATH'] = f"{os.environ.get('PATH')}:{os.environ.get('JAVA_HOME')}/bin"
        from pyserini.search.lucene import LuceneSearcher

        searcher = LuceneSearcher(args.pyserini_index)
        
        hit_lists = []
        I = []
        for i, query in enumerate(queries):
            hit_list = []
            id_list = []
            answer = answers[i]
            search_results = searcher.search(query, args.top_k)
            for j in range(args.top_k):
                id_list.append(search_results[j].docid)
                title_contents = search_results[j].lucene_document
                raw = title_contents.get('raw')
                json_data = json.loads(raw)
                _, doc = json_data['contents'].split(sep='\n')
                hit_list.append(has_answer(answer,doc))
            hit_lists.append(hit_list)
            I.append(id_list)

    else:
        queries = [queries[idx:idx+args.encoding_batch_size] for idx in range(0,len(queries),args.encoding_batch_size)]
        
        if args.model_type == "spar":
            embedding_dimension = 1536
        else:
            embedding_dimension = 768 
        index = faiss.IndexFlatIP(embedding_dimension)
        for idx in tqdm(range(args.num_shards),desc='building index from embedding...'):
            data = np.load(f"{args.doc_embedding_dir}/wikipedia_shard_{idx}.npy")
            index.add(data)  

        ## load wikipedia passages
        id_col,text_col,title_col=0,1,2
        wiki_passages = []
        with open(args.wikipedia_path) as f:
            reader = csv.reader(f, delimiter="\t")
            for row in tqdm(reader,total=args.num_docs,desc="loading wikipedia passages..."):
                if row[id_col] == "id":continue
                wiki_passages.append(row[text_col].strip('"'))

        device = "cuda" if torch.cuda.is_available() else "cpu"    
        ## load query encoder
        if args.model_type == 'spar':
            dpr_query_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-multiset-base")
            dpr_query_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-multiset-base")
            dpr_query_encoder.to(device).eval()

            lex_tokenizer = AutoTokenizer.from_pretrained('facebook/spar-wiki-bm25-lexmodel-query-encoder')
            lex_query_encoder = AutoModel.from_pretrained('facebook/spar-wiki-bm25-lexmodel-query-encoder')
            lex_query_encoder.to(device).eval()

        else:
            tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_path)
            query_encoder = BertModel.from_pretrained(args.pretrained_model_path,add_pooling_layer=False)
            query_encoder.to(device).eval()

        ## embed queries
        CLS_POS = 0
        query_embeddings = []
        for query in tqdm(queries,desc='encoding queries...'):
            with torch.no_grad():
                if args.model_type == 'spar':
                    dpr_query_embedding = dpr_query_encoder(**dpr_query_tokenizer(query,max_length=256,truncation=True,padding='max_length',return_tensors='pt').to(device)).pooler_output
                    lex_query_embedding = lex_query_encoder(**lex_tokenizer(query,max_length=256,truncation=True,padding='max_length',return_tensors='pt').to(device)).last_hidden_state[:,CLS_POS,:]
                    query_embedding = torch.cat([dpr_query_embedding, lex_query_embedding], dim=1)
                else:
                    query_embedding = query_encoder(**tokenizer(query,max_length=256,truncation=True,padding='max_length',return_tensors='pt').to(device))
                    if isinstance(query_encoder,DPRQuestionEncoder):
                        query_embedding = query_embedding.pooler_output
                    else:
                        query_embedding = query_embedding.last_hidden_state[:,CLS_POS,:]
            query_embeddings.append(query_embedding.cpu().detach().numpy())
        query_embeddings = np.concatenate(query_embeddings,axis=0)

        ## retrieve top-k documents
        print("searching index ",end=' ')
        start_time = time.time()
        _,I = index.search(query_embeddings,args.top_k)
        print(f"takes {time.time()-start_time} s")
        wandb.log({"search_duration": time.time()-start_time})

        hit_lists = []
        for answer_list,id_list in tqdm(zip(answers,I),total=len(answers),desc='calculating metrics...'):
            ## process single query
            hit_list = []
            for doc_id in id_list:
                doc = wiki_passages[doc_id]
                hit_list.append(has_answer(answer_list,doc))
            hit_lists.append(hit_list)

    out_file = "retrieval/"+args.model_type+".pickle"
    with open(file=out_file, mode='wb') as f:
        pickle.dump(I, f)

    top_k_hits = [0]*args.top_k
    best_hits = []
    for hit_list in hit_lists:
        best_hit = next((i for i, x in enumerate(hit_list) if x), None)
        if best_hit is not None:
            top_k_hits[best_hit:] = [v + 1 for v in top_k_hits[best_hit:]]
    
    top_k_ratio = [x/len(answers) for x in top_k_hits] # top_x 안에 answer이 있는 qa의 ratio
    
    for idx in range(args.top_k):
        if (idx+1) % 10 == 0:
            print(f"top-{idx+1} accuracy",top_k_ratio[idx])
            wandb.log({(f"top-{idx+1} accuracy",top_k_ratio[idx]): top_k_ratio[idx]})
            

    wandb.finish()