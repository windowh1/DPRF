import numpy as np
import pandas as pd
import pickle
import wandb

def rbo_list(s, t, p=0.9):
    list = []
    for d in range(len(s)):
        s_d = set(s[:d+1])
        t_d = set(t[:d+1])
        x_d = len(s_d.intersection(t_d))
        list.append((p**(d+1)) * x_d / (d+1))
    return list

def rbo_list_matrix(rank_list, args, p=0.9):
    n = len(rank_list)
    matrix = [[[0.0] * args.top_k for _ in range(n)] for _ in range(n)]

    for i, rank1 in enumerate(rank_list):
        for j, rank2 in enumerate(rank_list):
            if i < j:
                matrix[i][j] = rbo_list(rank1, rank2, p)
            elif i > j:
                matrix[i][j] = matrix[j][i]
    
    return matrix

def calculate_rbo_ext(rbo_list, k, p=0.9):
    rbo_ext = rbo_list[k-1] + (1-p) / p * sum(rbo_list[:k])
    return rbo_ext

def rbo_ext_matrix(rbo_list_matrix, k, p=0.9):
    n = len(rbo_list_matrix)
    matrix = [[0.0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i < j:
                matrix[i][j] = calculate_rbo_ext(rbo_list_matrix[i][j], k, p)
            elif i > j:
                matrix[i][j] = matrix[j][i]
    
    return matrix

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--bm25_rank",default="retrieval/bm25.pickle")
    parser.add_argument("--dpr_rank",default="retrieval/nanoDPR.pickle")
    parser.add_argument("--spar_rank",default="retrieval/spar.pickle")
    parser.add_argument("--kd_rank",default="retrieval/kd.pickle")

    args = parser.parse_args()
    wandb.init(project="knowledge-distillation", job_type="Inference",name="rbo-calculate")

    with open(args.bm25_rank, 'rb') as f:
        bm25_ranks = pickle.load(f)

    with open(args.dpr_rank, 'rb') as f:
        dpr_ranks = pickle.load(f)

    with open(args.spar_rank, 'rb') as f:
        spar_ranks = pickle.load(f)

    with open(args.kd_rank, 'rb') as f:
        kd_ranks = pickle.load(f)

    rank_title = ["BM25", "DPR", "SPAR", "KD"]
    rbo_matrixs_top5 = []
    rbo_matrixs_top20 = []
    rbo_matrixs_top100 = []

    for bm25_rank, dpr_rank, spar_rank, kd_rank in zip(bm25_ranks, dpr_ranks, spar_ranks, kd_ranks):        
        rbo_matrix = rbo_list_matrix([bm25_rank, dpr_rank, spar_rank, kd_rank], args)
        rbo_matrixs_top5.append(np.array(rbo_ext_matrix(rbo_matrix, k=5)))
        rbo_matrixs_top20.append(np.array(rbo_ext_matrix(rbo_matrix, k=20)))
        rbo_matrixs_top100.append(np.array(rbo_ext_matrix(rbo_matrix, k=100)))

    avg_rbo_top5 = np.array(rbo_matrixs_top5).mean(axis=0)
    avg_rbo_top20 = np.array(rbo_matrixs_top20).mean(axis=0)
    avg_rbo_top100 = np.array(rbo_matrixs_top100).mean(axis=0)

    rbo_ext_top5 = wandb.Table(pd.Dataframe(data = avg_rbo_top5,
                                            columns=rank_title,
                                            index=rank_title))   
    wandb.log({"rbd_ext: top-5", rbo_ext_top5})

    rbo_ext_top20 = wandb.Table(pd.Dataframe(data = avg_rbo_top20,
                                             columns=rank_title,
                                             index=rank_title))    
    wandb.log({"rbo_ext: top-20", rbo_ext_top20})

    rbo_ext_top100 = wandb.Table(pd.Dataframe(data = avg_rbo_top100,
                                              columns=rank_title,
                                              index=rank_title))    
    wandb.log({"rbo_ext: top-100", rbo_ext_top100})

    wandb.finish()
