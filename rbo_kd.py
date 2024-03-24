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

def rbo_list_matrix(rank_list, p=0.9):
    n = len(rank_list)
    matrix = [[None for _ in range(n)] for _ in range(n)]

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
    matrix = [[None for _ in range(n)] for _ in range(n)]

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
    parser.add_argument("--dpr_rank",default="retrieval/dpr.pickle")
    parser.add_argument("--spar_rank",default="retrieval/spar.pickle")
    parser.add_argument("--kd_rank",default="retrieval/kd.pickle")

    args = parser.parse_args()
    wandb.init(project="knowledge-distillation", job_type="Inference",name="rbo-calculate")

    with open(args.bm25_rank, 'rb') as f:
        bm25_rank = pickle.load(f)

    with open(args.dpr_rank, 'rb') as f:
        dpr_rank = pickle.load(f)

    with open(args.spar_rank, 'rb') as f:
        spar_rank = pickle.load(f)

    with open(args.kd_rank, 'rb') as f:
        kd_rank = pickle.load(f)

    rank_list = [bm25_rank, dpr_rank, spar_rank, kd_rank]
    rank_title = ["BM25", "DPR", "SPAR", "KD"]

    rbo_list_matrix = rbo_list_matrix(rank_list)

    rbo_ext_matrix_top5 = np.array(rbo_ext_matrix(rbo_list_matrix, k=5))
    rbo_ext_matrix_top20 = np.array(rbo_ext_matrix(rbo_list_matrix, k=20))
    rbo_ext_matrix_top100 = np.array(rbo_ext_matrix(rbo_list_matrix, k=100))

    rbo_ext_top5 = wandb.Table(pd.Dataframe(data = rbo_ext_matrix_top5,
                                            columns=rank_title,
                                            index=rank_title))   
    wandb.log({"rbd_ext: top-5", rbo_ext_top5})

    rbo_ext_top20 = wandb.Table(pd.Dataframe(data = rbo_ext_matrix_top20,
                                             columns=rank_title,
                                             index=rank_title))    
    wandb.log({"rbo_ext: top-20", rbo_ext_top20})

    rbo_ext_top100 = wandb.Table(pd.Dataframe(data = rbo_ext_matrix_top100,
                                              columns=rank_title,
                                              index=rank_title))    
    wandb.log({"rbo_ext: top-100", rbo_ext_top100})

    wandb.finish()
