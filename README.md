## Knowledge Distillation: SPAR to DPR
train_spar.py 파일명 train_kd.py로 바꿨습니다!

## Running train_kd.py
```bash
accelerate launch train_kd.py
```

## Running doc2embedding_kd.py
### SPAR (for baseline)
```bash
#!/bin/bash
#SBATCH --job-name=spar
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=0-24:00:00
#SBATCH --mem=80000MB
#SBATCH --cpus-per-task=8

srun accelerate launch doc2embedding_kd.py \
    --encoding_batch_size 512 \
    --model_type spar \
    --output_dir embedding/spar
```