# Knowledge Distillation for DPR

This repository implements **knowledge distillation** for the **Dense Passage Retriever (DPR)**, as proposed by Karpukhin et al., 2020. In this project, knowledge is distilled from **SPAR** (Chen et al., 2021), a retriever model with double the capacity of DPR, into the original DPR model.

The DPR codebase utilized here is based on [nanoDPR](https://github.com/Hannibal046/nanoDPR).

## References

- Chen et al., 2021. Salient Phrase Aware Dense Retrieval: Can a Dense Retriever Imitate a Sparse One?
- Karpukhin et al., 2020. Dense Passage Retrieval for Open-Domain Question Answering.
- [nanoDPR](https://github.com/Hannibal046/nanoDPR): A lightweight implementation of DPR.

## Usage

### Running train_kd.py
```bash
accelerate launch train_kd.py
```
