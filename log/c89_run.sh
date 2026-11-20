#!/bin/bash
# c89 full run; resumable: stage 1 scores are cached in results/c89_fullfold_vectors.npz
PR=/mnt/data2/Toan/Causal_ML_XAI/Causal_XAI_v3/P3_clear
cd $PR
rm -f log/c89_exit_code.txt
/mnt/data2/Toan/Causal_ML_XAI/.venv/bin/python -u code/c89_fullfold_artifact.py > log/c89_full.log 2>&1
echo $? > log/c89_exit_code.txt
date '+%Y-%m-%dT%H:%M:%S%z' >> log/c89_exit_code.txt
