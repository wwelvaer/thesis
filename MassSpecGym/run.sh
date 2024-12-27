#!/bin/bash
#PBS -N trainSmiles
#PBS -l gpus=1
#PBS -l walltime=8:00:00
#PBS -l mem=32gb

cd data/Thesis

source python3-11-venv/bin/activate

export WANDB_API_KEY="c98a88c3a37fb338089fe9d5d9b71abfc376a8d3"

python run.py --job_key 1 --run_name early-stop_topq-parallel_q3e-1 --max_epochs 50 --task de_novo --model smiles_transformer --log_only_loss_at_stages "train, val" --project_name "HPC" --check_val_every_n_epoch 1 --sampler top-q-parallel --patience 5 --q 0.3
