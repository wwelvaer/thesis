#!/bin/bash
#PBS -N trainSmiles
#PBS -l gpus=1
#PBS -l walltime=12:00:00
#PBS -l mem=32gb

cd data/Thesis

source python3-11-venv/bin/activate

export WANDB_API_KEY="c98a88c3a37fb338089fe9d5d9b71abfc376a8d3"

python run.py --job_key 1 --run_name selfies_transformer_paper --max_epochs 100 --task de_novo --model smiles_transformer --log_only_loss_at_stages "train, val" --project_name "HPC" --check_val_every_n_epoch 1 --sampler naive --patience 5 --lr 0.0003 --batch_size 1024 --d_model 256 --nhead 8 --num_encoder_layers 6 --temperature 1.0 --smiles_tokenizer selfies
