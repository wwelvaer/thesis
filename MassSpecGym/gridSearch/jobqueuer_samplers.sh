#!/bin/bash

PROJECT="HPC" #"SamplersGridSearch"

ml load CUDA-Python/12.1.0-gfbf-2023a-CUDA-12.1.1

python3 samplerparams.py | while IFS= read -r line; do
    JOBNAME=$(echo $line | cut -d "|" -f1)
    ARGS=$(echo $line | cut -d "|" -f2-)

    if [[ $JOBNAME == DEBUG* ]]; then
        # Handle debug lines
        echo $ARGS
    else
        echo $JOBNAME
        # Create a unique job script
        job_script=$(mktemp)
        cat <<EOF > $job_script
#!/bin/bash
#PBS -N $JOBNAME
#PBS -l walltime=4:00:00
#PBS -l gpus=1
#PBS -l mem=32gb

cd data/Thesis

source python3-11-venv/bin/activate

export WANDB_API_KEY="c98a88c3a37fb338089fe9d5d9b71abfc376a8d3"

python run.py --job_key 1 --run_name $JOBNAME --task de_novo --model smiles_transformer --log_only_loss_at_stages "train, test" --test_only --project_name $PROJECT $ARGS
EOF

        # Submit the job
        qsub $job_script

        # Clean up the temporary job script
        rm -f $job_script
    fi
done
