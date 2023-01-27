#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=vs_infer
#SBATCH --out="slurm-%j.out"
#SBATCH --time=0-2:00:00
#SBATCH --nodes=1 
#SBATCH --ntasks=1
#SBATCH --mem=60G
#SBATCH --gpus=1
#SBATCH --mail-type=ALL

mem_bytes=$(</sys/fs/cgroup/memory/slurm/uid_${SLURM_JOB_UID}/job_${SLURM_JOB_ID}/memory.limit_in_bytes)
mem_gbytes=$(( $mem_bytes / 1024 **3 ))

echo "Starting at $(date)"
echo "Job submitted to the ${SLURM_JOB_PARTITION} partition, the default partition on ${SLURM_CLUSTER_NAME}"
echo "Job name: ${SLURM_JOB_NAME}, Job ID: ${SLURM_JOB_ID}"
echo "  I have ${SLURM_CPUS_ON_NODE} CPUs and ${mem_gbytes}GiB of RAM on compute node $(hostname)"


module load miniconda
conda activate n2c2_env4

cd /home/vs428/Documents/n2c2_2022

ENTAILMENT="/home/vs428/project/n2c2_models/roberta_large_ner_so/roberta_large_ner_so/outputs/roberta_large/checkpoint-384/"
echo $ENTAILMENT

python ./final_inference.py model=roberta_large pretrained.entail_model=$ENTAILMENT train.add_ner=true train.add_ner_end=false train.batch_size=6 train.so_sections=true train.add_ner_so=false train.fast_dev_run=false save.save_model=false