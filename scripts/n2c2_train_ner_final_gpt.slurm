#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=gptneo
#SBATCH --out="slurm-%j.out"
#SBATCH --time=0-2:00:00
#SBATCH --nodes=1 
#SBATCH --ntasks=1
#SBATCH --mem=60G
#SBATCH --gpus=rtx5000:1
#SBATCH --mail-type=ALL

mem_bytes=$(</sys/fs/cgroup/memory/slurm/uid_${SLURM_JOB_UID}/job_${SLURM_JOB_ID}/memory.limit_in_bytes)
mem_gbytes=$(( $mem_bytes / 1024 **3 ))

echo "Starting at $(date)"
echo "Job submitted to the ${SLURM_JOB_PARTITION} partition, the default partition on ${SLURM_CLUSTER_NAME}"
echo "Job name: ${SLURM_JOB_NAME}, Job ID: ${SLURM_JOB_ID}"
echo "  I have ${SLURM_CPUS_ON_NODE} CPUs and ${mem_gbytes}GiB of RAM on compute node $(hostname)"


module load miniconda
module load CUDA/11.3.1
conda activate n2c2_env3

cd /home/vs428/Documents/n2c2_2022


python ./gpt_inference.py model=gptneo train.add_ner=false train.add_ner_end=false train.batch_size=1 train.so_sections=false train.fast_dev_run=false
