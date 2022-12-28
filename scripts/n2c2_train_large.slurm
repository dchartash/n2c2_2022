#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=n2c2_lg_basic
#SBATCH --out="slurm-%j.out"
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1 
#SBATCH --ntasks=1
#SBATCH --mem=95G
#SBATCH --gpus=1
#SBATCH --mail-type=ALL

mem_bytes=$(</sys/fs/cgroup/memory/slurm/uid_${SLURM_JOB_UID}/job_${SLURM_JOB_ID}/memory.limit_in_bytes)
mem_gbytes=$(( $mem_bytes / 1024 **3 ))

echo "Starting at $(date)"
echo "Job submitted to the ${SLURM_JOB_PARTITION} partition, the default partition on ${SLURM_CLUSTER_NAME}"
echo "Job name: ${SLURM_JOB_NAME}, Job ID: ${SLURM_JOB_ID}"
echo "  I have ${SLURM_CPUS_ON_NODE} CPUs and ${mem_gbytes}GiB of RAM on compute node $(hostname)"

module load CUDA/11.1.1-GCC-10.2.0
module load miniconda
conda activate n2c2_env2

cd /home/vs428/Documents/n2c2_2022

python ./trainer.py model=roberta_large
