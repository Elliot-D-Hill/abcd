#!/bin/bash
#SBATCH --job-name=abcd     
#SBATCH --output=logs/output.txt
#SBATCH --error=logs/error.txt       
#SBATCH --time=08:00:00                     # Max run time (hh:mm:ss)
#SBATCH --partition=engelhardlab-gpu        # Partition to submit to
#SBATCH --account=engelhardlab              # Account name
#SBATCH --cpus-per-task=16                  # Number of CPUs per task
#SBATCH --mem=20G                           # CPU Memory
#### #SBATCH --mem-per-cpu=2G                    # CPU Memory
#SBATCH --gres=gpu:1                        # Request GPUs
#SBATCH --exclusive                         # Request exclusive node
#SBATCH --exclude=dcc-engelhardlab-gpu-03   # Exclude node(s)
#### #SBATCH --array=1-3
#### #SBATCH --nodelist=dcc-engelhardlab-gpu-03  # Request specific node


echo "Starting job $SLURM_JOB_ID..."
trap 'echo "Error on line $LINENO. Terminating job $SLURM_JOB_ID." >&2; exit 1' ERR
source /hpc/group/engelhardlab/edh47/miniconda3/etc/profile.d/conda.sh
conda activate venv
python3 -m abcd
conda deactivate
echo "Job $SLURM_JOB_ID complete."