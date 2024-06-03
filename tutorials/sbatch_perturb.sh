#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --job-name=scGPT-large-perturb
#SBATCH --mem=120GB
#SBATCH --gres=gpu:1
#SBATCH --partition=a100
#SBATCH --qos=a100_bowang
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=3-00:00:00
#SBATCH --mail-user=subercui@gmail.com
#SBATCH --mail-type=ALL

#--nodelist=gpu182,gpu188
#--exclude=gpu183,gpu184,gpu186,gpu187,gpu189

# log the sbatch environment
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_JOB_PARTITION"=$SLURM_JOB_PARTITION
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURM_GPUS_ON_NODE"=$SLURM_GPUS_ON_NODE
echo "SLURM_SUBMIT_DIR"=$SLURM_SUBMIT_DIR
export NCCL_IB_DISABLE=1

# . /etc/profile.d/lmod.sh
bash ~/.bashrc
nvcc --version

# >>> conda initialize >>>
__conda_setup="$('/pkgs/anaconda3/bin/conda' 'shell.bash' 'hook' 2>/dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/pkgs/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/pkgs/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/pkgs/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
# PATH=~/.conda/envs/r-env/bin/:$PATH

conda activate ~/.conda/envs/scgptpy310
which python

cd /fs01/home/haotian/scGPT/tutorials
python tutorial_perturbation.py
