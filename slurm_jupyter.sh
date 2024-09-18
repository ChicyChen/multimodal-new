#!/bin/bash
#SBATCH --account=qingqu2
#SBATCH --job-name=jupyter
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --mem=48GB
#SBATCH --partition=spgpu
#SBATCH --gpus=a40:1
#SBATCH --cpus-per-task=4


module purge
module load cuda/11.8.0 cudnn/11.8-v8.7.0
source /home/siyich/miniconda3/etc/profile.d/conda.sh
conda activate mindgap

XDG_RUNTIME_DIR=""
port=$(shuf -i8000-9999 -n1)
node=$(hostname -s)
user=$(whoami)
cluster=$(hostname -f | awk -F"." '{print $2}')

# print tunneling instructions jupyter-log
echo -e "
MacOS or linux terminal command to create your ssh tunnel
ssh -N -L ${port}:${node}:${port} ${user}@greatlakes.arc-ts.umich.edu

Windows MobaXterm info
Forwarded port:same as remote port
Remote server: ${node}
Remote port: ${port}
SSH server: ${cluster}.arc-ts.umich.edu
SSH login: $user
SSH port: 22

Use a Browser on your local machine to go to:
localhost:${port}  (prefix w/ https:// if using password)
"

# load modules or conda environments here
# uncomment the following two lines to use your conda environment called notebook_env
# module load miniconda
# source activate /scratch/qingqu_root/qingqu1/zzekai/orthinit

# DON'T USE ADDRESS BELOW.
# DO USE TOKEN BELOW

# cat /etc/hosts
jupyter lab --no-browser --port=${port} --ip=${node}

