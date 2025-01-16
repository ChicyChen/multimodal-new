#!/bin/bash
# The interpreter used to execute the script
#“#SBATCH” directives that convey submission options:
#SBATCH --job-name=multimodal
#SBATCH --nodes=1
#SBATCH --mem=48GB
#SBATCH --time=02-00:00:00
#SBATCH --account=qingqu2
#SBATCH --partition=spgpu
#SBATCH --gpus=a40:1
#SBATCH --cpus-per-task=4
module purge
module load cuda/11.8.0 cudnn/11.8-v8.7.0
eval "$(conda shell.bash hook)"
conda activate mindgap
cd /scratch/qingqu_root/qingqu1/siyich/multimodal-gap

for i in $(seq 17 100);
do
    python train.py --TRAINER_CONFIG_PATH utils/train_config_7e-1.yaml --DATA_CONFIG_PATH dataloader/data_config_num2048.yaml \
                --saved_checkpoints nw_train_tiny2048_${i} --logs nw_train_tiny2048_${i} \
                --num_train_epochs 1
done
