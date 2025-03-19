#!/bin/bash
#SBATCH --time=22:30:00
#SBATCH --mem=300GB
#SBATCH --nodes=1
#SBATCH -o 'logs/%A.log'
#SBATCH -e 'logs/%A.err'
#SBATCH --mail-type=fail # send email if job fails
#SBATCH --mail-user=cc27@princeton.edu
#SBATCH --gres=gpu:1

if [[ "$HOSTNAME" == *"tiger"* ]]
then
    echo "It's tiger"
    module load anaconda
    source activate 247-main
elif [[ "$HOSTNAME" == *"della"* ]]
then
    echo "It's Della"
    module load anaconda3/2021.11
    source activate 247-main
    conda activate 247-main
else
    module load anacondapy
    source activate srm
fi

export TRANSFORMERS_OFFLINE=1
git checkout bridge
echo 'Requester:' $USER
echo 'Node:' $HOSTNAME
echo 'Start time:' `date`
echo "$@"
if [[ -v SLURM_ARRAY_TASK_ID ]]
then
    python "$@" --electrodes $SLURM_ARRAY_TASK_ID
else
    python "$@"
fi
echo 'End time:' `date`
