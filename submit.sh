#!/bin/bash
#SBATCH --nodes=1
##SBATCH --constraint=gpu80
##SBATCH --cpus-per-task=4
#SBATCH -o 'logs/%A_%x.log'
#SBATCH --mail-user=timna.kleinman+DSBATCH@mail.huji.ac.il
#SBATCH --mail-type=END,FAIL

if [[ "$HOSTNAME" == *"tiger"* ]]
then
    echo "It's tiger"
    module load anaconda
    source activate 247-main
elif [[ "$HOSTNAME" == *"della"* ]]
then
    echo "It's Della"
    module load anaconda3/2021.11
    conda activate encoding
else
    module load anacondapy
    source activate srm
fi

export TRANSFORMERS_OFFLINE=1

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
