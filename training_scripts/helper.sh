#!/bin/bash

# to run single experiment
# Slurm sbatch options
# add more cpu tasks
#SBATCH -n 40
#SBATCH --exclusive

# Loading the required module
source /etc/profile
module load anaconda/2020b
module load mpi/openmpi-4.0
export PMIX_MCA_gds=hash

while getopts s:d:m:b: flag
do
    case "${flag}" in
        s) seed=${OPTARG};;
        d) density=${OPTARG};;
        m) moveup_weight=${OPTARG};;
        b) buffer_weight=${OPTARG};;
    esac
done

out_title=$(echo "training_scripts/exp_output/s${seed}_d${density}_m${moveup_weight}_b${buffer_weight}.out")
echo ${out_title}
mpirun python -m RL.ppo_exp --buffer_weight=${buffer_weight} --moveup_weight=${moveup_weight} --seed=${seed} --flight_density=${density} --cpu=50 --epochs=5000 &> ${out_title}
