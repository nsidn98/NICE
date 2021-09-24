#!/bin/bash

export WANDB_MODE=offline

# Run the script        
# script to iterate through different hyperparameters
seeds=(0 1 2 3 4)
densities=(1 2 3)
moveup_weights=(0 1)
# Run the script
for i in ${!seeds[@]}; do
    # run for different seeds
    for j in ${!densities[@]}; do
        # run for different densities
        for k in ${!moveup_weights[@]}; do
            seed=${seeds[$i]}
            density=${densities[$j]}
            moveup_weight=${moveup_weights[$k]}
            buffer_weight=$(expr 1 - $moveup_weight)
            # run for all different envs
            sbatch training_scripts/helper.sh -s ${seed} -d ${density} -m ${moveup_weight} -b ${buffer_weight}
        done
    done
done