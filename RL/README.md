## Reinforcement Learning folder

### File:
* `config.py`: Has the configuration parameters for both the pilotRL environment and the RL experiment with Proximal Policy Optimisation(PPO)
* `logger.py`: Has a WandB wrapper to be used as a logger for logging experiment metrics.
* `model.py`: The definition of the actor-critic networks for PPO. The arguments of the network instantiation will take care of the input, output and the hidden layers.
* `mpi_utils.py`: Has utility functions for multiprocessing (MPI/mpi4py). Can run multiple copies of the environment in parallel fashion for faster training. NOTE: have not used GPU for torch model as I was not sure about the overhead in transferring stuff from multiple CPUs to GPUs.
* `ppo_exp.py`: The file to run experiment for pilotRL with the pilotRL environment.
* `spinup_ppo.py`: Almost similar to `ppo_exp.py` but just to use for checking the PPO implementation on simple OpenAI gym environments like `HalfCheetah-v2`, etc. The reward does increase to around 4000 (check `test_13_19_05-05_01_2021` in wandboard)
* `utils.py`: Misc helper functions for running experiments.

### Usage:
Techinically, for enabling multiprocessing we need to use the `mpirun` command. To run it with single CPU execute:

`python -m RL.ppo_exp` or `python RL/ppo_exp.py` from the scripts folder.