"""
    Utility functions for MPI
    Taken from:
    * https://github.com/openai/spinningup/blob/master/spinup/utils/mpi_tools.py
    * https://github.com/openai/spinningup/blob/master/spinup/utils/mpi_pytorch.py
"""
import multiprocessing
import numpy as np
import os, subprocess, sys
import torch
from mpi4py import MPI

def setup_pytorch_for_mpi():
    """
        Avoid slowdowns caused by each separate process's PyTorch using
        more than its fair share of CPU resources.
    """
    if torch.get_num_threads()==1:
        return
    fair_num_threads = max(int(torch.get_num_threads() / num_procs()), 1)
    torch.set_num_threads(fair_num_threads)

def mpi_avg_grads(module):
    """
        Average contents of gradient buffers across MPI processes. 
    """
    if num_procs()==1:
        return
    for p in module.parameters():
        p_grad_numpy = p.grad.numpy()   # numpy view of tensor data
        avg_p_grad = mpi_avg(p.grad)
        p_grad_numpy[:] = avg_p_grad[:]

def sync_params(module):
    """
        Sync all parameters of module across all MPI processes.
    """
    if num_procs()==1:
        return
    for p in module.parameters():
        p_numpy = p.data.numpy()
        broadcast(p_numpy)

def mpi_fork(n:int, bind_to_core:bool=False):
    """
        Re-launches the current script with workers linked by MPI.

        Also, terminates the original process that launched it.

        Taken almost without modification from the Baselines function of the
        `same name`_.

        .. _`same name`: https://github.com/openai/baselines/blob/master/baselines/common/mpi_fork.py

        Parameters:
        -----------
        n (int): Number of process to split into.

        bind_to_core (bool): Bind each MPI process to a core.
    """
    if n<=1: 
        return
    if os.getenv("IN_MPI") is None:
        env = os.environ.copy()
        env.update(
            MKL_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            IN_MPI="1"
        )
        # args = ["mpirun", "-np", str(n)]
        args = ["mpirun"]
        if bind_to_core:
            args += ["-bind-to", "core"]
        args += [sys.executable] + sys.argv
        subprocess.check_call(args, env=env)
        sys.exit()

def msg(m, string=''):
    print(('Message from %d: %s \t '%(MPI.COMM_WORLD.Get_rank(), string))+str(m))

def proc_id():
    """Get rank of calling process."""
    return MPI.COMM_WORLD.Get_rank()

def allreduce(*args, **kwargs):
    return MPI.COMM_WORLD.Allreduce(*args, **kwargs)

def num_procs():
    """Count active MPI processes."""
    return MPI.COMM_WORLD.Get_size()

def broadcast(x, root=0):
    MPI.COMM_WORLD.Bcast(x, root=root)

def mpi_op(x, op):
    x, scalar = ([x], True) if np.isscalar(x) else (x, False)
    x = np.asarray(x, dtype=np.float32)
    buff = np.zeros_like(x, dtype=np.float32)
    allreduce(x, buff, op=op)
    return buff[0] if scalar else buff

def mpi_sum(x):
    return mpi_op(x, MPI.SUM)

def mpi_avg(x):
    """
        Average a scalar or vector over MPI processes.
    """
    return mpi_sum(x) / num_procs()
    
def mpi_statistics_scalar(x, with_min_and_max=False):
    """
        Get mean/std and optional min/max of scalar x across MPI processes.

        Parameters:
        -----------
        x: An array containing samples of the scalar to produce statistics
            for.

        with_min_and_max (bool): If true, return min and max of x in 
            addition to mean and std.
    """
    x = np.array(x, dtype=np.float32)
    global_sum, global_n = mpi_sum([np.sum(x), len(x)])
    mean = global_sum / global_n

    global_sum_sq = mpi_sum(np.sum((x - mean)**2))
    std = np.sqrt(global_sum_sq / global_n)  # compute global std

    if with_min_and_max:
        global_min = mpi_op(np.min(x) if len(x) > 0 else np.inf, op=MPI.MIN)
        global_max = mpi_op(np.max(x) if len(x) > 0 else -np.inf, op=MPI.MAX)
        return mean, std, global_min, global_max
    return mean, std