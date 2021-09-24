import numpy as np
import torch
import scipy.signal
import argparse
import requests
import inspect
import functools
from mpi4py import MPI

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])
    
def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def discount_cumsum(x, discount:float):
    """
        Magic from rllab for computing discounted cumulative sums of vectors.
        input: 
            vector x, [x0, x1, x2]
        output:
            [x0 + discount * x1 + discount^2 * x2,  
            x1 + discount * x2,
            x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

def store_args(method):
    """
        https://stackoverflow.com/questions/6760536/python-iterating-through-constructors-arguments
        https://github.com/openai/baselines/blob/master/baselines/her/util.py
        Stores provided method args as instance attributes.
        Usage:
        ------
        class A:
            @store_args
            def __init__(self, a, b, c=3, d=4, e=5):
                pass

        a = A(1,2)
        print(a.a, a.b, a.c, a.d, a.e)
        >>> 1 2 3 4 5
    """
    argspec = inspect.getfullargspec(method)
    defaults = {}
    if argspec.defaults is not None:
        defaults = dict(
            zip(argspec.args[-len(argspec.defaults):], argspec.defaults))
    if argspec.kwonlydefaults is not None:
        defaults.update(argspec.kwonlydefaults)
    arg_names = argspec.args[1:]

    @functools.wraps(method)
    def wrapper(*positional_args, **keyword_args):
        self = positional_args[0]
        # Get default arg values
        args = defaults.copy()
        # Add provided arg values
        for name, value in zip(arg_names, positional_args[1:]):
            args[name] = value
        args.update(keyword_args)
        self.__dict__.update(args)
        return method(*positional_args, **keyword_args)

    return wrapper

def print_dash(num_dash:int=50):
    """
        Print "______________________"
        for num_dash times
    """
    print('_'*num_dash)

def print_box(text, num_dash:int=50):
    """
        Prints stuff with two print_dash, one above and one below
        Example:
            _________________
            text
            _________________
    """
    print_dash(num_dash)
    print(text)
    print_dash(num_dash)

def print_args(args:argparse.Namespace, box_dist:int=50):
    """
        Print the args in a pretty table
    """
    print_dash(box_dist)
    half_len = int((box_dist-len("Arguments")-5)/2)
    print("||" + " "*half_len + "Arguments" + " "*half_len + " ||")
    print_dash(box_dist)
    for k, v in vars(args).items():
        len_line = len(f"{k}: {str(v)}")
        print("|| " + f"{k}: {str(v)}" + " "*(box_dist-len_line-5) + "||")
    print_dash(box_dist)

def connected_to_internet(url:str='http://www.google.com/', timeout:int=5):
    """
        Check if system is connected to the internet
    """
    try:
        _ = requests.get(url, timeout=timeout)
        return True
    except requests.ConnectionError:
        if MPI.COMM_WORLD.Get_rank() == 0:
            print("No internet connection available.")
    return False

def check_steps_per_MPI_task(args:argparse.Namespace, 
                            num_procs:int) -> argparse.Namespace:
    """
        If we request for a lot of MPI processes, 
        the number of steps executed in the environment 
        in each process is quite small.
        Number of steps per task = args.steps_per_epoch / num_procs()
        So this method will take care of that by increasing
        args.steps_per_epoch so that at least one episode is carried
        out completely in one task.
        Parameters:
        ----------
        args: argparse.Namespace
            The arguments obtained from config.py
        num_procs: int
            Number of parallel processes used with MPI
            Can use 'num_procs()' method from RL/mpi_utils.py
    """
    avg_episode_length = (args.max_duration / 7) * args.avg_assignments_week
    steps_per_epoch_per_task = args.steps_per_epoch / num_procs
    if steps_per_epoch_per_task < avg_episode_length:
        new_steps_per_epoch = int(num_procs * avg_episode_length)
        print_box(f'Changing Steps per epoch from {args.steps_per_epoch} to {new_steps_per_epoch} to accomodate at least one episode per MPI process', 100)
        args.steps_per_epoch = new_steps_per_epoch
    return args
