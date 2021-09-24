from scipy.stats.stats import ttest_rel
import torch
import copy
import numpy as np
import sys
import argparse
import itertools as it
import time

from scipy.stats import ttest_ind
from typing import List
from datetime import timedelta
from tqdm import tqdm
from pilotRLEnv.utils import HistoryDict, checkValid
from lpSolver.optimization.linear_programming_solvers import solve_lp
from pilotRLEnv.env import PilotRLEnv
from RL.models.model import MLPActorCritic
from RL.utils.utils import print_args
import argparse
from distutils.util import strtobool
from testing.testing_utils import comparable, compare, getOriginalLeave, newDayDisruption, printResults, runLPDisruptions, runRLLoop, scheduleCarlo, getStart

parser = argparse.ArgumentParser(description='Testing file')
parser.add_argument("-p", '--path', type=str, default='model.ckpt')
parser.add_argument("-v",
                    "--visualize",
                    type=(lambda x: bool(strtobool(x))),
                    default=False)
parser.add_argument("-l",
                    "--lp",
                    choices=[
                        "feasibility", "buffer", "moveup"
                    ],
                    default="feasibility")
parser.add_argument("-s", "--seed", type=int, default=0)
parser.add_argument("-i", "--iterations", type=int, default=10)
parser.add_argument("-w", "--weeks", type=int, default=1, help="Weeks to schedule for.")
parser.add_argument("-c", "--density", help="Density of flights per week. (1 is average, 2 is twice the average)", 
                    type=int, default=1)
parser.add_argument("-d", "--delay_type", choices=["pilots_drop", "flights_delay"], default="flights_delay")
parser.add_argument("-n", "--delay_parameters", type=int, nargs="+", default=[50, 2],
    help="The parameters for your chosen disruption. If 'pilots_drop', provide one integer, "
         "the number of pilots you want to drop. If 'flights_delay', provide two integers: "
         "first, the percentage of missions that you want delayed and, second, the maximum"
         "delay of those missions.")
parser.add_argument("-r", "--repeats", help="Number of times to repeat the schedule for the CARLO agent", 
                    type=int, default=8)




args_path = parser.parse_args()
visualize: bool = args_path.visualize
lp_choice: str = args_path.lp
random_seed: int = args_path.seed
iterations: int = args_path.iterations
weeks: int = args_path.weeks
delay_type: str = args_path.delay_type
delay_params: List[int] = args_path.delay_parameters
repeats: int = args_path.repeats
density: int = args_path.density

if delay_type == "pilots_drop":
    assert len(delay_params) == 1, "Only one parameter should be supplied for delay_paramters when the delay type is 'pilots_drop'"
    delay_args = {
        "pilot_drops": delay_params[0]
    }
elif delay_type == "flights_delay":
    assert len(delay_params) == 2, "Only two parameters should be supplied for delay_paramters when the delay type is 'flights_delay'"
    delay_args = {
        "percent_delayed": delay_params[0],
        "max_delay": delay_params[1]
    }

# load the model checkpoint
ckpt = torch.load(args_path.path)
args = ckpt['args']
args.max_duration = 7 * weeks
args.flight_density = density
print_args(args, 80)

# init the environment with the arguments used to train the model
env = PilotRLEnv(args, seed=random_seed)  # Original is 13

# init the actor critic
ac_kwargs = dict(hidden_sizes=[args.hid] * args.l)
ac = MLPActorCritic(env.observation_space, env.action_space, **ac_kwargs)
ac.load_state_dict(ckpt['actor_state_dict'])  # load network weights
ac.eval()  # make sure to turn the network to eval mode

disruption_slots = []

methodNameList = ["LP", "RL_H", "CARLO"]

stats_tracker = {
    method: {
        stat: [] for stat in [
            "disruptions",
            "moveups",
            "callups",
            "new_event_slots", 
            "delayed_flight_slots",
            "pilot_drop_slots", 
            "new_event_slots_total", 
            "delayed_flight_slots_total",
            "time",
            "constraints",
            "variables"
        ]
    } for method in methodNameList
}

# run test episodes --seed times
rewards = []
ep_frac_lengths = []
ep_lens = []

original_leave = getOriginalLeave(env)
for test_rollouts in tqdm(range(iterations)):
    o = env.reset()
    stopped = {"LP": False, "RL_H": False, "CARLO": False}

    # First, the CARLO approach
    start = time.time()
    o_copy = copy.deepcopy(o)
    schedule_info = scheduleCarlo(env, ac, o, repeats)
    end = time.time()
    CARLO_schedule = schedule_info["filled_schedule"]
    stats_tracker["CARLO"]["constraints"].append(schedule_info["constraints"])
    stats_tracker["CARLO"]["variables"].append(schedule_info["variables"])
    stats_tracker["CARLO"]["time"].append(end - start)
    CARLO_scheduling_info = None

    start_date = getStart(env)

    if CARLO_schedule is None:
        stopped["CARLO"] = True
    else:
        CARLO_pilots = copy.deepcopy(env.history["pilots"])
        for eventId, event in CARLO_schedule.items():
            for pilot in event["pilots_assigned"]:
                CARLO_pilots[pilot]["assigned_flights"].add(eventId)
        CARLO_scheduling_info = {
            "schedule": CARLO_schedule,
            "pilots": CARLO_pilots
        }

    # Build the LP schedule
    start = time.time()
    schedule_info = solve_lp(env.history, lp_choice)
    end = time.time()
    LP_schedule = {
        "schedule": schedule_info["filled_schedule"],
        "pilots": copy.deepcopy(env.history["pilots"])
    }
    stats_tracker["LP"]["constraints"].append(schedule_info["constraints"])
    stats_tracker["LP"]["variables"].append(schedule_info["variables"])
    stats_tracker["LP"]["time"].append(end - start)
    if LP_schedule is None:
        # Can't continue with the hybrid or LP method if there is no feasible solution
        stopped["LP"] = True

    # Build the RL schedule
    RL_rewards = 0
    valid_actions = env.getValidPilotsVec()
        
    start = time.time()
    stats = runRLLoop(env=env, ac=ac, args=args, initial_o=o, 
                      initial_valid_actions=valid_actions)
    end = time.time()
    stopped["RL_H"] = stats["info"]["episode_end"] != "completed"
    stats_tracker["RL_H"]["time"].append(end - start)
    stats_tracker["RL_H"]["constraints"].append(None)
    stats_tracker["RL_H"]["variables"].append(None)
    RL_rewards += stats["reward_total"]

    # Get the RL schedule for later use handling disruptions with the LP
    RL_H_scheduling_info = {}
    for key in ["pilots", "schedule"]:
        RL_H_scheduling_info[key] = copy.deepcopy(env.history[key])
    RL_H_schedule = copy.deepcopy(env.history["schedule"])
    for flight in RL_H_schedule.values():
        if None in flight["pilots_assigned"]:
            RL_H_schedule = None


    # Create disruption for any of the remaining methods to handle.
    disruption_date = start_date + timedelta(days=1)
    disruption_list = [
        (disruption_date, newDayDisruption(disruption_date, env, weeks * 7 - 1, **delay_args))
    ]

    finalSchedules = {
        "LP": None,
        "RL_H": None,
        "CARLO": None
    }

    for stoppedVal, scheduling_info in [("LP", LP_schedule), 
                                        ("RL_H", RL_H_scheduling_info), 
                                        ("CARLO", CARLO_scheduling_info)]:
        disruption_stats = [None] * 3
        if not stopped[stoppedVal] and scheduling_info["schedule"]:
            finalSchedules[stoppedVal], disruption_stats = (
                runLPDisruptions(LP_scheduling_info=scheduling_info, 
                                 disruption_list=disruption_list))

        for name, val in zip(["disruptions", "moveups", "callups"], disruption_stats):
            stats_tracker[stoppedVal][name].append(val)

    LP_schedule = finalSchedules["LP"]
    CARLO_schedule = finalSchedules["CARLO"]
    RL_H_schedule = finalSchedules["RL_H"]


    scheduleList = [LP_schedule, RL_H_schedule, CARLO_schedule]

    for schedule1, schedule2 in it.combinations(scheduleList, 2):
        if schedule1 is not None and schedule2 is not None:
            assert compare(comparable(schedule1), comparable(schedule2), disruption_list)

    for schedule in scheduleList:
        checkValid(schedule, env, original_leave)


    if visualize:
        if LP_schedule:
            env.history["schedule"] = HistoryDict(LP_schedule)
            env.visualize(title=f"{test_rollouts} LP Final Schedule")
        if CARLO_schedule:
            env.history["schedule"] = HistoryDict(CARLO_schedule)
            env.visualize(title=f"{test_rollouts} CARLO schedule")

print()
print(' '.join(sys.argv))


printing = []
for method, stats in stats_tracker.items():
    for stat_name, values in stats.items():
        if values:
            if stat_name == "disruption_percentage":
                continue
            if stat_name in ["total_percentage", "disruption_percentage"] or "total" in stat_name:
                printing.append((f"{method} {stat_name}", values, 3))
            else:
                printing.append((f"{method} {stat_name}", values))
    printing.append(None)

for params in printing:
    if params is None:
        print()
    else:
        printResults(*params)

for stat in ["disruptions", "time", "constraints", "variables"]:
    print(f"NEW STAT REPORT FOR: {stat}")

    lp_stat = stats_tracker["LP"][stat]
    lp_nones = [i for i, e in enumerate(lp_stat) if e is None]
    lp_stat = np.array([i for i in lp_stat if i is not None])


    CARLO_stat = stats_tracker["CARLO"][stat]
    CARLO_nones = [i for i, e in enumerate(CARLO_stat) if e is None]
    CARLO_stat = np.array([i for i in CARLO_stat if i is not None])

    rl_stat = stats_tracker["RL_H"][stat]
    rl_nones = [i for i, e in enumerate(rl_stat) if e is None]
    rl_stat = np.array([i for i in rl_stat if i is not None])

    print(f"Average of {stat}:")
    print("LP:", np.average(lp_stat), "CARLO:", np.average(CARLO_stat), "RL First:", np.average(rl_stat))
    print()
    std_devs = [lp_stat.std(), CARLO_stat.std(), rl_stat.std()]
    std_devs_lp = std_devs[0:2]
    std_devs_rl = std_devs[1:3]
    print(f"Standard deviation of {stat}:")
    print(" ".join([f"{a}{b}" for a, b in zip(["LP: ", "CARLO: ", "RL First: "], std_devs)]))
    print()

    if CARLO_nones == lp_nones:
        print("Paired LP")
        ttestPairedLP = ttest_rel(CARLO_stat, lp_stat)
        print(ttestPairedLP)
        print("p-value %:", ttestPairedLP.pvalue * 100)
        print()
        print("Number of Nones:", len(CARLO_nones))
        print("% Nones:", len(CARLO_nones) * 100 / iterations)
        print()
    else:
        assert False, "Nones do not match between CARLO and IP"

    if CARLO_nones == rl_nones:
        print("Paired RL")
        ttestPairedRL = ttest_rel(CARLO_stat, rl_stat)
        print(ttestPairedRL)
        print("p-value %:", ttestPairedRL.pvalue * 100)
        print()

    print("Independent LP")
    ttestIndLP = ttest_ind(CARLO_stat, lp_stat, equal_var=False)
    print(ttestIndLP)
    print("p-value %:", ttestIndLP.pvalue * 100)
    print()

    print("Independent RL")
    ttestIndRL = ttest_ind(CARLO_stat, rl_stat, equal_var=False)
    print(ttestIndRL)
    print("p-value %:", ttestIndRL.pvalue * 100)
    print()
    print()
    print()
    print()