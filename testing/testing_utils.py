from collections import deque
from datetime import date, timedelta
import datetime
import copy
from pprint import pprint
import random
import itertools as it
import time
from lpSolver.optimization.linear_programming_solvers import solve_lp 

from lpSolver.optimization.linear_programming_solvers import both_handle_disruptions
from pilotRLEnv.utils import calc_disruptions
from pilotRLEnv.env import PilotRLEnv
from RL.models.model import MLPActorCritic
from typing import Dict, Any, List, Set, Tuple
import numpy as np
import torch

def comparable(schedule):
    """ 
        Takes a schedule (formatted as the output from an RL environment) and 
        removes the pilots_assigned. Used for comparing schedules created by
        different methods to assert that they are scheduling the same events.
    """
    ret = {}
    for k, v in schedule.items():
        ret[k] = {
            key: value
            for key, value in v.items() if key != "pilots_assigned"
        }
    return ret

def compare(sched1, sched2, disruption_list):
    for k, v in sched1.items():
        if sched2[k] != v:
            print(k)
            print(sched1[k])
            print(sched2[k])
            return False
    return True


def newDayDisruption(
        date: date,
        env: PilotRLEnv,
        days_out: int,
        pilot_drops: int = 0,
        percent_delayed: int = 0,
        max_delay: int = 0
    ) -> Tuple[List[Tuple[str, Tuple[date, date]]], List[Tuple[int, int]]]:
    """
        Makes a series of disruptions. Note that only static values are read
        from env, so the series of actions performed on env does not matter.
    """
    pilot_drop_dates = []
    possibilities = list(it.product(list(env.history["pilots"].keys()), range(days_out)))
    random.shuffle(possibilities)
    pilot_drop_dates = [(pilot, (date + timedelta(days=days), date + timedelta(days=days))) 
                            for pilot, days in possibilities[:pilot_drops]]

    # Flight ids that start after disruption date
    candidates = [fId for fId, flight in env.history["schedule"].items() 
                          if flight["start"] >= date]
    total = len(candidates)
    selecting = int((percent_delayed / 100) * total)
    delayed_ids = [(i, random.randrange(1, max_delay + 1)) for i in random.sample(candidates, selecting)]

    return pilot_drop_dates, delayed_ids


def updateEnv(schedule, pilots, disruption_date, 
              pilot_drops: List[Tuple[str, Tuple[date, date]]],
              delayed_events: List[Tuple[int, int]]):
    # Pilot drops
    for pilot, (start, end) in pilot_drops:
        pilots[pilot]["unavailability"].append((start, end))
        to_remove = []
        for flight_id in pilots[pilot]["assigned_flights"]:
            flight = schedule[flight_id]
            if (flight["start"] <= end and flight["end"] >= start 
                and flight["start"] > disruption_date):
                pilot_indices = [i for i, e in enumerate(flight["pilots_assigned"]) 
                                 if e == pilot]
                for index in pilot_indices:
                    flight["pilots_assigned"][index] = None
                assert len(pilot_indices) <= 1, "\nSame pilot assigned multiple times to same flight.\n"
                to_remove.append(flight_id)
        pilots[pilot]["assigned_flights"].difference_update(to_remove)

    for flightId, days_delayed in delayed_events:
        flight = schedule[flightId]
        flight["start"] += timedelta(days=days_delayed)
        flight["end"] += timedelta(days=days_delayed)
        for i in range(len(flight["pilots_assigned"])):
            flight["pilots_assigned"][i] = None

def getStart(env: PilotRLEnv) -> datetime.date:
    assert isinstance(env.current_date, datetime.date), "Invalid start date"
    return env.current_date


def runRLLoop(*, env: PilotRLEnv, ac: MLPActorCritic, args, initial_o, 
              initial_valid_actions):
    start = time.time()
    valid_actions = initial_valid_actions
    o = initial_o
    reward_total = 0
    ep_len = 0
    if valid_actions.min() == valid_actions.max() == 0:
        return {
            "info": {"episode_end": "early_stoppage"},
            "ep_len": ep_len,
            "reward_total": reward_total,
        }

    event_slot_pilot = {}
    while True:
        event_slot = env.toFill[0]
        if args.mask_actions:
            a, _, _, pi = ac.step(torch.as_tensor(o, dtype=torch.float32),
                              deterministic=True,
                              act_mask=torch.as_tensor(valid_actions,
                                                       dtype=torch.float32))
        else:
            a, _, _, pi = ac.step(torch.as_tensor(o, dtype=torch.float32),
                              deterministic=True)
        for i, prob in enumerate(pi):
            event_slot_pilot[(*event_slot, env.pilotList[i])] = prob
        # print(event_slot)
        next_o, r, d, info = env.step(a)
        valid_actions = env.getValidPilotsVec()
        reward_total += r
        ep_len += 1
        o = next_o
        if d:
            # print("n round done!", info)
            end = time.time()
            return {
                "info": info,
                "ep_len": ep_len,
                "reward_total": reward_total,
                "time": end - start,
                "event_slot_pilot": event_slot_pilot
            }

def runLPDisruptions(*, LP_scheduling_info, 
                     disruption_list):
    # ret = copy.deepcopy(LP_scheduling_info["schedule"])
    total_changes = 0
    total_moveups = 0
    total_callups = 0
    LP_result = None


    disruption_date = disruption_list[0][0]
    disruptions = disruption_list[0][1]
    old_schedule = copy.deepcopy(LP_scheduling_info)
    end = False
    for flight in old_schedule["schedule"].values():
        assert None not in flight["pilots_assigned"], "None value in filled schedule"

    updateEnv(LP_scheduling_info["schedule"], 
              LP_scheduling_info["pilots"], 
              disruption_date, *disruptions)
    result = both_handle_disruptions(old_schedule,
                                     LP_scheduling_info)
    LP_result = result["filled_schedule"]

    # Update types of disruptions handled
    if LP_result is not None:

        changes, moveups, callups = calc_disruptions(old_schedule["schedule"], LP_result)
        total_changes += changes
        total_moveups += moveups
        total_callups += callups

        for eventId, event in LP_result.items():
            for pilot in event["pilots_assigned"]:
                LP_scheduling_info["pilots"][pilot]["assigned_flights"].add(eventId)
    else:
        total_changes = total_moveups = total_callups = None
    return LP_result, (total_changes, total_moveups, total_callups)

def scheduleCarlo(freshEnv, ac, o, n):
    baseEnv = copy.deepcopy(freshEnv)
    event_slot_pilot_vals = {}
    event_slots = copy.deepcopy(freshEnv.toFill)
    e_s_set = set(event_slots)

    if n == 0:
        for pilot in freshEnv.pilotList:
            for e_s in event_slots:
                esp = (*e_s, pilot)
                event_slot_pilot_vals[esp] = None

        for e_s in event_slots:
            env = copy.deepcopy(baseEnv)
            check = set(env.toFill)
            assert check == e_s_set, (check - e_s_set, e_s_set - check)

            i = env.toFill.index(e_s)
            env.toFill = deque(list(env.toFill)[i:])
            # env.toFill.rotate(-i)

            # env = env.getValidPilotsVec()
            unprocessed_state = env.get_state()
            o = env.process_state(*unprocessed_state)
            _, _, _, probs = ac.step(torch.as_tensor(o, dtype=torch.float32),
                                    deterministic=True)

            # log = np.log10(probs)
            # probs = log / np.sum(log)
            for i, prob in enumerate(probs):
                pilotId = env.pilotList[i]
                esp = (*e_s, pilotId)
                assert event_slot_pilot_vals[esp] is None
                event_slot_pilot_vals[esp] = prob

    # ----------------------------------------------------

    else:
        for pilot in freshEnv.pilotList:
            for e_s in event_slots:
                esp = (*e_s, pilot)
                event_slot_pilot_vals[esp] = []

        for _ in range(n):
            env = copy.deepcopy(baseEnv)
            random.shuffle(env.toFill)
            env.validActions = env.getActions()
            valid_actions = env.getValidPilotsVec()
            unprocessed_state = env.get_state()
            o = env.process_state(*unprocessed_state)

            check = set(env.toFill)
            assert check == e_s_set, (check - e_s_set, e_s_set - check)
            stats = runRLLoop(env=env, ac=ac, args=env.args, initial_o=o, 
                            initial_valid_actions=valid_actions)
            
            event_slot_pilot = stats["event_slot_pilot"]
            for pilot in env.pilotList:
                for event_slot in event_slots:
                    esp = (*event_slot, pilot)
                    if esp in event_slot_pilot:
                        event_slot_pilot_vals[esp].append(event_slot_pilot[esp])
            
        for key, valList in event_slot_pilot_vals.items():
            if valList:
                event_slot_pilot_vals[key] = np.average(valList)
            else:
                event_slot_pilot_vals[key] = 0


    empty_schedule = {
        "pilots": copy.deepcopy(freshEnv.history["pilots"]),
        "schedule": copy.deepcopy(freshEnv.history["schedule"])
    }

    event_slot_pilot_vals = {k: v * 1 for k, v in event_slot_pilot_vals.items()}
    schedule_info = solve_lp(empty_schedule, "CARLO", obj_data=event_slot_pilot_vals)

    return schedule_info

def getOriginalLeave(env):
    return {pId: [{"start": s, "end": e} 
                  for s, e in copy.deepcopy(p["unavailability"])] 
            for pId, p in env.pilotsInfo.items()}

def printResults(title, resultList, digits=1):
    results = []
    for result in resultList:
        if isinstance(result, float):
            if digits == 1:
                results.append(f"{result:>5.1f}")
            elif digits == 2:
                results.append(f"{result:>5.2f}")
            elif digits == 3:
                results.append(f"{result:>5.3f}")
        elif isinstance(result, int):
            results.append(f"{result:>5d}")
        else:
            results.append(f"{str(result):>5}")
    print(f"{title:<37}[{', '.join(results)}]")

