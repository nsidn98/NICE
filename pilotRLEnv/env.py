"""
    Code for defining the pilot scheduling environment
    Environment:
    ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    Episode:
    ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
        • Consider one episode as assigning pilots to events for a duration of 
            1-week.
        • After each episode (whether it gets completed or not) move to next 
            week.
        • Start from June and end in December and keep rolling back.
    ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    State:
    –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
        • Pilot availability        |   (ep_horizon x num_pilots) shaped matrix |
            matrix (Optional)       |                                           |
        • Current event type        |   (num_event_types) shaped one-hot vector |
            (one-hot)               |                                           |
        • Current event-pilot       |   (num_pilots) shaped binary vector       |
            assignments             |                                           |
        • Valid pilots for the      |   (num_pilots) shaped binary vector       |
            current event           |                                           |
        • Event duration            |   integer denoting how many days the      |
                                    |   current event takes place               |
        • Event start day number    |   integer denoting how many days would    |
                                    |   have elapsed for the start date for the |
                                    |   event since the beginning of the episode|
        • Event end day number      |   integer denoting how many days would    |
                                    |   have elapsed for the end date for the   |
                                    |   event since the beginning of the episode|
    ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    Action:
        Choose one pilot out of 'num_pilots'
        NOTE: There is no explicit constraint of only choosing from valid pilots 
        for the current event. The agent is supposed to learn the relation 
        between the valid pilots for the current event and qualification ranks
    ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    Reward: 
    ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
        • +r for placing a valid pilot in an event                              |
            r is defined in the paper                                           |
        • -10 if did not place a valid pilot to the event                       |
        • -10 if did not complete assignment of all pilots for the current      |
            duration                                                            |
        • +25 if successful in completing all assignments for the current       |
            duration                                                            |
    ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    Usage:
    ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    from pilotRLEnv.env import PilotRLEnv
    from pilotRLEnv.env_config import args
    # test with random sampling of only valid pilots
    env = PilotRLEnv(args, seed=args.seed)
    s  = env.reset()
    i = 0
    done = False
    reward = 0
    while not d:
        ns, r, d, info = env.step(env.sampleAction())
        reward += r
    print(reward, env.total_episode_event_num, env.episode_horizon)
    ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
"""
import os
import json
import warnings
import pandas as pd
import argparse
import copy
from typing import Dict, Tuple, List
import datetime
from datetime import date, timedelta
import numpy as np
import random
import itertools as it
import matplotlib.pyplot as plt
from collections import deque, defaultdict
from gym import spaces
import shutil

import pilotRLEnv.utils as utils
from pilotRLEnv.utils import RandomEventCreator, print_dash, print_box, HistoryDict

class PilotRLEnv():
    def __init__(self, args:argparse.Namespace, seed:int=0, verbose:bool = False):
        """
            An openai gym type environment for flight scheduling
            Have to only use the following output ports from this class:
            env.reset() -> to reset the environment
            env.step(action)
            Usage:
            ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
            env = PilotRLEnv(args)
            state = env.reset()
            # while the episode is not done do:
            while not done:
                action = env.sampleAction() # or can use agent.act(state);
                # sampleAction will choose a random action from all actions 
                # which are valid at that point 
                next_state, reward, done, info = env.step(action)
                # perform RL stuff here
                state = next_state
            ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
            Parameters:
            –––––––––––
            args: argparse.Namespace
            Should contain the following
                –––––––––––– filepath to date ––––––––––––
                flight_distribution_path : str
                    Path to the json file for flight distributions
                    Example : 'flight_distribution.json'
                tr_path: str
                    Path to the json file for training requirements
                    Example : 'pilotRLEnv/data/Minimized Dataset - All Months/ \
                                training_requirements.csv'
                sim_distribution_path : str
                    Path to the json file for sim distributions
                    Example : 'sim_distribution.json'
                folder : str
                    Path to the directory with the csvs containing pilot 
                    information
                    Example : "Minimized Dataset - All Months/"
                flight_data: str
                    Name of the local missions file; used when loading events 
                    manually
                    Example: 'LocalMissions.csv'
                sim_data: str
                    Name of the sims file; used when loading events manually
                    Example: 'Sims.csv'
                manual_event_load: bool
                    Whether events are to be loaded manually from the AF dataset 
                    or random creation
                ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
                –––––––––––– state representation ––––––––––––
                num_pilots : int
                    Number of pilots taking part in the scheduling process
                    Example : 87
                PA_look_ahead: int
                    Number of days to look ahead in the pilot availability matrix
                num_event_types: int
                    Number of types of events
                normalise_state: bool
                    Whether the date related features in the states should be 
                    normalised or not
                use_pa_matrix: bool
                    Whether the pilot availability matrix is to be included in 
                    the state representation or not
                modify_terminal_PA: bool
                    If True, then modify PA matrix as -1 for terminal state
                use_training_req: bool
                    Whether the training requirement vector is to be included in 
                    the state representation or not
                include_moveup_buffer_vec: bool
                    Whether the moveup and buffer vector is to be included in 
                    the state representation or not
                use_event_type: bool
                    Use event type in state if true, otherwise, use requirement 
                    of current slot
                ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
                –––––––––––– reward structure ––––––––––––
                pilot_place_wt: float
                    Weight given to placing a valid pilot
                buffer_weight: float
                    Weight given to buffer score
                moveup_weight: float
                    Weight given to moveup score
                ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
                –––––––––––– action space ––––––––––––
                mask_actions: bool
                    If we only want valid pilots in the action space of the agent
                    If `True`, renormalise the probability distributions 
                    predicted by actor to only pilots who are eligible
                ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
                –––––––––––– environment ––––––––––––
                no_valid_moves : str
                    To check if there are any valid moves
                    Example : "no valid"
                illegal_action : str
                    To check for illegal actions
                    Example : "illegal"
                max_duration : int
                    The maximum game length
                    Example : 20
                ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
                –––––––––––– heuristics obtained from data ––––––––––––
                avg_assignments_week: int
                    Average number of pilot-event assignments per week
                    Example : 80
                flight_density: int
                    The density of flights/week, where 1 is average. 
                    (2 would be twice the average)
                    Example : 1
                ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
                –––––––––––– dates in schedule ––––––––––––
                START_DATE : str
                    Start date of the schedule in "YYYY-MM-DD" format
                    Example : '2019-06-24'
                END_DATE : str
                    End date of the schedule in "YYYY-MM-DD" format
                    Example : '2019-08-02'
                ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
            seed: int
                seed for all randomness
        """
        self.args = args
        self.verbose = verbose
        self.episode_number = 0 # this will change after every reset; 
        # basically keeps a count of number of times we shift the start date
        self._load_dicts()
        self._load_info()
        self._load_pilots()
        self._load_training_requirements()
        self.randomEventCreator = RandomEventCreator(seed)
        self.repeat_events = False  # whether we want to re-run the assignments 
        # for the same set of events with different order
        self.copied_events = False  # whether the events dict have been copied

        # set seeds for randomness
        random.seed(seed)
        np.random.seed(seed)

        # define action space
        self.action_space = spaces.Discrete(self.args.num_pilots)

        # dummy date
        start_date = utils.convert_date(self.args.START_DATE)
        self.episode_start_date = start_date

        dummy_event = {'end': datetime.date(833, 4, 18),
                        'pilots_assigned': [None, None], 
                        'requirements': ['r17', 'r13'],
                        'start': datetime.date(883, 4, 11), 
                        'type': 't16', 'A': False, 'B': False}
        dummy_processed_event = self._process_event(dummy_event, 0)
        dummy_valid_pilots = np.zeros(self.args.num_pilots)
        dummy_pa_matrix = np.zeros((self.args.PA_look_ahead, self.args.num_pilots))
        dummy_moveup_vec = np.zeros(self.args.num_pilots)
        dummy_buffer_vec = np.zeros(self.args.num_pilots)
        dummy_req_vec = np.zeros(self.args.num_pilots)
        dummy_state = self.process_state(dummy_pa_matrix, dummy_valid_pilots, 
                                        dummy_processed_event, dummy_req_vec, 
                                        dummy_buffer_vec, dummy_moveup_vec)
        state_dim = dummy_state.shape[0]
        

        self.observation_space = spaces.Box(low=0, high=np.inf, 
                                            shape=(state_dim,), dtype=np.uint8)

        
        self.num_weeks = int(self.args.max_duration / 7)
        self.assignment_density = self.args.flight_density
        if self.args.manual_event_load:
            print_box('Loading the csvs for manual event loading...')
            flight_data_path = os.path.join(self.args.folder, self.args.flight_data)
            sim_data_path    = os.path.join(self.args.folder, self.args.sim_data)
            self.flightData  = pd.read_csv(flight_data_path)
            self.simData     = pd.read_csv(sim_data_path)

    def __repr__(self):
        """
            Will print the arguments used in the environment
            Usage:
                env = PilotRLEnv(args, seed=0)
                print(env)
        """
        box_dist = 100
        env_args = ['flight_distribution_path', 'sim_distribution_path', 'folder',
                    'tr_path', 'flight_data', 'sim_data', 'manual_event_load', 
                    'num_pilots', 'PA_look_ahead', 'num_event_types', 
                    'normalise_state', 'use_pa_matrix', 'modify_terminal_PA', 
                    'include_moveup_buffer_vec', 'use_event_type', 'pilot_place_wt',
                    'buffer_weight', 'moveup_weight', 'mask_actions'
                    'no_valid_moves', 'illegal_action', 'max_duration', 
                    'avg_assignments_week', 'flight_density',
                    'START_DATE', 'END_DATE', 'seed']
        print_dash(box_dist)
        half_len = int((box_dist-len("Arguments")-5)/2)
        print("||" + " "*half_len + "Arguments" + " "*half_len + " ||")
        print_dash(box_dist)
        for k, v in vars(self.args).items():
            len_line = len(f"{k}: {str(v)}")
            if k in env_args:
                print("|| " + f"{k}: {str(v)}" + " "*(box_dist-len_line-5) + "||")
        print_dash(box_dist)
        return ''

    def reset(self):
        """
            Reset the environment
            • This will shift the start date by `max_duration` till we reach 
                the end date
            • Create a week worth of random events to assign pilots to
            Returns:
            ––––––––
            np.ndarray
        """
        while True:
            start_date = utils.convert_date(self.args.START_DATE) + \
                        timedelta(self.episode_number * self.args.max_duration)
            # if we have reached the end, then get back the the actual start date
            if start_date > utils.convert_date(self.args.END_DATE):
                start_date = utils.convert_date(self.args.START_DATE)
                self.episode_number = 0
            self.episode_start_date = start_date
            self.current_date = start_date    # this will change

            self.history = {
                # The game treats HistoryDict like a normal dictionary. The 
                # getHistory() method can see all past events, including ones 
                # that were removed, to evaluate schedules on various metrics.
                "schedule": HistoryDict(),
                "pilots": copy.deepcopy(self.pilotsInfo),
                "date": [start_date, 0, 0] 
            }
            # Used to generate unique flight/sim IDs
            self.id = 0
            # The queue of all flights the scheduler needs to fill 
            self.toFill = None
            # pilot lists and index converter
            # self.pilotList = list(self.history["pilots"].keys())
            # self.pilotToIndex = {p: i for i, p in enumerate(self.pilotList)}

            self._newWeekSchedule(start = start_date)
            self._updateToFill()
            self.pilotsFlying = defaultdict(list)
            # this is equal to the horizon of the episode; total number of 
            # event-pilot assignments required to complete the episode
            self.episode_horizon = len(self.toFill)
            self.num_pilots_assigned_in_episode = 0
            # overqualification
            # the sum of overqual differences for all assigned pilots  
            self.episode_qual_diff = 0
            # to keep track of overquals over the episode
            self.episode_overquals = []
            self.requirement_matrix = copy.deepcopy(self.requirement_matrix_init)
            # to track if pilots are getting assigned number of days fairly
            self.pilot_quality_life = np.zeros(self.args.num_pilots)

            self.validActions = self.getActions()
            if self.validActions == [None]:
                continue
            state_args = self.get_state()
            state = self.process_state(*state_args)
            self.episode_number += 1
            return state

    def step(self, action:int):
        """
            Performs the action on the environment and returns the next state, 
            reward, a bool about termination of episode and info about the episode 
            Parameters:
            –––––––––––
            action : int pilot_number (NOT pilot_id)
                Example: 10 –––––––> (pilot_id='10010')
            Returns:
            ––––––––
            state, reward, done, info = Tuple[np.ndarray, float, bool, Dict]
        """
        info = {}; info['episode_end']='not_done'
        done = False
        # toFill has the events that have to be filled
        if len(self.toFill):
            # this is just the start date of the current event that is being scheduled
            self.current_date = self.history['schedule'][self.toFill[0][0]]['start']
        pilot_id = list(self.pilotToIndex.keys())[list(self.pilotToIndex.values()).index(action)] 
        event_slot_info = self.validActions[0][1] # Example: (10,0)
        action = (pilot_id, event_slot_info) # Example: ('10000', (10, 0))
        buffer_score, moveup_score, self.validActions = self._takeAction(action)

        if buffer_score == self.args.illegal_action:
            reward = -10    # because illegal action, give penalty
            info['episode_end'] = 'illegal_action'
            done = True
        else:
            reward = (self.args.pilot_place_wt + 
                      buffer_score * self.args.buffer_weight + 
                      moveup_score * self.args.moveup_weight)
        ############################################
            if self.validActions == self.args.no_valid_moves:
                info['episode_end'] = 'early_stoppage'
                done = True
                # Couldn't schedule any pilot to an event so infeasible assignment
                if self.verbose:
                    print_dash()
                    print('Stopping early as no valid pilots found')
                    print_dash()
                # stopping early due to bad scheduling at some 
                # intermediate time-step, give penalty
                reward = -10
            ################## NOTE DEBUGGING ##################
            if self.validActions ==  [None] or (len(self.toFill) < 1):
                # every event in the week scheduled; go to next week events
                info['episode_end'] = 'completed'
                done = True
                reward = 25
        next_state_args = self.get_state()
        next_state = self.process_state(*next_state_args)
        return next_state, reward, done, info
    
    def _buffer(self, pilot_id, event_num):
        previous_events = []
        assigned_event = self.history["schedule"][event_num]
        NO_PREVIOUS_REWARD = self.args.max_duration - 2

        for e_id, e in self.history["schedule"].items():
            for assigned_id in e["pilots_assigned"]:
                if assigned_id == pilot_id and e_id != event_num:
                    previous_events.append(e)
        if not previous_events:
            return NO_PREVIOUS_REWARD
        pilot_prev_event = max(previous_events, key=lambda a:a["end"])
        buffer = (assigned_event["start"] - pilot_prev_event["end"]).days - 1
        return buffer

    def _get_buffer_reward(self, action):
        pilot_id, (event_num, _slot_num) = action
        return self._buffer(pilot_id, event_num)

    def _move_ups(self, j_id, g_id) -> float:
        # We add pilot j to flight g. 
        # How many slots s on other flights f can pilot j be a moveup pilot for?
        j_info = self.history["pilots"][j_id]
        j_pilot_unavailability = j_info["unavailability"]
        j_assigned_flights = j_info["assigned_flights"]
        j_qual = j_info["qualification"]

        schedule = self.history["schedule"]
        g = schedule[g_id]
        moveup_count = 0
        # g_range = [(g["start"], g["end"])]
        # g_req = g["requirements"][slot_num]
        T_moveup = 2

        for f_id, f in schedule.items():
            # Condition 1
            if f_id == g_id: 
                continue
            # Condition 2 and 3
            if (g["start"] >= f["start"] 
                and (g["start"] - f["start"]).days <= T_moveup
                and g["end"] >= f["end"]):

                # Condition 4
                f_range = [(f["start"], f["end"])]
                if utils.overlap(f_range, j_pilot_unavailability):
                    continue

                # Condition 5
                ineligible = False
                for o_id in j_assigned_flights:
                    o = schedule[o_id]
                    if o["start"] < f["start"] and utils.overlap(f_range, [(o["start"], o["end"])]):
                        ineligible = True
                        break
                if ineligible:
                    continue

                # Condition 6
                for req, i_pilot in zip(f["requirements"], f["pilots_assigned"]):
                    if i_pilot is None or not utils.satisfiableMember(req, j_qual):
                        continue
                    moveup_count += 1

        return moveup_count

    def _get_moveup_reward(self, action:Tuple) -> float:
        # We add pilot j to flight g. 
        # How many slots s on other flights f can pilot j be a moveup pilot for?
        j_id, (g_id, _slot_num) = action
        return self._move_ups(j_id, g_id)

    def _load_info(self) -> None:
        """
            Load information
            flightInfo and simInfo
            Read-only so the game doesn't have to do file io
            when it updates the weekly schedule.
        """
        with open(self.args.flight_distribution_path) as f:
            self.flightInfo = json.load(f)
        with open(self.args.sim_distribution_path) as f:
            self.simInfo = json.load(f)

    def _load_pilots(self) -> None:
        """
            Load the pilot information
            This will get the qualifications of each pilot and the 
            dates of unavailability
            Returns:
            --------
            Dict
        """
        pilot_df = pd.read_csv(self.args.folder + "Pilot Letter of X.csv")[1:-1]
        leave = pd.read_csv(self.args.folder + "Leave TDY.csv", 
                            parse_dates=["First", "Last"], dayfirst=True)

        self.pilotsInfo = {}
        for _, row in pilot_df.iterrows():
            self.pilotsInfo[row.Name] = {
                "qualification": row.QUAL,
                "unavailability": [],
                "assigned_flights": set()
            }
        for _, row in leave.iterrows():
            self.pilotsInfo[str(row.Name)]["unavailability"].append(
                                            (row.First.date(), row.Last.date()))
        
        self.pilotList = list((copy.deepcopy(self.pilotsInfo)).keys())
        self.pilotToIndex = {p: i for i, p in enumerate(self.pilotList)}

    def _load_dicts(self) -> None:
        """
            Load all the dicts for:
            • event_types
            • qualification rankings
            • AD ranking
            • SOLL ranking
            • minimum qualification required for each event
            • event requirement column names
            • event requirements considered
        """
        # mapping from event type to a number
        with open("pilotRLEnv/data/Minimized Dataset - All Months/eventInfo.json") as f:
            data = json.load(f)
            reqs = set()
            for val in data["sims"].values():
                reqs.update(val["crew"])
            for val in data["flights"].values():
                reqs.update(val)
            self.req_mapping = {r: i for i, r in enumerate(reqs)}
            self.event_type_mapping = data["event_type_mapping"]

            self.event_tr_mapping = data["event_tr_mapping"]
            self.tr_event_mapping = defaultdict(list)
            for tr, events in self.event_tr_mapping.items():
                for event in events:
                    self.tr_event_mapping[event].append(tr)
            self.tr_event_mapping = dict(self.tr_event_mapping)
            self.event_trs = list(self.event_tr_mapping.keys())

    def _load_training_requirements(self) -> None:
        """
            Load the training requirement csv and make a matrix of 
            pilot-event_requirement. Just initialise the matrix once and then 
            copy this in the state so that we do not have to read with pandas 
            every episode
        """
        self.requirement_matrix_init = np.zeros((self.args.num_pilots,
                                                len(self.event_trs)))
        self.df_requirements = pd.read_csv(self.args.tr_path)
        # pull only relevant requirement columns
        self.df_requirements = self.df_requirements[['Qual'] + self.event_trs]
        for pilot_id, pilot_info in self.pilotsInfo.items():
            pilot_index = self.pilotToIndex[pilot_id]
            pilot_qual = pilot_info['qualification']  
            # eg: array([0, 1, 2, 0, 2, 0, 1, 0, 0, 1, 1, 3, 1, 1, 2, 0, 1], dtype=object)
            requirements = np.array(self.df_requirements[
                            self.df_requirements['Qual'] == pilot_qual])[0][1:]
            self.requirement_matrix_init[pilot_index] = requirements

    def _updateToFill(self) -> None:
        """
            Update the events that have to be filled
        """
        targets = []
        if self.toFill is None:
            scheduleList = self._retrieveScheduleList()
            # if we want to repeat the same set of events with a different order
            # if repeat:
            #     scheduleList = self._retrieveScheduleListRandom()
            for mId, missionValues in scheduleList:
                pilots_assigned = missionValues["pilots_assigned"]
                for slotId, pilot_assigned in enumerate(pilots_assigned): 
                    if pilot_assigned == None:
                        targets.append((mId, slotId))
        else:
            for mId, slotId in self.toFill:
                pilot_assigned = self.history["schedule"][mId]["pilots_assigned"][slotId]
                if pilot_assigned is None:
                    targets.append((mId, slotId))
        self.toFill = deque(targets)

    def _retrieveScheduleList(self):
        """
            Returns:
            --------
            List
        """
        # The order of filling upcoming flights will always 
        # be chronological, with the id as a tie breaker
        return sorted(list(self.history["schedule"].items()), 
                    key=lambda a: (a[1]["start"], a[0]))
    
    def _retrieveScheduleListRandom(self):
        """
            The order of filling upcoming flights will always be chronological
            Same as self._retrieveScheduleList() but here the events are randomly 
            shuffled and only arranged according to the dates rather than also 
            having the tie-breaker of id.
            This will be used when we want to shuffle the order of events but 
            want to keep them chronologically
            Returns:
            --------
            List
        """
        scheduleList = list(self.history["schedule"].items())
        random.shuffle(scheduleList)    # shuffle the events
        return sorted(list(self.history["schedule"].items()), 
                        key=lambda a: (a[1]["start"]))

    def _newWeekSchedule(self, start:datetime.time, repeat:bool=False) -> None:
        """
            Make new week schedule
            Adds sims and flights from distribution in minimized dataset
        """
        end = start + timedelta(days=self.args.max_duration)
        if self.verbose:
            print(f'Adding new week schedule from {start} to {end-timedelta(1)}')

        if not repeat:
            currentSchedule = self.history["schedule"]
            simAvg = self.simInfo["avg"]
            simStd = self.simInfo["stddev"]
            numSims = round(np.random.normal(simAvg, simStd) / 2) * self.num_weeks * self.assignment_density
            
            for _ in range(numSims):
                flight_id = self.id
                self.id += 1
                sim = self.randomEventCreator.randomSim(start, end, self.simInfo)
                currentSchedule[flight_id] = sim
            
            flightAvg = self.flightInfo["avg"]
            flightStd = self.flightInfo["stddev"]
            # average and mean are for one week
            numFlights = round(np.random.normal(flightAvg, flightStd) / 2) * self.num_weeks * self.assignment_density

            for _ in range(numFlights):
                flight_id = self.id
                self.id += 1
                currentSchedule[flight_id] = self.randomEventCreator.randomFlight(start, end, self.flightInfo)
            
            # copy the schedule for further use
            self.schedule_copy = copy.deepcopy(currentSchedule)
            
        else:
            print('Copying events')
            self.history['schedule'] = copy.deepcopy(self.schedule_copy)

        self.total_episode_event_num = len(currentSchedule)
    
    def getActions(self):
        """
            Get the valid actions
            Returns:
            --------
            List
        """
        schedule = self.history["schedule"]
        if self.toFill:
            target = self.toFill[0]
        else:
            # Only action is do nothing because all flights are filled. This is good!
            return [None]
        flight_i, pilot_i = target
        flight = schedule[flight_i]
        requirement = flight["requirements"][pilot_i]
        eligible_pilots = []
        for pilot, pilot_info in self.history["pilots"].items():
            flight_range = [(flight["start"], flight["end"])]
            # Make sure pilot is availabile
            if utils.overlap(flight_range, pilot_info["unavailability"]):
                continue
            assigned_flights = pilot_info["assigned_flights"]
            assigned_flight_ranges = [(schedule[flight_id]["start"], 
                                      schedule[flight_id]["end"]) 
                                      for flight_id in assigned_flights 
                                      if flight_id in schedule]
            # Make sure pilot is not assigned to another flight
            if utils.overlap(flight_range, assigned_flight_ranges):
                continue
            qual = pilot_info["qualification"]
            # Make sure pilot is qualified for the flight
            if utils.satisfiableMember(requirement, qual):
                eligible_pilots.append(pilot)
        if len(eligible_pilots) == 0:
            # There are no more moves. This is bad!
            return self.args.no_valid_moves
        actions = [(pilot, target) for pilot in eligible_pilots]
        return actions

    def getValidPilotsVec(self):
        """
            Convert self.validActions: [('10000', (15, 1)), ('10001', (15, 1))]
            to binary vector [1,1,0,...,0]
        """
        validPilotsVec = np.zeros(self.args.num_pilots)
        # if no one is valid, then return zero vec
        if self.validActions == 'no valid':
            return validPilotsVec
        # only iterate if validActions is not [None]
        # validActions = [None] means that the episode is complete
        if None not in list(self.validActions):
            for pilot in self.validActions:
                pilotId = self.pilotToIndex[pilot[0]]
                validPilotsVec[pilotId] = 1
        return validPilotsVec
    
    # next three methods are related to the dynamics of the environment
    def sampleAction(self, onlyValid:bool=True) -> int:
        """
            Sample a random pilot from validActions
            Parameter:
            ----------
            onlyValid: bool
                If this is true, then it will sample only 
                from the valid pilots for the current event
                else it will sample randomly from the whole
                set of pilots
        """
        if onlyValid:
            random_pilot = random.choice(self.validActions) # example: ('10081', (1, 0))
            random_pilot_num = self.pilotToIndex[random_pilot[0]]
        else:
            random_pilot_num = random.choice(np.arange(0,self.args.num_pilots,1))
        return random_pilot_num

    def _takeAction(self, action:Tuple):
        """
            Take action in the environment
            NOTE that env.step() calls this inside it
            Parameter:
            ----------
            action : Tuple
                    ('pilot_id', ('event_num', 'slot_num'))
                Example: ('10000',       (2,         0))
                        ('pilot_id', (event_num, slot_num))
            NOTE: ONLY valid actions are supposed to be given
            Returns:
            --------
            Tuple[float, float, List]
        """
        schedule = self.history["schedule"]
        overqual_score = 0
        req_score = 0
        # Play action None when everything is filled to reset game board
        if action != None:
            if action[1] != self.toFill[0]:
                if self.verbose:
                    print_dash()
                    print('Illegal action as events do not match')
                    print_dash()
                return self.args.illegal_action, None, []
            self.toFill.popleft()
            # Assign pilot to appropriate slot
            pilot, (flight_i, pilot_i) = action
            flight = schedule[flight_i]
            pilot_info = self.history["pilots"][pilot]
            flight_range = [(flight["start"], flight["end"])]
            # Repeats checks. remove in future if confident about game's correctness
            if utils.overlap(flight_range, pilot_info["unavailability"]):
                if self.verbose:
                    print_dash()
                    print("Pilot not available and is assigned. Check actions")
                    print_dash()
                return self.args.illegal_action, None, []
            if not utils.satisfiableMember(flight["requirements"][pilot_i], 
                                           pilot_info["qualification"]):
                if self.verbose:
                    print_dash()
                    print("Pilot is not qualified and is assigned. Check actions")
                    print_dash()
                return self.args.illegal_action, None, []
            assigned_flights = pilot_info["assigned_flights"]
            assigned_flight_ranges = [(schedule[flight_id]["start"], 
                                      schedule[flight_id]["end"]) 
                                      for flight_id in assigned_flights 
                                      if flight_id in schedule]
            if utils.overlap(flight_range, assigned_flight_ranges):
                if self.verbose:
                    print_dash()
                    print("Flight overlaps not matching. Please check here")
                    print_dash()
                return self.args.illegal_action, None, []
            flight["pilots_assigned"][pilot_i] = pilot
            pilot_info["assigned_flights"].add(flight_i)
            num_event_days = (flight['end'] - flight['start']).days + 1
            self.pilot_quality_life[self.pilotToIndex[pilot]] += num_event_days
            self.num_pilots_assigned_in_episode += 1
            self.history["date"][2] += 1
            # update availability of pilots before getting next eligible pilots
            self._update_pilot_availability()
            actions = self.getActions()
            buffer_score = self._get_buffer_reward(action)
            moveup_score = self._get_moveup_reward(action)

        # Check if all open slots have been filled
        all_filled = True
        for flight in schedule.values():
            if None in flight['pilots_assigned']:
                all_filled = False
                break
        if all_filled:
            if self.verbose:
                print_dash()
                print("Filled all slots with pilots. Can move to next week")
                print_dash()
        actions = self.getActions()

        return buffer_score, moveup_score, actions

    def _update_pilot_availability(self) -> None:
        """
            Once the pilots are assigned to an event,
            mark them as unavailable for that duration.
            Also, remove all the events from history which 
            have been scheduled before.
        """
        # to_remove = []
        pilots_to_remove = []
        schedule = self.history['schedule']
        # if all pilots assigned to an event remove it from self.history['schedule']
        for i, flight in schedule.items():
            if flight["start"] <= self.current_date:
                pilots_to_remove.append(i)
        for index in pilots_to_remove:
            flightInfo = schedule[index]
            removed_pilots = [i for i in flightInfo["pilots_assigned"] if i != None]
            for pilot in removed_pilots:
                pilotRecord = self.history["pilots"][pilot]
                pilotUnavailability = pilotRecord["unavailability"]
                # Push to unavailability and pilotsFlying
                if (flightInfo["start"], flightInfo["end"]) not in pilotUnavailability:
                    pilotUnavailability.append((flightInfo["start"], 
                                                flightInfo["end"]))
                self.pilotsFlying[pilot].append((flightInfo["start"], 
                                                flightInfo["end"]))

        # Update queue of new flights that need to be scheduled
        self._updateToFill()

    # all state representation methods below this
    def get_state(self):
        """
            Will return a tuple of:
                (pilot_availability_matrix, valid_pilots, 
                event_info, event_type_requirements)
            where 
                event_info = (event_one_hot, pilot_allotment, 
                        duration, start_day_num, end_day_num)
            ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
            pilot_availability_matrix: 
                [max_duration, num_pilots] shaped np.ndarray (Optional)
                binary matrix indicating the availability of pilots for all days
            valid_pilots: [num_pilots] shaped np.ndarray
                pilots which are available for the current event
            event_one_hot: [num_event_types] shaped np.ndarray 
                one hot vector indicating which event type it is
            pilot_allotment: [num_pilots] shaped np.ndarray
                zero or one-hot or two-hot vector
            duration: int
                indicating how long the current event is; 
                normalised with max_event_duration
            start_day_num: int 
                indicating how many days after the start of the episode is the 
                start date
            end_day_num: int
                indicating how many days would have elapsed since the start of 
                the episode for the end date
            event_type_requirements: [num_pilots] shaped np.ndarray
                The requirements for each pilot for the current event 
            ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
            Returns 
                ([max_duration, num_pilots], [num_event_types], 
                [num_pilots], [3], [max_nodes, feat_dim])
                OR
                ([max_duration, num_pilots], [num_event_types], 
                [2], [num_pilots], [3], [max_nodes, feat_dim])
            Example:
                (14x87, 13, 87, 3, 40x103)
                (14x87, 16, 2, 87, 3, 40x106)
            This will basically be the state used by the RL agent
        """
        # PA_look_aheadx87
        pilot_availability_matrix = self._get_pilot_availability(self.current_date, 
                                self.args.max_duration)[:self.args.PA_look_ahead]
        valid_pilots = np.zeros(self.args.num_pilots)
        # if episode complete and no more events, then return a terminal state
        if (self.validActions == [None]) or (len(list(self.toFill)) < 1):
            if self.args.modify_terminal_PA:
                pilot_availability_matrix = -1 * np.ones((self.args.PA_look_ahead, 
                                                self.args.num_pilots))# 7 x 87
            valid_pilots = -1 * np.ones(self.args.num_pilots)                                      
            processed_event = -1 * np.ones(self.args.num_pilots + 
                            self.args.num_event_types + 3 + 3 + 
                            (2 if self.args.use_training_req else 0))
            buffer_vec = -1 * np.ones(self.args.num_pilots)
            moveup_vec = -1 * np.ones(self.args.num_pilots)
            req_vec = -1 * np.ones(self.args.num_pilots)
            return (pilot_availability_matrix, valid_pilots, processed_event, 
                    req_vec, buffer_vec, moveup_vec)
        else:
            if self.validActions != self.args.no_valid_moves:
                # make all pilots available equal to 1
                valid_pilots[[self.pilotToIndex[pilot[0]] 
                            for pilot in self.validActions]] = 1
            event_num, slot = list(self.toFill)[0]
            current_event = self.history["schedule"][event_num] # this is a dict
            processed_event = self._process_event(current_event, slot)
            buffer_vec = np.array([self._buffer(pilot_id, event_num) 
                                    for pilot_id in self.pilotList])
            moveup_vec = np.array([self._move_ups(pilot_id, event_num) 
                                    for pilot_id in self.pilotList])
            req_vec = self._get_event_requirement_vec(current_event)
            return (pilot_availability_matrix, valid_pilots, processed_event, 
                    req_vec, buffer_vec, moveup_vec)

    def _get_event_requirement_vec(self, event:Dict):
        """
            Given the event return a vector of the number of training 
            requirements left for each pilot for that partilcular event
            Parameter:
            ––––––––––
            event: Dict
            Example:
                {'end': datetime.date(2020, 10, 5),
                'pilots_assigned': [None, None],
                'requirements': ['MP+', 'FPC+'],
                'start': datetime.date(2020, 9, 28),
                'AR': False,    (Optional)
                'night': False, (Optional)
                'type': 'Basic'}
            Returns:
            ––––––––
            req_vec: [num_pilots] sized np.ndarray
        """
        event_name = event['type']
        event_B = event['B']
        event_A = event['A']
        event_type = event_name + event_B * (' B') + event_A * (' A')
        ########################################
        req_vec = np.zeros(self.args.num_pilots)
        ########################################
        # if this event satisfies some training requirement
        if event_type in list(self.tr_event_mapping.keys()):
            # example: ['Total AR Flights', ..., 'AR every 60 days']
            event_type_reqs = self.tr_event_mapping[event_type]
            # sum all the requirement types it can satisfy
            for event_name in event_type_reqs:
                col_num = self.event_trs.index(event_name)
                req_vec += self.requirement_matrix[:,col_num]
        # else just return a zero vector
        return req_vec

    def _process_event(self, event:Dict, slot_num:int):
        """
            Param event: dict
            Example:
                {'end': datetime.date(2020, 10, 5),
                'pilots_assigned': [None, None],
                'requirements': ['MP+', 'FPC+'],
                'start': datetime.date(2020, 9, 28),
                'type': 'Basic'}

            Return:
                • A 13(or)16-dim one-hot vector which is hot at that event-type 
                    position
                • A 87-dim zero/one-hot/two-hot vector which is hot at the 
                    pilots assigned to that event
                • The current event duration in days.
                • The current event start day number from episode start date.
                • The current event end day number from episode start date.
            Example: 
            type='Basic', pilots_assigned=[2,None], 
            start=datetime.date(2020, 9, 28), end= datetime.date(2020, 10, 5),
            output = [0,0,0,0,0,0,0,0,0,0,1,0,0],[0,0,1,0,...,0], 7, 1, 8
            output shape = 13 + 87 + 3 = 103
                                OR
            output = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],[0,0], 0,0,1,0,...,0], 7, 1, 8
            output shape = 16 + 2 + 87 + 3 = 108
            Returns:
            --------
            np.ndarray
        """
        ################## event one-hot ##################
        B_A = None
        if self.args.use_training_req:
            B_A = np.zeros(2)
            if event['B']:
                B_A[0] = 1
            if event['A']:
                B_A[1] = 1

        if self.args.use_event_type:
            event_type = event['type']
            # [0,0,0,0,0,0,0,0,0,0,0,0,0]
            event_one_hot = np.zeros(len(self.event_type_mapping))
            event_one_hot[self.event_type_mapping[event_type]] = 1
        else:
            slot_req = event['requirements'][slot_num]
            # [0,0,0,0,0,0,0,0,0,0,0,0,0]
            event_one_hot = np.zeros(len(self.req_mapping))
            event_one_hot[self.req_mapping[slot_req]] = 1
        ###################################################

        ################## event-pilot allotment ##################
        pilot_allotment = np.zeros(self.args.num_pilots)
        for pilots_assigned_num in event['pilots_assigned']:
            if pilots_assigned_num: # if not None
                pilot_allotment[self.pilotToIndex[pilots_assigned_num]] = 1
        ###########################################################
        event_start = event['start']
        event_end   = event['end']
        ################# event duration ################
        duration = (event_end - event_start).days + 1
        #################################################

        ########## event start day number ###############
        start_day_num = (event_start - self.episode_start_date).days
        #################################################

        ########### event end day number ################
        end_day_num = (event_end - self.episode_start_date).days
        #################################################
        if self.args.normalise_state:
            # divide by 7 or 14 depending on how long the episode is
            duration = duration / self.args.max_duration
            start_day_num = start_day_num / self.args.max_duration  
            end_day_num = end_day_num / self.args.max_duration      

        if self.args.use_training_req:
            processed_event = np.hstack((event_one_hot, B_A, pilot_allotment, 
                                        duration, start_day_num, end_day_num))
        else:
            processed_event = np.hstack((event_one_hot, pilot_allotment, 
                                        duration, start_day_num, end_day_num))
        return processed_event

    def process_state(self, pilot_availability_matrix:np.ndarray, 
                    valid_pilots:np.ndarray, processed_event:np.ndarray,
                    req_vec:np.ndarray,
                    buffer_vec:np.ndarray, moveup_vec:np.ndarray):
        """
            Process the state obtained by get_state() method
            so that we can directly feed it into the network
            flatten everything and concatenate
            If use_pa_matrix:
                add pilot availability matrix in the state
            Parameters:
            –––––––––––
            pilot_availability_matrix:np.ndarray
                [max_duration x num_pilots] shaped array
            valid_pilots:np.ndarray x
                [num_pilots] shaped array
            processed_event:np.ndarray x
                [num_event_types + num_pilots + 3] shaped array
                OR
                [num_event_types + 2 + num_pilots + 3] shaped array
            Returns:
            ––––––––
            np.ndarray
        """
        pilot_availability_matrix = pilot_availability_matrix.flatten()
        state = np.concatenate((valid_pilots, processed_event))
        if self.args.use_pa_matrix:
            state = np.concatenate((pilot_availability_matrix, state))
        if self.args.use_training_req:
            state = np.concatenate((req_vec, state))
        if self.args.include_moveup_buffer_vec:
            state = np.concatenate((buffer_vec, moveup_vec, state))
        return state

    def _get_pilot_availability(self, start:datetime.time, duration:int):
        """
            Get the availability matrix of the pilots for the next 
            'max_duration' days
            Parameters:
            –––––––––––
            start: datetime.date(2020, 10, 7)
            returns max_duration x num_pilots sized binary matrix
            1 if available; 0 if not available
            Returns:
            --------
            np.ndarray
        """
        availability_matrix = np.ones((duration, self.args.num_pilots))
        end = start + timedelta(duration)
        for pilot_num in self.history['pilots']:
            # pilot_num: '10000'
            # contains the durations for which pilots are unavailable
            pilot_dates = self.history['pilots'][pilot_num]['unavailability']
            pilot_index = self.pilotToIndex[pilot_num]
            for date in pilot_dates:
                s,e = date
                for i in range((e-s).days + 1):
                    day = s+timedelta(i)
                    if start <= day <= end-timedelta(1):
                        availability_matrix[(day-start).days][pilot_index] = 0
        return availability_matrix
