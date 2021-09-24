from collections import defaultdict
from datetime import date, timedelta
import random
import json

class HistoryDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__history = dict(*args, **kwargs)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.__history[key] = value

    def getHistory(self):
        return self.__history

def print_dash(num_dash:int=50):
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

with open("pilotRLEnv/data/Minimized Dataset - All Months/eventInfo.json") as f:
    data = json.load(f)
    sims = data["sims"]
    flights = data["flights"]
    requirements = data["requirements"]


def satisfiableMember(req, member):
    return member in requirements[req]

def singleOverlap(one, two):
    return not (one[1] < two[0] or one[0] > two[1])

def overlap(date_range_list_one, date_range_list_two):
    for rng1 in date_range_list_one:
        start_one, end_one = [rng1[i] for i in range(2)]
        for rng2 in date_range_list_two:
            start_two, end_two = [rng2[i] for i in range(2)]

            # if one has any overlap with two
            if singleOverlap((start_one, end_one), (start_two, end_two)):
                return True
    return False

def checkLeaveConflict(pilot, event):
    return overlap(pilot["unavailability"], [(event["start"], event["end"])])
class RandomEventCreator:
    def __init__(self, seed:int):
        """
            Parameters:
            -----------
            seed: int
                Seed for randomness
        """
        random.seed(seed)
    
    def randomSim(self, start:date, end:date, simInfo:dict):
        """
            Create a random simulation event
            start: datetime.date
                The start date of the event
            end: datetime.date
                The end date of the event
            simInfo: dict
                Information about the dataset of simulation events
        """
        dateRange = (end - start).days

        simTypes, simTypeWeights = zip(*simInfo["type_distribution"].items())
        dates = [start + timedelta(days=d) for d in range(dateRange)]
        simType = random.choices(simTypes, simTypeWeights, k=1)[0]
        date = random.choice(dates)
        return {
            "type": simType, 
            "start": date,
            "end": date,
            "A": False,
            "B": False,
            "requirements": sims[simType]["crew"],
            "pilots_assigned": [None] * len(sims[simType]["crew"])
        }

    def randomFlight(self, start:date, end:date, flightInfo:dict):
        """
            Create a random flight event
            start: datetime.date
                The start date of the event
            end: datetime.date
                The end date of the event
            flightInfo: dict
                Information about the dataset of flight events
        """
        flightTypeLengths = flightInfo["type_length_distribution"]

        flightTypes, flightTypeWeights = zip(*[(t, len(durs)) for t, durs in flightTypeLengths.items()])
        chosenFlightType = random.choices(flightTypes, flightTypeWeights, k=1)[0]
        chosenFlight = random.choice(flightTypeLengths[chosenFlightType])
        chosenFlightLength = chosenFlight["length"]
        dateRange = (end - start).days
        dates = [start + timedelta(days=d) for d in range(dateRange)]
        chosenFlightDate = random.choice(dates)
        return {
            "type": chosenFlightType,
            "start": chosenFlightDate,
            "end": chosenFlightDate + timedelta(days=chosenFlightLength),
            "requirements": flights[chosenFlightType],
            "A": chosenFlight["A"],
            "B": chosenFlight["B"],
            "pilots_assigned": [None] * len(flights[chosenFlightType])
        }

def convert_date(date_string:str):
    """
        Convert `date_string` to a datetime.time object
        date_string should be of the form "YYYY-MM-DD"
    """
    dates = date_string.split('-')
    dates = [int(s) for s in dates]
    return date(dates[0], dates[1], dates[2])

def checkValid(schedule, env, original_leave):
    if schedule is None:
        return
    pilotToEvents = defaultdict(list)
    for eventId, event in schedule.items():
        for pilot, req in zip(event["pilots_assigned"], event["requirements"]):
            if pilot is None:
                continue
            pilotToEvents[pilot].append(event)
            pilot_qual = env.history["pilots"][pilot]["qualification"]
            assert satisfiableMember(req, pilot_qual)

    for pilot, events in pilotToEvents.items():
        events.extend(original_leave[pilot])
        events.sort(key=lambda e:e["start"])
        prevEnd = None
        wasPrevLeave = False
        for event in events:
            isLeave = True if event.get("type") is None else False
            # If both are leave, ignore the conflict
            if prevEnd and event["start"] <= prevEnd and not (isLeave and wasPrevLeave):
                assert False
            prevEnd = event["end"]
            wasPrevLeave = isLeave

def calc_disruptions(before_schedule, after_schedule):
    """Calculates schedule disruptions
       
    Calculates the number of changes to flight-slot assignments in 
    after_schedule relative to existing flights in before_schedule.
    (New flights don't count as a disruption.)

    Args:
        before_schedule (env_schedule): Schedule before the disruption
        after_schedule (env_schedule): Schedule after the disruption
    """
    # key: (flightId, slotNumber)
    # value: pilotId
    slotToAssignment = {}
    pilotToEvents = defaultdict(list)
    for flightId, flight in before_schedule.items():
        for slotNumber, pilotId in enumerate(flight["pilots_assigned"]):
            if pilotId is not None: #, "Tried to calculate disruptions in an incomplete schedule"
                slotToAssignment[(flightId, slotNumber)] = pilotId
                pilotToEvents[pilotId].append((flightId, flight))

    changes = 0
    moveups = 0
    callups = 0
    for flightId, flight in after_schedule.items():
        for slotNumber, pilotId in enumerate(flight["pilots_assigned"]):
            oldAssignment = slotToAssignment.get((flightId, slotNumber))
            # Not a new flight, slot assigned to new pilot
            if oldAssignment != pilotId:
                changes += 1
                newPilotsOldEvents = pilotToEvents[pilotId]
                is_moveup = False
                for eventId, oldEvent in newPilotsOldEvents:
                    if singleOverlap((oldEvent["start"], oldEvent["end"]), (flight["start"], flight["end"])):
                        if flight["start"] <= oldEvent["start"] and flight["end"] <= oldEvent["end"]:
                            is_moveup = True
                            break
                if is_moveup:
                    # Now, check if its a swap
                    moveups += 1
                else:
                    callups += 1

    return changes, moveups, callups


