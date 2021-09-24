from datetime import timedelta
import os
import numpy as np
from collections import Counter, defaultdict
import pandas as pd
import json

DAYS_PER_WEEK = 7

def main():
    simTypes = Counter()
    missionLenthByType = defaultdict(list)
    sims = pd.read_csv(f"./Minimized Dataset - All Months/Sims.csv", parse_dates=["Date"])


    recent_date = None
    first_date = None
    week_sim_counts = Counter()

    for index, simInfo in sims.iterrows():
        simTypes[simInfo["Type"]] += 1
        date = simInfo["Date"].date()

        if first_date is None:
            first_date = date

        since_first = (date - first_date).days
        period_index = since_first // DAYS_PER_WEEK
        week_sim_counts[period_index] += 1

        if recent_date is not None:
            assert date >= recent_date, "Missions not ascending"

        recent_date = date


    recent_start = None
    first_start = None
    week_flight_counts = Counter()

    missions = pd.read_csv(f"./Minimized Dataset - All Months/LocalMissions.csv", parse_dates=["Premission", "Postmission"])
    for index, flightInfo in missions.iterrows():
        end = flightInfo["Postmission"].date()
        start = flightInfo["Premission"].date()

        if first_start is None:
            first_start = start

        since_first = (start - first_start).days
        period_index = since_first // DAYS_PER_WEEK
        week_flight_counts[period_index] += 1

        if recent_start is not None:
            assert start >= recent_start, "Missions not ascending"

        recent_start = start


        length = (end - start).days
        if end > start:
            t = flightInfo["Crew Type"]
            missionLenthByType[t].append({
                "length": length,
                "A": flightInfo.A,
                "B": flightInfo.B
            })

    week_flight_counts = dict(week_flight_counts)
    partial_week_index = max(week_flight_counts.keys())
    del week_flight_counts[partial_week_index]

    week_sim_counts = dict(week_sim_counts)
    partial_week_index_sim = max(week_sim_counts.keys())
    del week_sim_counts[partial_week_index_sim]

    for key in week_sim_counts.keys() - week_flight_counts.keys():
        week_flight_counts[key] = 0

    for key in week_flight_counts.keys() - week_sim_counts.keys():
        week_sim_counts[key] = 0

    flight_vals = list(week_flight_counts.values())
    avg_flights_per_week = np.average(flight_vals)
    std_dev_flights_per_week = np.std(flight_vals)

    sim_vals = list(week_sim_counts.values())
    avg_sims_per_week = np.average(sim_vals)
    std_dev_sims_per_week = np.std(sim_vals)
    print(len(week_sim_counts), len(week_flight_counts))
    print(len(simTypes), len(missionLenthByType))

    simData = {
        "avg": avg_sims_per_week,
        "stddev": std_dev_sims_per_week,
        "type_distribution": dict(simTypes)
    }

    missionData = {
        "avg": avg_flights_per_week,
        "stddev": std_dev_flights_per_week,
        "type_length_distribution": dict(missionLenthByType)
    }

    with open("sim_distribution.json", "w") as f:
        json.dump(simData, f)

    with open("flight_distribution.json", "w") as f:
        json.dump(missionData, f)

if __name__ == "__main__":
    main()