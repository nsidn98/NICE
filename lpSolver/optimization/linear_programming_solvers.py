from collections import defaultdict
from ortools.linear_solver import pywraplp
from datetime import date
from pprint import pprint
from pilotRLEnv.utils import satisfiableMember, checkLeaveConflict
import copy

from ..optimization import linear_programming_encoders as lp_encoders


def getSolverFailureStatusString(status) -> str:
  vars = {
    pywraplp.Solver.ABNORMAL: "ABNORMAL",
    pywraplp.Solver.FEASIBLE: "FEASIBLE",
    pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
    pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
    pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
  }
  return vars[status]

# Note: these lists, dictionaries, and computeOverqual were taken from pilotRLenv.
# Depending on where this goes, we may want to refactor to have one "source of truth"
# between the LP and RL environments

def checkPossible(person, event, schedule):
  if checkLeaveConflict(person, event):
    return False
  if not any(satisfiableMember(slot_requirement, person["qualification"])
                   for slot_requirement in event["requirements"]):
    return False
  return True

def solve_lp(schedule, objective_fxn, obj_data=None):
  # Returns LP that returns feasible solution to staff assignment

  # Create the MIP solver with the Gurobi backend.
  solver = pywraplp.Solver(objective_fxn, pywraplp.Solver.GUROBI_MIXED_INTEGER_PROGRAMMING)
  solver.SetSolverSpecificParametersAsString("Seed=0")

  # Initialize variables.
  personnel_assignments = lp_encoders.init_vars_personnel_event_slot(schedule, solver, False)

  # Indicate if person is assigned to event.
  if objective_fxn in {"moveup", "buffer"}:
    pilot_on_event = {}
    for pilotID in schedule["pilots"].keys():
      for eventID, event in schedule["schedule"].items():
        var_name = "Person: " + pilotID + "  Event: " + str(eventID) + " Assigned"
        pilot_on_event[(pilotID, eventID)] = solver.IntVar(0, 1, var_name) 
        constraint_name = var_name
        solver.Add(pilot_on_event[(pilotID, eventID)] == sum(personnel_assignments[(pilotID, eventID, slot_number)] 
                                                                  for slot_number in range(len(event["requirements"]))),
                   constraint_name)      

  if objective_fxn == "buffer":
    # Buffers
    event_buffer_penalties = []
    T_buffer = 4
    for event1ID, event1 in schedule["schedule"].items():
      for event2ID, event2 in schedule["schedule"].items():
        if event1ID == event2ID:
          continue
        buffer = (event2["start"] - event1["end"]).days - 1
        if 0 <= buffer <= T_buffer:
          penalty = - (T_buffer + 1 - buffer) / (T_buffer + 1)
          event_buffer_penalties.append((
            event1ID,
            event2ID, 
            event1,
            event2,
            penalty
          ))
          assert penalty < 0, "Invalid penalty"
    print("event buffer penalties length:", len(event_buffer_penalties))
    P_if = defaultdict(list)
    buffer_tuples = []
    for pilotID, pilot in schedule["pilots"].items():
      for (e1ID, e2ID, e1, e2, penalty) in event_buffer_penalties:
        if not checkPossible(pilot, e1, schedule):
          continue
        if not checkPossible(pilot, e2, schedule):
          continue
        buffer_tuples.append((pilotID, pilot, e1ID, e2ID, e1, e2, penalty))
        P_if[(pilotID, e1ID)].append((e2ID, e2))
    print(len(buffer_tuples))
    assigned_both = lp_encoders.init_vars_assigned_both(schedule, solver, buffer_tuples, pilot_on_event, P_if)

  if objective_fxn == "moveup":
    # Move-up crews
    print("Build moveup tuples")

    swap_flights = {fID: [] for fID in schedule["schedule"].keys()}
    T_moveup = 2
    for fID, f in schedule["schedule"].items():
      for gID, g in schedule["schedule"].items():
        if fID == gID:
          continue
        buffer = (g["start"] - f["start"]).days
        if buffer <= T_moveup and buffer >= 0 and g["end"] >= f["end"]:
          swap_flights[fID].append((gID, g))
    moveup_tuples = []
    print("total swaps!", sum(len(v) for v in swap_flights.values()))
    for jID, j in schedule["pilots"].items():
      for fID, f in schedule["schedule"].items():
        for gID, g in swap_flights[fID]:
          if checkLeaveConflict(j, f):
            continue
          for slot_num, req in enumerate(g["requirements"]):
            if satisfiableMember(req, j["qualification"]):
              moveup_tuples.append((jID, j, gID, g, fID, f, slot_num))

    print("moveup tuples built, create and constrain vars:")
    print("len movup tuples:", len(moveup_tuples))
    moveup_vars = lp_encoders.init_vars_moveup(schedule, solver, moveup_tuples, pilot_on_event)
    
  # Given a personnel and event, at most 1 slot can be assigned to that personnel.
  lp_encoders.add_constraint_personnel_single_slot(schedule, personnel_assignments, solver)

  # Given an event slot, exactly 1 personnel needs to occupy that slot.
  lp_encoders.add_constraint_fill_slot(schedule, personnel_assignments, solver)

  # Flight Conflict Constraints - No pilot can fly on two missions if they conflict
  lp_encoders.add_constraint_flight_conflicts(schedule, personnel_assignments, solver)

  # Define the objective.
  objective = solver.Objective()
  if objective_fxn == "feasibility":
    for pilotID, pilot in schedule["pilots"].items():
      for eventID, event in schedule["schedule"].items():
        for slot_number in range(len(event["requirements"])):
          objective.SetCoefficient(personnel_assignments[(pilotID, eventID, slot_number)], 1)
  elif objective_fxn == "buffer":
    for pilotID, pilot, e1ID, e2ID, e1, e2, penalty in buffer_tuples:
      objective.SetCoefficient(assigned_both[(pilotID, e1ID, e2ID)], penalty)
  elif objective_fxn == "moveup":
    for jID, j, gID, g, fID, f, slot_num in moveup_tuples:
      objective.SetCoefficient(moveup_vars[(jID, gID, fID, slot_num)], 1)
  elif objective_fxn == "CARLO":
    assert obj_data is not None, "No data passed in for CARLO coefficients."
    for pilotID, pilot in schedule["pilots"].items():
      for eventID, event in schedule["schedule"].items():
        for slot_number in range(len(event["requirements"])):

          key = (int(eventID), slot_number, str(pilotID))
          prob_val = obj_data[key]

          objective.SetCoefficient(personnel_assignments[(pilotID, eventID, slot_number)], prob_val)
      

  objective.SetMaximization()
  print("Start solving")
  # Solve.
  status = solver.Solve()

  # Display solution.
  returnSchedule = None
  if status == pywraplp.Solver.OPTIMAL:
    print()
    print('Objective value =', solver.Objective().Value())
    print('Problem solved in %f milliseconds' % solver.wall_time())
    print('Problem solved in %d iterations' % solver.iterations())
    print('Problem solved in %d branch-and-bound nodes' % solver.nodes())
    print()
    returnSchedule = copy.deepcopy(schedule["schedule"])

    for pilotID, pilot in schedule["pilots"].items():
      for eventID, event in schedule["schedule"].items():
        for slot_number in range(len(event["requirements"])):
          if personnel_assignments[(pilotID, eventID, slot_number)].solution_value() > 0.9:
            returnSchedule[eventID]["pilots_assigned"][slot_number] = pilotID

  else:
    print('NO OPTIMAL SOLUTION:', getSolverFailureStatusString(status))
    print('Time: %f milliseconds' % solver.wall_time())
    print('%d iterations' % solver.iterations())
    print('%d branch-and-bound nodes' % solver.nodes())
  return {
    "filled_schedule": returnSchedule, 
    "time": solver.wall_time() / 1_000,
    "constraints": solver.NumConstraints(),
    "variables": solver.NumVariables()
  }
  

# old_schedule is completely filled from the previous round of LP scheduling
def both_handle_disruptions(old_schedule, new_schedule): 

  # Create the MIP solver with the Gurobi backend.
  solver = pywraplp.Solver("training_requirements_equal_weight",
                           pywraplp.Solver.GUROBI_MIXED_INTEGER_PROGRAMMING)
  solver.SetSolverSpecificParametersAsString("Seed 0")

  # This is Y 
  new_personnel_assignments = lp_encoders.init_vars_personnel_event_slot(new_schedule, solver, True)
  # This is X 
  old_personnel_assignments = lp_encoders.init_vars_personnel_event_slot(new_schedule, solver, True, old_schedule)
  # Note that all of the old_personnel_assignments are completely constrained.
  # The LP doesn't have to make a choice here!


  # Variables representing the difference between old and new. This is Z 
  difference = {}
  for key in old_personnel_assignments.keys():
    personId, eventId, slot_number = key
    var_name = "Person: " + str(personId) + "  Event: " + str(eventId) + " Slot: " + str(slot_number)
    difference[key] = solver.IntVar(0, 1, var_name)
    
  lp_encoders.add_constraint_difference(
      old_personnel_assignments, new_personnel_assignments, difference, solver)

  # Given a personnel and event, at most 1 slot can be assigned to that personnel
  lp_encoders.add_constraint_personnel_single_slot(
      new_schedule, new_personnel_assignments, solver)

  # Given an event slot, exactly 1 personnel needs to occupy that slot
  lp_encoders.add_constraint_fill_slot(new_schedule, new_personnel_assignments, solver)

  # Flight Conflict Constraints - No pilot can fly on two missions if they conflict
  lp_encoders.add_constraint_flight_conflicts(
      new_schedule, new_personnel_assignments, solver)

  # Define the objective.
  objective = solver.Objective()
  for pilotID, pilot in new_schedule["pilots"].items():
    for eventID, event in new_schedule["schedule"].items():
      for slot_number in range(len(event["requirements"])):
        objective.SetCoefficient(difference[(pilotID, eventID, slot_number)], -1)

  objective.SetMaximization()
  # Solve.
  status = solver.Solve()

  # Display solution.
  returnSchedule = None
  if status == pywraplp.Solver.OPTIMAL:
    print()
    print('Objective value =', solver.Objective().Value())
    print('Problem solved in %f milliseconds' % solver.wall_time())
    print('Problem solved in %d iterations' % solver.iterations())
    print('Problem solved in %d branch-and-bound nodes' % solver.nodes())
    print()
    returnSchedule = copy.deepcopy(new_schedule["schedule"])

    for pilotID, pilot in new_schedule["pilots"].items():
      for eventID, event in new_schedule["schedule"].items():
        for slot_number in range(len(event["requirements"])):
          if new_personnel_assignments[(pilotID, eventID, slot_number)].solution_value() > 0.9:
            returnSchedule[eventID]["pilots_assigned"][slot_number] = pilotID

  else:
    print('NO OPTIMAL SOLUTION:', getSolverFailureStatusString(status))
    print('Time: %f milliseconds' % solver.wall_time())
    print('%d iterations' % solver.iterations())
    print('%d branch-and-bound nodes' % solver.nodes())

  return {
    "filled_schedule": returnSchedule, 
    "time": solver.wall_time() / 1_000
  }


