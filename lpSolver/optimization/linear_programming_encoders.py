from __future__ import annotations
import itertools as it
from pilotRLEnv.utils import satisfiableMember, singleOverlap, checkLeaveConflict

from ortools.linear_solver import pywraplp

def init_vars_personnel_event_slot(schedule,
                                   solver:pywraplp.Solver, 
                                   handle_disrupt: bool,
                                   old_schedule=None):
  personnel_assignments = {}
  for pilotID, pilot in schedule["pilots"].items():
    for eventID, event in schedule["schedule"].items():
      for slot_number in range(len(event["requirements"])):
        slot_requirement = event["requirements"][slot_number]
        var_name = "Pilot: " + pilotID + "  Event: " + str(eventID) + " Slot: " + str(slot_number)

        # Initialize Variables based on data.
        if old_schedule is None:
          assignment = schedule["schedule"][eventID]["pilots_assigned"][slot_number]
          if assignment is not None:
            assert handle_disrupt, "pilot filled in on initial schedule creation"
            val = assignment == pilotID
            personnel_assignments[(pilotID, eventID, slot_number)] = solver.IntVar(val, val, var_name)
          if checkLeaveConflict(pilot, event):
            personnel_assignments[(pilotID, eventID, slot_number)] = solver.IntVar(0, 0, var_name) # Leave prevents assignment.
          elif not satisfiableMember(slot_requirement, pilot["qualification"]):
            personnel_assignments[(pilotID, eventID, slot_number)] = solver.IntVar(0, 0, var_name) # Not qualified for slot.
          else:
            personnel_assignments[(pilotID, eventID, slot_number)] = solver.IntVar(0, 1, var_name) # Assignment may be made.
        else:
          pilot_assigned = old_schedule["schedule"][eventID]["pilots_assigned"][slot_number]
          assert pilot_assigned is not None, "no pilot assigned to schedule that should be filled"
          val = pilot_assigned == pilotID
          personnel_assignments[(pilotID, eventID, slot_number)] = solver.IntVar(val, val, var_name) 

  # Assignments indicate whether a person is assigned to an event in a specific slot.
  return personnel_assignments

def init_vars_assigned_both(schedule, solver:pywraplp.Solver, buffer_tuples, pilot_on_event, P_if):
  # Personnel_Training_Requirement_p_r_n = whether person p has completed the nth reptition of training requirement r
  assigned_both = {}
  eid_to_e = {}
  for pilotID, pilot, e1ID, e2ID, e1, e2, _ in buffer_tuples:
    eid_to_e[e1ID] = e1
    assigned1 = pilot_on_event[(pilotID, e1ID)]
    assigned2 = pilot_on_event[(pilotID, e2ID)]
    var_name = "Person: " + pilotID + " Assigned to: " + str(e1ID) + " and " + str(e2ID)
    # Initialize all variables and add relevant constraints
    var = assigned_both[(pilotID, e1ID, e2ID)] = solver.IntVar(0, 1, var_name)
    # Since the objective is a maximization, and it gets negative reward when var is 1,
    # it will assign var 0 if it has the choice between 1 and 0. It will only choose
    # 1 if assigned1 and assigned2 are 0 
    solver.Add(var >= (assigned1 + assigned2) - 1)
  for (personId, fId), others in P_if.items():
    sorted_others = sorted(others, key=lambda e:e[1]["start"])
    for (eID, _), (f_primeID, _) in it.combinations(sorted_others, 2):
      # Buffer penalties can only be assigned in consecutive fashion.
      solver.Add(assigned_both[(personId, fId, eID)] <= 1 - assigned_both[(personId, fId, f_primeID)])
  return assigned_both

def init_vars_moveup(schedule, solver, moveup_tuples, pilot_on_event):
  moveup_vars = {}
  for tup in moveup_tuples:
    jID, j, gID, g, fID, f, slot_num = tup
    var_name = f"Pilot {jID} from flight {gID} can fill slot {slot_num} on flight {fID}"
    var = moveup_vars[(jID, gID, fID, slot_num)] = solver.IntVar(0, 1, var_name)
    # j must be on g
    solver.Add(var <= pilot_on_event[(jID, gID)])
    for oID, o in schedule["schedule"].items():
      if singleOverlap((o["start"], o["end"]), (f["start"], f["end"])) and o["start"] < f["start"]:
        solver.Add(var <= 1 - pilot_on_event[(jID, oID)])
  print("Vars created and constrained")
  return moveup_vars

def add_constraint_personnel_single_slot(schedule, personnel_assignments:dict, solver:pywraplp.Solver):
  # Given a personnel and event, at most 1 slot can be assigned to that personnel.
  for pilotID, pilot in schedule["pilots"].items():
    for eventID, event in schedule["schedule"].items():
      constraint_name = "Personnel: " + pilotID + "  Event: " + str(eventID) + " Single Role"
      constraint = solver.RowConstraint(0, 1, constraint_name)

      for slot_number in range(len(event["requirements"])):
        constraint.SetCoefficient(personnel_assignments[(pilotID, eventID, slot_number)], 1)


def add_constraint_fill_slot(schedule, personnel_assignments:dict, solver:pywraplp.Solver):
  for eventID, event in schedule["schedule"].items():
    for slot_number in range(len(event["requirements"])):
      constraint_name = "Event: " + str(eventID) + "  Slot: " + str(slot_number) + " Personnel Requirement"
      constraint = solver.RowConstraint(1, 1, constraint_name)

      for pilotID in schedule["pilots"].keys():
        constraint.SetCoefficient(personnel_assignments[(pilotID, eventID, slot_number)], 1)


def add_constraint_flight_conflicts(schedule, personnel_assignments:dict, solver:pywraplp.Solver):
  for pilotID, pilot in schedule["pilots"].items():
    for event1ID, event1 in schedule["schedule"].items():
      for event2ID, event2 in schedule["schedule"].items():

        # If the two events conflict ensure personnel are not assigned to both.
        if singleOverlap((event1["start"], event1["end"]), (event2["start"], event2["end"])):
          constraint_name = "Event1: " + str(event1ID) + "  Event2: " + str(event2ID) + "  Personnel: " + pilotID + " Flight Conflict"
          constraint = solver.RowConstraint(0, 1, constraint_name)

          # Consider all slots on the first event.
          for slot_number in range(len(event1["requirements"])):
            constraint.SetCoefficient(personnel_assignments[(pilotID, event1ID, slot_number)], 1)

          # Consider all slots on the second event.
          for slot_number in range(len(event2["requirements"])):
            constraint.SetCoefficient(personnel_assignments[(pilotID, event2ID, slot_number)], 1)

def add_constraint_difference(old_personnel_assignments, new_personnel_assignments, difference, solver):
  for key in old_personnel_assignments.keys():
    # Because each difference has a negative penalty and we want to maximize,
    # if new/old is the same, diff \in {0, 1}, so the solver will pick 0.
    # if new/old is different, then diff >= 1 and diff >= -1, so diff will have to be 1
    solver.Add(difference[key] >= new_personnel_assignments[key] - old_personnel_assignments[key])
    solver.Add(difference[key] >= - (new_personnel_assignments[key] - old_personnel_assignments[key]))
