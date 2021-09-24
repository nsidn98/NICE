"""
    Contains all the configuration parameters for the environment
    Use this config when testing the environment
    Use this only when running inside `pilotRLEnv` folder
"""
import argparse
from distutils.util import strtobool

parser = argparse.ArgumentParser(description='Test pilotRL env')

# filepath to data
parser.add_argument('--flight_distribution_path', type=str, 
                    default='pilotRLEnv/data/flight_distribution.json',
                    help='Path to the json file for flight distributions')
parser.add_argument('--tr_path', type=str, 
                    default='pilotRLEnv/data/Minimized Dataset - '
                    'All Months/training_requirements.csv',
                    help='Path to the json file for training requirements')
parser.add_argument('--sim_distribution_path', type=str, 
                    default='pilotRLEnv/data/sim_distribution.json',
                    help='Path to the json file for flight distributions')
parser.add_argument('--folder', type=str, 
                    default="pilotRLEnv/data/Minimized Dataset - All Months/",
                    help='Path to the directory with the csvs containing '
                    'pilot information')
parser.add_argument('--flight_data', type=str, default='LocalMissions.csv',
                    help='Name of the local missions file; used when loading '
                    'events manually')
parser.add_argument('--sim_data', type=str, default='Sims.csv',
                    help='Name of the sims file; used when loading events '
                    'manually')
parser.add_argument('--manual_event_load',type=lambda x:bool(strtobool(x)), 
                    default=False, help='Whether events are to be loaded '
                    'manually from the dataset or random creation')


# state representation
parser.add_argument('--num_pilots', type=int, default=87, 
                    help='Number of pilots taking part in the scheduling process')
parser.add_argument('--PA_look_ahead', type=int, default=7, 
                    help='Number of days to look ahead in the pilot '
                    'availability matrix')
parser.add_argument('--num_event_types', type=int, default=13, 
                    help='Number of types of events')
parser.add_argument('--normalise_state', type=lambda x:bool(strtobool(x)), 
                    default=False, help='Whether the date related features in '
                    'the states are to be normalised or not')
parser.add_argument('--use_pa_matrix', type=lambda x:bool(strtobool(x)), 
                    default=False, help='Whether the pilot availability matrix '
                    'is to be included in the state representation or not')
parser.add_argument('--modify_terminal_PA', type=lambda x:bool(strtobool(x)), 
                    default=True, help='If True, then modify PA matrix as -1 '
                    'for terminal state')
parser.add_argument('--use_training_req', type=lambda x:bool(strtobool(x)), 
                    default=True, help='Whether we want to consider training '
                    'requirements in the state representation')
parser.add_argument('--include_moveup_buffer_vec', 
                    type=lambda x:bool(strtobool(x)), default=False,
                    help='Whether the moveup and buffer vectors are included in '
                    'the state space')
parser.add_argument('--use_event_type',type=lambda x:bool(strtobool(x)), 
                    default=True, help='Use event type in state if true, '
                    'otherwise, use requirement of current slot')

# reward structure
parser.add_argument('--pilot_place_wt', type=float, default=1, 
                    help='Weight given to placing a valid pilot')
parser.add_argument('--buffer_weight', type=float, required=True, 
                    help='Weight given to buffer score')
parser.add_argument('--moveup_weight', type=float, required=True, 
                    help='Weight given to buffer score')

# action space
parser.add_argument('--mask_actions', type=lambda x:bool(strtobool(x)), 
                    default=True, help='If we only want valid pilots in the '
                    'action space of the agent; Having this True will prune '
                    'the probability distributions predicted by actor to only '
                    'pilots who are eligible')

# environment
parser.add_argument('--no_valid_moves', type=str, default="no valid", 
                    help='To check if there are any valid moves')
parser.add_argument('--illegal_action', type=str, default="no illegal", 
                    help='To check for illegal actions')

# parameters
parser.add_argument('--max_duration', type=int, default=7, 
                    help='The maximum game length ')
# heuristic obtained from flight_distribution.json and sim_distribution.json dataset
parser.add_argument('--avg_assignments_week', type=int, default=80, 
                    help='Average number of pilot-event assignments per week')
parser.add_argument('--flight_density',type=int, default=1,
                    help='The density of flights/week, where 1 is average. '
                    '(2 would be twice the average)')

# dates in schedules
parser.add_argument('--START_DATE', type=str, default = '2019-06-15', # this is a Monday
                    help = 'The start date in YYYY-MM-DD')
parser.add_argument('--END_DATE', type=str, default = '2020-01-02', # this is a Saturday
                    help = 'The end date in YYYY-MM-DD')

# seed for reproducibility
parser.add_argument('--seed', type=int, default=0, 
                    help='Seed for all randomness')

def config_check(args:argparse.Namespace):
    """
        Check if the arguments are compatible
    """
    if args.PA_look_ahead > args.max_duration:
        print('_'*50)
        print(f'Changing look ahead for PA matrix from {args.look_ahead} '
        'to {args.max_duration}')
        print('_'*50)
        args.look_ahead = args.max_duration
    return args

args = parser.parse_args()
args = config_check(args)
