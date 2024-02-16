"""
Main File for running the user study interface
"""
import os
import sys

sys.path.append(os.getcwd())
from pomcp_solvers.solver import *
from pomcp_solvers.root_node import *
from pomcp_solvers.robot_action_node import *
from pomcp_solvers.human_action_node import *
from frozen_lake.frozen_lake_interface import *
from frozen_lake.frozen_lake_map import MAPS, FOG, HUMAN_ERR, ROBOT_ERR
import time
import pygame
import string
import json

# Experiment order --> index 0: Bayes-POMCP;
#                      index 1: Adversarial Bayes-POMCP;
#                      index 2: Heuristic
order = [0, 1, 2]
# random.shuffle(order)
practice_rounds = [0, 1, 2, 3]  # 1 Demo round by the experimenter followed by three practice rounds for the user.

heuristic_order = [0, 1]  # First index is the order of interrupting agent, second is the order of taking control agent.
random.shuffle(heuristic_order)

# Two consecutive rounds have the same condition
# Mapping between the round number and the experiment condition
CONDITION = {
    'practice': practice_rounds,
    'pomcp': [2 * order[0] + len(practice_rounds), 2 * order[0] + len(practice_rounds) + 1],
    'pomcp_inverse': [2 * order[1] + len(practice_rounds), 2 * order[1] + len(practice_rounds) + 1],
    'interrupt': [2 * order[2] + len(practice_rounds) + heuristic_order[0]],
    'take_control': [2 * order[2] + len(practice_rounds) + heuristic_order[1]]
}

expOrder = [0, 1, 2, 3, order[0],
            order[0] + 1, order[1],
            order[1] + 1, order[2] + heuristic_order[0],
            order[2] + heuristic_order[1]]

mapOrder = [10, 5, 7, 4, 11, 12]
# random.shuffle(mapOrder)  # Shuffle the maps for the user study
mapOrder = [0, 2, 3, 6] + mapOrder
print("mapOrder", mapOrder)

username = ''.join(
    random.SystemRandom().choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(5))
filename = 'frozen_lake/files/user_study/' + username + ".json"
userdata = {}
userdata['expOrder'] = expOrder
userdata['mapOrder'] = mapOrder


class HeuristicAgent:
    def __init__(self, intervention_type):
        """
        Initializes a robot agent using a heuristic policy to choose actions (interrupt/take control)
        :param intervention_type: (type: int) The action type of intervention
                                             (i.e., interrupt/take control + explanation/no explanation)
        """
        self.robot_action = 0
        self.type = intervention_type
        self.max_interrupts = 3  # Number of times the heuristic agent interrupts continuously at the same step
        self.num_interrupt = 0  # Number of interruption when taking a longer path (<3)

    def get_action(self, env):
        """
        Return a robot action following a heuristic policy.
        The robot will choose a pre-defined intervention action when 1. the robot thinks next state will be dangerous,
        2. the robot thinks the human action will lead to a longer path.

        :param env: (type: Environment) Instance of the FrozenLake environment
        :return: (type: int) a robot action index
        """
        position = env.world_state[0]
        last_position = env.world_state[1]
        robot_slippery = env.world_state[3]
        robot_err = env.world_state[5]
        last_path = env.find_shortest_path(env.desc, robot_slippery, last_position, env.ncol)
        current_path = env.find_shortest_path(env.desc, robot_slippery, position, env.ncol)

        if env.desc[position // env.ncol, position % env.ncol] in b'HS' and \
                ((position // env.ncol, position % env.ncol) not in env.robot_err or
                 (position // env.ncol, position % env.ncol) in robot_err):
            self.robot_action = self.type

        elif env.desc[position // env.ncol, position % env.ncol] in b'F' and \
                ((position // env.ncol, position % env.ncol) in env.robot_err and
                 (position // env.ncol, position % env.ncol) not in robot_err):
            self.robot_action = self.type

        elif 1 < len(last_path) <= len(current_path) and self.num_interrupt < self.max_interrupts:
            self.robot_action = self.type
            self.num_interrupt += 1

        else:
            self.robot_action = 0
        return self.robot_action


class Driver:
    def __init__(self, env, solver, num_steps, agent=None, max_detection=5):
        """
        Initializes a driver : uses particle filter to maintain belief over hidden states,
        and uses POMCP to determine the optimal robot action

        :param env: (type: Environment) Instance of the FrozenLake environment
        :param solver: (type: POMCPSolver) Instance of the POMCP Solver for the robot policy
        :param num_steps: (type: int) Episode length
        :param agent: (type: RobotPolicy) Set agent type for heuristic agents only
        :param max_detection: (type: int) Maximum number of times the users can utilize the detection sensor
        """
        self.env = env
        self.solver = solver
        self.num_steps = num_steps
        self.agent = agent
        self.max_detection = max_detection  # Number of times the user can use the sensor for detecting slippery grids

        # Env world state:
        #   Current position of the robot;
        #   Previous Position of the robot;
        #   Slippery states encountered thus far that the robot thinks the human knows
        #   Slippery states encountered thus far that the robot knows
        #   Human errors thus far in identifying slippery regions
        #   Robot errors thus far in identifying slippery regions
        self.num_world_states = len(self.env.world_state)

        # Augmented state: World state + latent state (i.e., compliance)
        self.num_total_states = self.num_world_states + 1

    def invigorate_belief(self, current_human_action_node, parent_human_action_node, robot_action, human_action, env):
        """
        Invigorates the belief space when a new human action node is created
        Updates the belief to match the world state, whenever a new human action node is created
        :param current_human_action_node: (type: HumanActionNode) Current human action node is the hao node.
        :param parent_human_action_node: (type: HumanActionNode) Parent human action node is the h node (root of the search tree).
        :param robot_action: (type: int) Robot action (a) taken after parent node state
        :param human_action: (type: List) Human action (o) in response to the robot's action
        :param env: (type: Environment) gym env object to determine current world state of the Frozen Lake
        """
        # Parent human action node is the h node (root of the current search tree).
        # Current human action node is the hao node.

        for belief_state in parent_human_action_node.belief:
            # Update the belief world state for the current human action node
            # if the belief of the parent human action node is the same as the actual world state

            # Update parent belief state to match world state (i.e., after robot action)
            belief_state = env.augmented_state_transition(belief_state, robot_action, None)

            if belief_state[:self.num_world_states] == env.world_state:
                next_augmented_state = env.augmented_state_transition(belief_state, None, human_action)
                current_human_action_node.update_belief(next_augmented_state)
            else:
                print("Node belief is empty!!! Particle Reinvigoration failed!!!")

    def updateBeliefWorldState(self, human_action_node, env):
        """
        Updates the world state in the belief if there are any discrepancies...
        :param human_action_node: (type: HumanActionNode) Current human action node is the hao node.
        :param env: (type: Environment) gym env object to determine current world state of the Frozen Lake
        """
        if len(human_action_node.belief) == 0:
            print("Belief in the current node is empty, i.e., no particles!!!")
            return
        # Update the belief (i.e., all particles) in the current node to match the current world state
        if human_action_node.belief[0][:self.num_world_states] != env.world_state:
            human_action_node.belief = [env.world_state[:] + [belief[-1]] for belief in human_action_node.belief]

    def updateBeliefTrust(self, human_action_node, human_action):
        """
        Update the latent states in the belief for the next root node of the search tree
        :param human_action_node: (type: HumanActionNode) Current human action node is the hao node.
        :param human_action: (type: List) Human action (o) in response to the robot's action
        """

        human_accept, detect, human_choice_idx = human_action  # human accept: 0:no-assist, 1:accept, 2:reject

        for belief in human_action_node.belief:
            if human_accept != 0:  # In case of robot assistance
                # Update trust in particle (belief)
                # index 0 of the particle is acceptance count, and index 1 is rejection count
                belief[self.num_world_states][human_accept - 1] += 1

    def render_score(self, tmp, round_num, final_env_reward, step, detecting_num):
        """
        Render the current round number, score, remaining steps, user id, remaining detection numbers in the interface
        :param tmp: Pygame window interface
        :param round_num: (type: int): the round number of the game
        :param final_env_reward: (type: int): The current score the user gets
        :param step: (type: int): the number of steps the user has moved
        :param detecting_num: (type: int): the number of detections the user has used

        """
        if round_num == 0:
            x = font.render(
                "Demo {}".format(round_num + 1),
                True, (0, 0, 0))
            tmp.blit(x, (5, 5))
            x = font.render(
                "Score: {}".format(int(final_env_reward)),
                True, (0, 0, 0))
            tmp.blit(x, (5, 30))
            x = font.render(
                "Steps Left: {}".format(max(self.num_steps - step, 0)),
                True, (0, 0, 0))
            tmp.blit(x, (5, 55))
            x = font.render(
                "ID: " + username,
                True, (0, 0, 0))
            tmp.blit(x, (5, 80))
            x = font.render(
                "Detections Left: {}".format(self.max_detection - detecting_num),
                True, (0, 0, 0))
            tmp.blit(x, (5, 105))
        elif round_num in CONDITION['practice']:
            x = font.render(
                "Practice {}".format(round_num),
                True, (0, 0, 0))
            tmp.blit(x, (5, 5))
            x = font.render(
                "Score: {}".format(int(final_env_reward)),
                True, (0, 0, 0))
            tmp.blit(x, (5, 30))
            x = font.render(
                "Steps Left: {}".format(max(self.num_steps - step, 0)),
                True, (0, 0, 0))
            tmp.blit(x, (5, 55))
            x = font.render(
                "ID: " + username,
                True, (0, 0, 0))
            tmp.blit(x, (5, 80))
            x = font.render(
                "Detections Left: {}".format(self.max_detection - detecting_num),
                True, (0, 0, 0))
            tmp.blit(x, (5, 105))
        else:
            x = font.render(
                "Round {}".format(round_num - 3),
                True, (0, 0, 0))
            tmp.blit(x, (5, 5))
            x = font.render(
                "Score: {}".format(final_env_reward),
                True, (0, 0, 0))
            tmp.blit(x, (5, 30))
            x = font.render(
                "Steps Left: {}".format(max(self.num_steps - step, 0)),
                True, (0, 0, 0))
            tmp.blit(x, (5, 55))
            x = font.render(
                "ID: " + username,
                True, (0, 0, 0))
            tmp.blit(x, (5, 80))
            x = font.render(
                "Detections Left: {}".format(self.max_detection - detecting_num),
                True, (0, 0, 0))
            tmp.blit(x, (5, 105))
        pygame.display.update()

    def execute(self, round_num):
        """
        Executes one round of search with the POMCP policy
        :param round_num: (type: int) the round number of the current execution
        :return: (type: float) final reward from the environment
        """
        robot_actions = []
        human_actions = []
        all_states = []

        # create a deep copy of the env and the solver
        env = self.env
        solver = self.solver

        print("Execute round {} of search".format(round_num))
        start_time = time.time()
        final_env_reward = 0
        step = 0

        # Initial human action
        robot_action = (0, None)  # No interruption

        # Keyboard input
        action = None
        detecting = 0
        is_accept = 0
        truncated = False
        detection_num = 0

        tmp = env.window_surface
        history = []
        data = {}

        while action is None:
            # Show scores
            tmp.fill((255, 255, 255), rect=(0, 0, 256, 256))
            self.render_score(tmp, round_num, final_env_reward, step, detection_num)

            for event in pygame.event.get():
                if event.type == pygame.KEYUP:
                    if event.key == pygame.K_BACKSPACE:
                        if detecting:
                            detecting = 0
                            env.render(round_num, None, None, env.world_state)
                        else:
                            detecting = 1
                            env.render(round_num, None, None, env.world_state)
                    if event.key == pygame.K_LEFT:
                        action = 0
                    if event.key == pygame.K_RIGHT:
                        action = 2
                    if event.key == pygame.K_UP:
                        action = 3
                    if event.key == pygame.K_DOWN:
                        action = 1
        human_action = tuple([is_accept, detecting, action])
        if detecting:
            detection_num += 1

        data['human_action'] = human_action

        last_robot_action = [0, None]

        # Here we are adding to the tree as this will become the root for the search in the next turn
        human_action_node = HumanActionNode(env)

        # We call invigorate belief when we add a new human action node to the tree
        self.invigorate_belief(human_action_node, solver.root_action_node, robot_action, human_action, env)
        solver.root_action_node = human_action_node
        env.world_state = env.world_state_transition(env.world_state, robot_action, human_action)

        if round_num in CONDITION['pomcp_inverse']:
            final_env_reward -= env.reward(env.world_state, robot_action, human_action)
        else:
            final_env_reward += env.reward(env.world_state, robot_action, human_action)
        all_states.append(env.world_state[0])
        human_actions.append(human_action)

        while True:
            # One extra step penalty if using detection
            if human_action[1] == 1:
                step += 2
            else:
                step += 1

            # Show scores
            tmp.fill((255, 255, 255), rect=(0, 0, 256, 256))
            self.render_score(tmp, round_num, final_env_reward, step, detection_num)

            if round_num == 0:
                # Hard coded for first round -- Demo by experimenter
                robot_action_type = 0
                if env.world_state[0] == 9:
                    robot_action_type = 2
                if env.world_state[0] == 3:
                    robot_action_type = 1
            elif round_num == 1:
                if human_action[1] == 0:
                    _ = solver.search()
                robot_action_type = 0
            elif round_num in [2, 3]:
                # epsilon-greedy pomcp agent  --> slightly random agent for the practice rounds
                # So that users don't get biased or familiar with the POMCP agents in practice rounds
                if human_action[1] == 1:
                    robot_action_type = 0
                else:
                    robot_action_type = solver.search()
                    if np.random.uniform() < 0.5:
                        robot_action_type = np.random.choice([0, 1, 2, 3, 4],
                                                             p=[0.5, 0.5 / 4, 0.5 / 4, 0.5 / 4, 0.5 / 4])
            elif round_num in CONDITION['pomcp'] + CONDITION['pomcp_inverse']:
                if human_action[1] == 1:
                    robot_action_type = 0
                else:
                    # Here the robot action indicates the type of assistance
                    robot_action_type = solver.search()  # One iteration of the POMCP search
            else:
                _ = solver.search()  # Make the heuristic agent execution slower by running POMCP in the background
                if last_robot_action[0] or human_action[1]:
                    robot_action_type = 0  # Cannot interrupt twice successively
                else:
                    robot_action_type = self.agent.get_action(env)

            robot_action = env.get_robot_action(env.world_state[:6], robot_action_type)
            robot_action_node = solver.root_action_node.robot_node_children[robot_action[0]]

            if robot_action_node == "empty":
                # We're not adding to the tree here
                # It doesn't matter because we are going to update the root from h to hao
                robot_action_node = RobotActionNode(env)

            if round_num in CONDITION['practice']:
                data['condition'] = 'practice'
            elif round_num in CONDITION['pomcp']:
                data['condition'] = 'pomcp'
            elif round_num in CONDITION['pomcp_inverse']:
                data['condition'] = 'pomcp_inverse'
            elif round_num in CONDITION['interrupt']:
                data['condition'] = 'interrupt'
            elif round_num in CONDITION['take_control']:
                data['condition'] = 'take_control'

            last_robot_action = robot_action
            data['robot_action'] = robot_action

            # Update the environment
            env.world_state = env.world_state_transition(env.world_state, robot_action, None)
            data['last_state'] = env.world_state[1]
            data['current_state'] = env.world_state[0]
            if env.desc[env.world_state[0] // env.ncol, env.world_state[0] % env.ncol] in b'S':
                data['type'] = 'slippery'
            elif env.desc[env.world_state[0] // env.ncol, env.world_state[0] % env.ncol] in b'H':
                data['type'] = 'hole'
            elif env.world_state == env.ncol * env.ncol - 1:
                data['type'] = 'goal'
            else:
                data['type'] = 'ice'
            robot_action_node.position = env.world_state[0]

            all_states.append(env.world_state[0])

            curr_row = env.world_state[0] // env.ncol
            curr_col = env.world_state[0] % env.ncol
            last_row = env.world_state[1] // env.ncol
            last_col = env.world_state[1] % env.ncol

            # When robot takes control, undo human action and execute the robot's action instead
            if human_action[2] - 2 >= 0:
                undo_action = human_action[2] - 2
            else:
                undo_action = human_action[2] + 2

            # Move from last state with robot action instead
            actual_state = env.move(env.move(env.world_state[1], undo_action), robot_action[1])

            # Check for termination...
            # If current state is in hole or slippery grid, or
            # If the robot takes control and falls into hole/slippery grid, resetting the env, or
            # If the human moves to hole or slippery grid and the robot does not intervene
            if env.desc[curr_row, curr_col] in b'HS' or \
                    (env.desc[last_row, last_col] in b'HS' and env.world_state[0] == 0
                     and robot_action[0] in [2, 4] and actual_state != 0) or \
                    (env.desc[last_row, last_col] in b'HS' and robot_action[0] == 0):
                truncated = True
            env.render(round_num, human_action, robot_action, env.world_state, truncated=truncated)

            if round_num in CONDITION['pomcp_inverse']:
                final_env_reward -= env.reward(env.world_state, robot_action, human_action)
            else:
                final_env_reward += env.reward(env.world_state, robot_action, human_action)

            data['score'] = final_env_reward
            history.append(copy.deepcopy(data))
            data = {}

            # Terminates if goal is reached
            if env.isTerminal(env.world_state):
                # Show scores
                tmp.fill((255, 255, 255), rect=(0, 0, 256, 256))
                self.render_score(tmp, round_num, final_env_reward, step, detection_num)
                # Wait to open next game
                wait = True
                while wait:
                    for event in pygame.event.get():
                        if event.type == pygame.KEYUP:
                            if event.key == pygame.K_SPACE:
                                wait = False
                break

            # Terminates if reaching the maximum step
            if step >= self.num_steps:
                env.render(round_num, None, None, env.world_state, timeout=True)
                # Show scores
                tmp.fill((255, 255, 255), rect=(0, 0, 256, 256))
                self.render_score(tmp, round_num, final_env_reward, step, detection_num)
                wait = True
                while wait:
                    # Show scores
                    tmp.fill((255, 255, 255), rect=(0, 0, 256, 256))
                    self.render_score(tmp, round_num, final_env_reward, step, detection_num)
                    for event in pygame.event.get():

                        # Show scores
                        tmp.fill((255, 255, 255), rect=(0, 0, 256, 256))
                        self.render_score(tmp, round_num, final_env_reward, step, detection_num)
                        if event.type == pygame.KEYUP:
                            if event.key == pygame.K_SPACE:
                                wait = False

                break

            # We now use the real observation / human action (i.e., from the simulated human model)
            if robot_action[0] or truncated:
                pause = True
            else:
                pause = False
            action = None
            pygame.event.clear()

            while action is None:
                # Show scores
                tmp.fill((255, 255, 255), rect=(0, 0, 256, 256))
                self.render_score(tmp, round_num, final_env_reward, step, detection_num)

                for event in pygame.event.get():
                    if event.type == pygame.KEYUP:
                        if event.key == pygame.K_RETURN and pause:
                            pause = False
                            if truncated:
                                truncated = False
                            env.render(round_num, None, None, env.world_state, truncated=truncated)
                        if event.key == pygame.K_BACKSPACE and not truncated:
                            if detecting:
                                detecting = 0
                                pause = False
                                env.render(round_num, None, None, env.world_state, end_detecting=1)
                            elif detection_num < self.max_detection:
                                detecting = 1
                                pause = False
                                # Set detection = 1 in human action
                                env.render(round_num, (None, 1, None), None, env.world_state)
                            else:
                                detecting = 1
                                env.render(round_num, None, None, env.world_state, end_detecting=2)
                            # env.render(None, None, detecting, None, False)
                        if not detecting or (detecting and detection_num < self.max_detection):
                            if event.key == pygame.K_LEFT and not pause:
                                action = 0
                            if event.key == pygame.K_RIGHT and not pause:
                                action = 2
                            if event.key == pygame.K_UP and not pause:
                                action = 3
                            if event.key == pygame.K_DOWN and not pause:
                                action = 1
                        elif detecting and detection_num >= self.max_detection:
                            if event.key in [pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP, pygame.K_DOWN]:
                                pause = True
                                env.render(round_num, None, None, env.world_state, end_detecting=2)

            if detecting and detection_num < self.max_detection:
                detection_num += 1

            if (robot_action[0] == 1 or robot_action[0] == 3) and detecting != 1 and action == human_action[2]:
                is_accept = 2
            elif (robot_action[0] == 2 or robot_action[0] == 4) and detecting != 1 and \
                    abs(human_action[2] - robot_action[1]) == 2:
                is_accept = 2
            elif robot_action[0] == 0 or detecting == 1:
                is_accept = 0
            else:
                is_accept = 1
            human_action = tuple([is_accept, detecting, action])
            human_action_node = robot_action_node.human_node_children[human_action[1] * 4 + human_action[2]]

            data['human_action'] = human_action

            if human_action_node == "empty":
                # Here we are adding to the tree as this will become the root for the search in the next turn
                human_action_node = robot_action_node.human_node_children[
                    human_action[1] * 4 + human_action[2]] = HumanActionNode(env)
                # This is where we call invigorate belief... When we add a new human action node to the tree
                self.invigorate_belief(human_action_node, solver.root_action_node, robot_action, human_action, env)

            # Update the environment
            solver.root_action_node = human_action_node  # Update the root node from h to hao
            env.world_state = env.world_state_transition(env.world_state, robot_action, human_action)
            all_states.append(env.world_state[0])

            # Updates the world state in the belief to match the actual world state
            # Technically if all the belief updates are performed correctly, then there's no need for this.
            self.updateBeliefWorldState(human_action_node, env)

            # Updates robot's belief of the human latent state based on human action
            self.updateBeliefTrust(human_action_node, human_action)

            # For bookkeeping
            robot_actions.append(robot_action)
            human_actions.append(human_action)

        print("===================================================================================================")
        print("Round {} completed!".format(round_num))
        print("Time taken:")
        print("{} seconds".format(time.time() - start_time))
        print('Robot Actions: {}'.format(robot_actions))
        print('Human Actions: {}'.format(human_actions))
        return final_env_reward, history


def write_json(path, data, indent=4):
    '''
    Function to write json files
    '''

    class npEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.int32):
                return int(obj)
            return json.JSONEncoder.default(self, obj)

    with open(path, 'w') as file:
        json.dump(data, file, indent=indent, cls=npEncoder)


if __name__ == '__main__':
    pygame.init()
    pygame.display.init()
    pygame.display.set_caption("Frozen Lake")
    window_surface = pygame.display.set_mode((min(100 * 8, 1024) + 256 * 2, min(100 * 8, 800) + 300))
    pygame.font.init()
    font = pygame.font.Font(None, 30)
    x = font.render(
        "Participant ID: {}. ".format(username), True,
        (0, 0, 0))
    window_surface.fill((255, 255, 255))
    window_surface.blit(x, (10, 20))
    x = font.render(
        "Please finish the pre-experiment questionnaire and ask the experimenter".format(username), True,
        (0, 0, 0))
    window_surface.blit(x, (10, 50))
    x = font.render(
        "to start the study.".format(username), True,
        (0, 0, 0))
    window_surface.blit(x, (10, 80))
    x = font.render(
        "Instructions:", True,
        (0, 0, 0))
    window_surface.blit(x, (10, 140))
    x = font.render(
        "1. Use the arrow keys to control the agent.", True,
        (0, 0, 0))
    window_surface.blit(x, (20, 170))
    x = font.render(
        "2. Use Backspace to enter detection mode. Use arrow keys to detect adjacent grids. Press Backspace" +
        " again to exit detection mode.", True,
        (0, 0, 0))
    window_surface.blit(x, (20, 200))
    pygame.display.flip()
    print("Participant ID", username)
    input("Press Enter to continue...")

    # Choose heuristic agent type
    exp_type = 2  # 0: Without Explanation; 2: With explanation
    explanation = exp_type == 2

    # Set appropriate seeds
    SEED = 0
    random.seed(SEED)
    np.random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)

    # The following two parameters are for human behavior. They are currently not used.
    beta = 0.9  # Boltzmann rationality parameter (for human behavior)

    # factors for POMCP
    gamma = 0.99  # gamma for terminating rollout based on depth in MCTS
    c = 20  # 400  # exploration constant for UCT (taken as R_high - R_low)
    e = 0.1  # For epsilon-greedy policy

    epsilon = math.pow(gamma, 40)  # tolerance factor to terminate rollout
    num_iter = 100

    num_steps = 80

    # Executes num_rounds of experiments
    num_rounds = 10
    mean_rewards = []
    std_rewards = []
    all_rewards = []
    for n in range(4, num_rounds):
        start_t = time.time()
        initial_belief = []
        print("*********************************************************************")
        print("Executing Round {}......".format(n))
        print("*********************************************************************")

        # Robot's belief of human parameters
        all_initial_belief_trust = []
        for _ in range(1000):
            all_initial_belief_trust.append((1, 1))  # Uniform Prior

        # Setup Driver
        map_num = mapOrder[n]
        map = MAPS["MAP" + str(map_num)]
        foggy = FOG["MAP" + str(map_num)]
        human_err = HUMAN_ERR["MAP" + str(map_num)]
        robot_err = ROBOT_ERR["MAP" + str(map_num)]

        if n in CONDITION['pomcp_inverse']:
            env = InverseFrozenLakeEnv(desc=map, foggy=foggy, human_err=human_err, robot_err=robot_err,
                                       render_mode="human",
                                       seed=SEED,
                                       human_type="epsilon_greedy", round=n)
        else:
            env = FrozenLakeEnvInterface(desc=map, foggy=foggy, human_err=human_err, robot_err=robot_err,
                                         render_mode="human",
                                         seed=SEED,
                                         human_type="epsilon_greedy", round=n)

        # Reset the environment for each round of the study
        env.reset(round_num=n)
        init_world_state = env.world_state

        for i in range(len(all_initial_belief_trust)):
            initial_belief.append(init_world_state + [list(all_initial_belief_trust[i])])

        root_node = RootNode(env, initial_belief)  # Initialize root node
        solver = POMCPSolver(epsilon, env, root_node, num_iter, c)  # Initialize Solver

        if n in CONDITION['interrupt']:
            agent = HeuristicAgent(intervention_type=exp_type + 1)
            driver = Driver(env, solver, num_steps, agent=agent)
        elif n in CONDITION['take_control']:
            agent = HeuristicAgent(intervention_type=exp_type + 2)
            driver = Driver(env, solver, num_steps, agent=agent)
        else:
            driver = Driver(env, solver, num_steps)
            explanation = None

        env_reward, history = driver.execute(n)  # Start Episode and search for robot actions

        userdata[str(n)] = {'history': history,
                            'duration': time.time() - start_t,
                            'explanation': explanation,
                            'optimal_path': len(env.find_shortest_path(env.desc, env.hole + env.slippery, 0, env.ncol))}

        print("===================================================================================================")
        print("Final reward:{}".format(env_reward + num_steps))
        print("===================================================================================================")
