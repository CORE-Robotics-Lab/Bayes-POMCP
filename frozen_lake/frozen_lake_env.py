"""
Creates a simulator for mixed-initiative human-robot interaction in Frozen Lake.
Adapted from: https://github.com/openai/gym/blob/master/gym/envs/toy_text/frozen_lake.py
The robot can have five different modalities - no interruption, interruption, taking control,
interruption with explanation, taking control with explanation.
We are modeling the environment from the robot's perspective.
(Augmented) State space:
  - World state (observable): slippery, optional direction, ice hole, goal, fog X 8 grids (2 X 5 X 8)
  - Hidden states (non-observable): human trust
Action space:
    - Robot Action: No-assist, Interruption, Taking control,
                    Interruption with explanation, Taking control with explanation

Observation space:
    - Human action: Move X 4 directions, detect X 4 directions

Reward:
    Reward schedule:
    - Total Reward = Max_steps - steps_taken -10 * # fall_into_hole - 2 * # detections + 30 * goal_state
    - Step Reward:
            -1 for step
            -2 for detect
            -10 if state == hole
            +30 if state == goal
"""

from typing import List, Optional
import random
import gym
import numpy as np
import copy
import os
from gym import spaces, utils
from frozen_lake.frozen_lake_map import MAPS, FOG, HUMAN_ERR, ROBOT_ERR

# Define directions for moving in the Frozen Lake grid
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

# Experiment Condition / Robot action types
CONDITION = {
    'no_interrupt': 0,
    'interrupt': 1,
    'control': 2,
    'interrupt_w_explain': 3,
    'control_w_explain': 4
}


class FrozenLakeEnv:
    """
    Frozen lake involves crossing a frozen lake from start to goal without falling into any holes
    by walking over the frozen lake.
    ## Description
    The game starts with the player at location [0,0] of the frozen lake grid world with the
    goal located at far extent of the world e.g. [3,3] for the 4x4 environment.
    Holes in the ice are distributed in set locations when using a pre-determined map.
    The player makes moves until they reach the goal or reach the step number limitation.
    The lake is slippery so the player slips into a hole rather than moving
    to the intended direction sometimes.
    Maps will always have a path to the goal.
    Elf and stool from [https://franuka.itch.io/rpg-snow-tileset](https://franuka.itch.io/rpg-snow-tileset).
    All other assets by Mel Tillery [http://www.cyaneus.com/](http://www.cyaneus.com/).
    ## Action Space
    The action shape is `(1,)` in the range `{0, 3}` indicating
    which direction to move the player.
    - 0: Move left
    - 1: Move down
    - 2: Move right
    - 3: Move up
    ## Observation Space
    The observation is a value representing the player's current position as
    current_row * nrows + current_col (where both the row and col start at 0).
    For example, the goal position in the 4x4 map can be calculated as follows: 3 * 4 + 3 = 15.
    The number of possible observations is dependent on the size of the map.
    The observation is returned as an `int()`.
    ## Starting State
    The episode starts with the player in state `[0]` (location [0, 0]).
    ## Rewards
    Reward schedule:
    - Total Reward = Max_steps - steps_taken -10 * # fall_into_hole - 2 * # detections + 30 * goal_state
    - Step Reward:
            -1 for step
            -2 for detect
            -10 if state == hole
            +30 if state == goal
    ## Episode End
    The episode ends if the following happens:
    - Termination:
        1. The player reaches the step number limitation.
        2. The player reaches the goal at `max(nrow) * max(ncol) - 1` (location `[max(nrow)-1, max(ncol)-1]`).
    """

    metadata = {
        "render_modes": ["human", "ansi", "rgb_array"],
        "render_fps": 4,
    }

    def __init__(
            self,
            render_mode: Optional[str] = None,
            desc=None,
            foggy=None,
            human_err=None,
            robot_err=None,
            map_name="4x4",
            round=0,
            seed=None,
            human_type="random",
            update_belief=True
    ):
        """
        :param desc: The map showing the hole and slippery positions
        :param foggy: The map showing the foggy and non-foggy regions
        :param human_err: A list of the position pairs (row, column) that the human player will make errors
        :param robot_err: A list of the position pairs (row, column) that the robot agent will make errors
        :param map_name: The number of current map
        :param round: The current round number
        :param seed: The random seed
        :param human_type: The simulated human model type
        :param update_belief: Whether to update the beta params in augmented transition
        """
        # Set Appropriate Random seeds
        self.s = 0
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

        self.update_belief = update_belief  # Whether to update the beta params in augmented transition

        self.is_error = False
        assert (desc is not None and map_name is not None)

        if desc is None:
            desc = MAPS[map_name]
        self.desc = desc = np.asarray(desc, dtype="c")
        self.fog = np.asarray(foggy, dtype="c")  # Foggy and non-foggy grids
        self.nrow, self.ncol = nrow, ncol = desc.shape  # Num rows and cols in the frozen lake grid
        self.human_err = human_err
        self.robot_err = robot_err
        self.condition = round

        # The agent's initial position is always fixed to the start (top-left) grid on every map.
        self.initial_state_distrib = np.array(desc == b"B").astype("float64").ravel()
        self.initial_state_distrib /= self.initial_state_distrib.sum()

        self.robot_action = None
        self.last_interrupt = [None, None]

        # Position of holes (currently fully accessible to human and robot)
        rows, cols = np.where(self.desc == b'H')
        self.hole = [(r, c) for r, c in zip(rows, cols)]
        # Slippery regions are only partially observable to both the human and the robot 
        rows, cols = np.where(self.desc == b'S')
        self.slippery = [(r, c) for r, c in zip(rows, cols)]

        self.interrupted = 0
        self.truncated = False
        self.is_error = False
        self.num_error = 0
        self.num_interrupt = 0
        self.interrupt_state = []

        self.running = True

        self.world_state = []

        # Robot action space is action_type and the direction to move
        self.robot_action_space = spaces.MultiDiscrete([5, 4], seed=seed)
        # Human's action space is whether they accepted the robot's suggestion and the direction that they choose
        # (no-assist/accept/reject, detect/no-detect, LEFT/DOWN/RIGHT/UP)
        self.human_action_space = spaces.MultiDiscrete([3, 2, 4], seed=seed)
        # Robot's observation space => the human's last action
        self.robot_observation_space = spaces.MultiDiscrete([2, 4], seed=seed)

        self.seed = seed
        self.human_type = human_type

        self.render_mode = render_mode

        # pygame utils
        self.window_size = (min(100 * ncol, 1024) + 256 * 2, min(100 * nrow, 800) + 300)
        self.cell_size = (
            min(100 * ncol, 1024) // self.ncol,
            min(100 * nrow, 1024) // self.nrow,
        )
        self.window_surface = None
        self.clock = None
        self.hole_img = None
        self.cracked_hole_img = None
        self.ice_img = None
        self.elf_images = None
        self.goal_img = None
        self.start_img = None
        self.fog_img = None
        self.smoke_img = None
        self.slippery_img = None

    def to_s(self, row, col):
        # Convert from (row, col) -> grid number (0 to MxN-1)
        return row * self.ncol + col

    def inc(self, row, col, a):
        # Move based on (row, col) + direction
        if a == LEFT:
            col = max(col - 1, 0)
        elif a == DOWN:
            row = min(row + 1, self.nrow - 1)
        elif a == RIGHT:
            col = min(col + 1, self.ncol - 1)
        elif a == UP:
            row = max(row - 1, 0)
        return row, col

    def move(self, s, a):
        # Transition to the next state s' from s after action a
        row = s // self.ncol
        col = s % self.ncol
        next_row, next_col = self.inc(row, col, a)
        return self.to_s(next_row, next_col)

    def find_shortest_path(self, board, slippery_region, start, max_size):
        """
        Use BFS to find the shortest path to the goal based on current knowledge of the agent's slippery regions
        :param board: Frozen Lake Grid
        :param slippery_region: Known slippery regions to the agent
        :param start: Starting location of the agent
        :param max_size: Max number of grids = MxN
        """
        path_list = [[[(start // max_size, start % max_size), None]]]
        path_index = 0
        # To keep track of previously visited nodes
        previous_nodes = {(start // max_size, start % max_size)}
        if start == max_size * max_size - 1:
            return path_list[0]

        while path_index < len(path_list):
            current_path = path_list[path_index]
            last_node, _ = current_path[-1]
            r, c = last_node
            directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
            for i in range(4):
                x, y = directions[i]
                r_new = r + x
                c_new = c + y
                if r_new < 0 or r_new >= max_size or c_new < 0 or c_new >= max_size:
                    continue
                if board[r_new][c_new] == b"G":
                    next_node = (r_new, c_new)
                    current_path.append([next_node, i])
                    return current_path
                if (r_new, c_new) not in slippery_region:
                    next_node = (r_new, c_new)
                    # Add new paths
                    if not next_node in previous_nodes:
                        new_path = current_path[:]
                        new_path.append([next_node, i])
                        path_list.append(new_path)
                        # To avoid backtracking
                        previous_nodes.add(next_node)
            # Continue to next path in list
            path_index += 1
        # No path found
        return []

    def detect_slippery_region(self, position, human_slippery, robot_slippery, human_err, robot_err):
        """
        Update knowledge of both human and robot in case the human agent uses the detection sensor
        :param position: Current position (grid ID ranges between 0 to MxN)
        :param human_slippery: List of slippery grids known to the human
        :param robot_slippery: List of slippery grids known to the robot
        :param human_err: List of errors made by the human in identifying slippery regions
        :param robot_err: List of errors made by the robot in identifying slippery regions
        """
        curr_row = position // self.ncol
        curr_col = position % self.ncol
        actions = [0, 1, 2, 3]
        next_human_slippery = {i for i in human_slippery}  # Convert to set
        next_robot_slippery = {i for i in robot_slippery}  # Convert to set
        for a in actions:
            row, col = self.inc(curr_row, curr_col, a)
            # Add robot slippery regions
            if (self.desc[row, col] in b"S" and ((row, col) not in self.robot_err or (row, col) in robot_err)) or \
               (self.desc[row, col] in b"F" and ((row, col) in self.robot_err and (row, col) not in robot_err)):
                next_robot_slippery.add((row, col))

            # Add human slippery regions
            if (self.desc[row, col] in b"S" and ((row, col) not in self.human_err or (row, col) in human_err)) or \
               (self.desc[row, col] in b"F" and ((row, col) in self.human_err and (row, col) not in human_err)):
                next_human_slippery.add((row, col))

        return next_human_slippery, next_robot_slippery

    def get_last_action(self, curr_position, last_position):
        # Based on the current and previous grid positions, calculate the direction that the agent moved
        curr_row = curr_position // self.ncol
        curr_col = curr_position % self.ncol
        last_row = last_position // self.ncol
        last_col = last_position % self.ncol
        if self.desc[last_row, last_col] in b'HS' and curr_position == 0:
            return 0  # The game was restarted
        if curr_row == last_row:
            if curr_col == last_col + 1:
                return 2  # Right
            else:
                return 0  # Left
        else:
            if curr_row == last_row + 1:
                return 1  # Down
            else:
                return 3  # Up

    def augmented_state_transition(self, current_augmented_state, robot_action, human_action):
        """
        Based on the current state, robot action and human action, transit to the next (augmented) state
        :param current_augmented_state: The current augmented world state
        :param robot_action: The next action that the robot agent chooses
        :param human_action: The next action that the human player chooses
        :return: the next augmented world state after executing the robot action and human action
        """
        # observed states
        current_world_state = current_augmented_state[:-1]

        # Latent states
        human_trust = current_augmented_state[-1]

        # World state (observable) (human's)
        next_world_state = self.world_state_transition(current_world_state, robot_action, human_action)

        if not self.update_belief:
            next_human_trust = human_trust

        else:
            if not human_action:
                next_human_trust = human_trust  # TODO: Add noise for unseen robot actions
            elif human_action[0] == 0:
                # No assist condition: do not update trust
                next_human_trust = human_trust
            else:
                human_accept = human_action[0]  # 1 indicates accept, 2 indicates reject
                human_trust[human_accept - 1] += 1  # index 0 is acceptance count, index 1 is rejection count
                next_human_trust = human_trust

        next_augmented_state = next_world_state[:] + [next_human_trust]
        return next_augmented_state

    def world_state_transition(self, current_world_state, robot_action, human_action):
        # world_state[1] is the last human position,
        # it will be updated if the human takes an action but not if it's robot's turn.
        position, last_human_position, human_slippery, robot_slippery, human_err, robot_err = current_world_state

        if human_action:
            # If human's turn, set robot_action as None for the step function
            next_position, next_human_slippery, next_robot_slippery, next_human_err, next_robot_err = self.step(
                current_world_state, None, human_action)
            next_world_state = [next_position, position, next_human_slippery, next_robot_slippery, next_human_err,
                                next_robot_err]
        else:
            # If robot's turn, set human_action as None for the step function
            next_position, next_human_slippery, next_robot_slippery, next_human_err, next_robot_err = self.step(
                current_world_state, robot_action, None)

            next_world_state = [next_position, position, next_human_slippery, next_robot_slippery, next_human_err,
                                next_robot_err]
        return next_world_state

    def step(self, current_world_state, robot_action, human_action):
        # Update the position of the agent based on current world state and human/robot action
        position, last_position, human_slippery, robot_slippery, human_err, robot_err = current_world_state

        if human_action is not None:
            # Human's turn
            human_accept = human_action[0]
            human_detect = human_action[1]
            human_direction = human_action[2]

            if human_detect:
                # Human used the detection function
                self.s = s = position
                detected_s = self.move(position, human_direction)
                next_human_slippery = {i for i in human_slippery}
                next_robot_slippery = {i for i in robot_slippery}
                next_human_err = {i for i in human_err}
                next_robot_err = {i for i in robot_err}
                row = detected_s // self.ncol
                col = detected_s % self.ncol
                if self.desc[row, col] in b'S':
                    if (row, col) not in next_human_slippery:
                        next_human_slippery.add((row, col))
                        next_human_err.add((row, col))
                    if (row, col) not in next_robot_slippery:
                        next_robot_slippery.add((row, col))
                        next_robot_err.add((row, col))

                elif self.desc[row, col] in b'F':
                    if (row, col) in next_human_slippery:
                        next_human_slippery.remove((row, col))
                        next_human_err.add((row, col))
                    if (row, col) in next_robot_slippery:
                        next_robot_slippery.remove((row, col))
                        next_robot_err.add((row, col))
                return s, next_human_slippery, next_robot_slippery, next_human_err, next_robot_err

            else:
                # No detection -> Move
                s = self.move(position, human_direction)
                self.s = s
                # Only update the world state after robot action
                return s, human_slippery, robot_slippery, human_err, robot_err

        else:
            # Robot's turn
            robot_type = robot_action[0]
            robot_direction = robot_action[1]
            if robot_type == CONDITION['interrupt'] or robot_type == CONDITION['interrupt_w_explain']:
                self.s = s = last_position  # don't move the agent position in case of interruption
                return s, human_slippery, robot_slippery, human_err, robot_err

            elif robot_type == CONDITION['control'] or robot_type == CONDITION['control_w_explain']:
                s = self.move(last_position, robot_direction)
                self.s = s
                next_human_slippery = {i for i in human_slippery}
                next_robot_slippery = {i for i in robot_slippery}
                next_human_err = {i for i in human_err}
                next_robot_err = {i for i in robot_err}
                if self.desc[s // self.ncol, s % self.ncol] in b'HS':  # if human falls into a hole, restart
                    self.s = 0
                    self.interrupted = 0
                    self.truncated = True
                    self.last_interrupt = [None, None]
                    # Remove the error and show the ground truth
                    if (s // self.ncol, s % self.ncol) not in human_slippery:
                        next_human_slippery.add((s // self.ncol, s % self.ncol))
                        if (s // self.ncol, s % self.ncol) in self.human_err:
                            next_human_err.add((s // self.ncol, s % self.ncol))
                    if (s // self.ncol, s % self.ncol) not in robot_slippery:
                        next_robot_slippery.add((s // self.ncol, s % self.ncol))
                        if (s // self.ncol, s % self.ncol) in self.robot_err:
                            next_robot_err.add((s // self.ncol, s % self.ncol))
                    return 0, next_human_slippery, next_robot_slippery, next_human_err, next_robot_err
                next_human_slippery, next_robot_slippery = self.detect_slippery_region(s, human_slippery,
                                                                                       robot_slippery, human_err,
                                                                                       robot_err)
                # Remove the false slippery regions after passing it
                if self.desc[s // self.ncol, s % self.ncol] in b'F' and (
                        s // self.ncol, s % self.ncol) in human_slippery:
                    next_human_slippery.remove((s // self.ncol, s % self.ncol))
                    if (s // self.ncol, s % self.ncol) in self.human_err:
                        next_human_err.add((s // self.ncol, s % self.ncol))
                if self.desc[s // self.ncol, s % self.ncol] in b'F' and (
                        s // self.ncol, s % self.ncol) in robot_slippery:
                    next_robot_slippery.remove((s // self.ncol, s % self.ncol))
                    if (s // self.ncol, s % self.ncol) in self.robot_err:
                        next_robot_err.add((s // self.ncol, s % self.ncol))
                return s, next_human_slippery, next_robot_slippery, next_human_err, next_robot_err

            else:
                # Robot did not intervene --> proceed with updating the world state based on human action
                curr_row = position // self.ncol
                curr_col = position % self.ncol
                next_human_slippery = {i for i in human_slippery}
                next_robot_slippery = {i for i in robot_slippery}
                next_human_err = {i for i in human_err}
                next_robot_err = {i for i in robot_err}
                if robot_type == CONDITION['no_interrupt'] and self.desc[curr_row, curr_col] in b'HS':
                    # if the robot did not intervene when the human proceeded to a hole/slippery region => restart
                    self.s = 0
                    self.interrupted = 0
                    self.truncated = True
                    self.last_interrupt = [None, None]
                    if (curr_row, curr_col) not in human_slippery:
                        next_human_slippery.add((curr_row, curr_col))
                        if (curr_row, curr_col) in self.human_err:
                            next_human_err.add((curr_row, curr_col))
                    if (curr_row, curr_col) not in robot_slippery:
                        next_robot_slippery.add((curr_row, curr_col))
                        if (curr_row, curr_col) in self.robot_err:
                            next_robot_err.add((curr_row, curr_col))
                    return 0, next_human_slippery, next_robot_slippery, next_human_err, next_robot_err

                self.s = position
                next_human_slippery, next_robot_slippery = self.detect_slippery_region(position, human_slippery,
                                                                                       robot_slippery, human_err,
                                                                                       robot_err)
                # Remove the false positive slippery regions after passing it
                if self.desc[curr_row, curr_col] in b'F' and (curr_row, curr_col) in human_slippery:
                    next_human_slippery.remove((curr_row, curr_col))
                    if (curr_row, curr_col) in self.human_err:
                        next_human_err.add((curr_row, curr_col))
                if self.desc[curr_row, curr_col] in b'F' and (curr_row, curr_col) in robot_slippery:
                    next_robot_slippery.remove((curr_row, curr_col))
                    if (curr_row, curr_col) in self.robot_err:
                        next_robot_err.add((curr_row, curr_col))
                return position, next_human_slippery, next_robot_slippery, next_human_err, next_robot_err

    def get_rollout_observation(self, current_augmented_state, robot_action):
        """
        Used to simulate human actions during POMCP rollout of the environment
        Uses current belief of the human and an epsilon-greedy heuristic to determine their next actions in case of
        non compliance
        :param current_augmented_state: (type: List): Current world state + belief about human compliance
        :param robot_action: (type: Tuple): Indicating the robot's intervention type, and direction of chosen action
        :return human_action: (type: Tuple): whether they complied, detected and the direction they chose to move
        """
        current_world_state = current_augmented_state[:6]
        current_human_trust = current_augmented_state[6]
        current_position = current_world_state[0]

        # Get human action from heuristic_interrupt model (Needs access to game state info)
        robot_assist_type = robot_action[0]
        robot_direction = robot_action[1]
        human_slippery = current_augmented_state[2]
        robot_slippery = current_augmented_state[3]

        # When the robot is assisting -> human choice depends on trust
        human_acceptance_probability = np.random.beta(current_human_trust[0], current_human_trust[1])

        # For actions with explanation, increase the human acceptance probability
        if robot_assist_type in [3, 4]:
            human_acceptance_probability = np.minimum(human_acceptance_probability + 0.1, 1.0)

        # If acceptance <= prob < acceptance + 0.5(1-acceptance) then reject + no detection,
        # if prob >= acceptance + 0.5(1-acceptance) then reject + detection
        actions = [0, 1, 2, 3]
        prob = np.random.uniform()
        detect = 0
        detect_new_grid_prob = 0.2
        if self.human_type == "random":
            human_acceptance_probability = np.random.uniform()  # Make this random
            if robot_assist_type == 0:
                # No assistance
                accept = 0
                if human_acceptance_probability <= prob < 0.5 + 0.5*human_acceptance_probability:
                    detect = 1

            elif robot_assist_type == 1 or robot_assist_type == 3:  # Interrupt
                # User either chooses the robot's suggestion or
                # their own based on their trust in the robot and their capability
                if robot_direction - 2 >= 0:
                    undo_action = robot_direction - 2
                else:
                    undo_action = robot_direction + 2
                if prob < human_acceptance_probability:
                    actions.remove(undo_action)
                    accept = 1  # Accept
                elif human_acceptance_probability <= prob < 0.5 + 0.5 * human_acceptance_probability:
                    # Currently will choose the last position following epsilon-greedy strategy
                    if np.random.uniform() < detect_new_grid_prob:
                        actions.remove(undo_action)
                    else:
                        actions = [undo_action]
                    detect = 1
                    accept = 2
                else:
                    actions = [undo_action]
                    accept = 2  # Reject

            else:  # taking control
                if prob < human_acceptance_probability:
                    accept = 1
                    if robot_direction is not None:
                        if robot_direction - 2 >= 0:
                            actions.remove(robot_direction - 2)
                        else:
                            actions.remove(robot_direction + 2)
                # Detection when robot took over control: check one surrounding grid
                elif human_acceptance_probability <= prob < 0.5 + 0.5 * human_acceptance_probability:
                    detect = 1
                    accept = 2
                    if robot_direction is not None:
                        if robot_direction - 2 >= 0:
                            actions.remove(robot_direction - 2)
                        else:
                            actions.remove(robot_direction + 2)
                # Return to the last state after refusing
                else:
                    accept = 2
                    if robot_direction is not None:
                        if robot_direction - 2 >= 0:
                            actions = [robot_direction - 2]
                        else:
                            actions = [robot_direction + 2]

            human_choice = np.random.choice(actions)
            s = self.move(current_position, human_choice)
            while s == current_position and len(actions) > 1:
                actions.remove(human_choice)
                human_choice = np.random.choice(actions)
                s = self.move(current_position, human_choice)

        elif self.human_type == "epsilon_greedy":
            epsilon = 0.2
            if robot_assist_type == 0:
                accept = 0
                if human_acceptance_probability <= prob < 0.5 + 0.5*human_acceptance_probability:
                    detect = 1
            elif robot_assist_type == 1 or robot_assist_type == 3:  # Interrupt
                # User either chooses the robot's suggestion or their own
                # based on their trust in the robot and their capability
                if robot_direction - 2 >= 0:
                    undo_action = robot_direction - 2
                else:
                    undo_action = robot_direction + 2
                if prob < human_acceptance_probability:
                    actions.remove(undo_action)
                    accept = 1  # Accept
                elif human_acceptance_probability <= prob < 0.5 + 0.5*human_acceptance_probability:
                    # Currently will choose the last position following epsilon-greedy strategy
                    if np.random.uniform() < detect_new_grid_prob:
                        actions.remove(undo_action)
                    else:
                        actions = [undo_action]
                    detect = 1
                    accept = 2
                else:
                    actions = [undo_action]
                    accept = 2  # Reject
            else:  # taking control
                if prob < human_acceptance_probability:
                    accept = 1
                    if robot_direction is not None:
                        if robot_direction - 2 >= 0:
                            actions.remove(robot_direction - 2)
                        else:
                            actions.remove(robot_direction + 2)
                # Detection when robot took over control: check one surrounding grid
                elif human_acceptance_probability <= prob < 0.5 + 0.5 * human_acceptance_probability:
                    detect = 1
                    accept = 2
                    if robot_direction is not None:
                        if robot_direction - 2 >= 0:
                            actions.remove(robot_direction - 2)
                        else:
                            actions.remove(robot_direction + 2)
                    # Return to the last state after refusing
                else:
                    accept = 2
                    if robot_direction is not None:
                        if robot_direction - 2 >= 0:
                            actions = [robot_direction - 2]
                        else:
                            actions = [robot_direction + 2]

            shortest_path = self.find_shortest_path(self.desc, human_slippery, current_position, self.ncol)

            e = np.random.uniform()
            true_shortest_path = self.find_shortest_path(self.desc, self.hole + self.slippery, current_position, self.ncol)
            if len(true_shortest_path) > 1:
                true_best_action = true_shortest_path[1][1]
            else:
                true_best_action = np.random.choice([0, 1, 2, 3])
            if e < epsilon: # Choose best action using the human map
                if len(shortest_path) > 1:
                    best_action = shortest_path[1][1]
                else:
                    best_action = np.random.choice([0, 1, 2, 3])
                if best_action in actions:
                    human_choice = best_action
                else:
                    # Cannot choose the best action because of acceptance,
                    # so randomly choose another suboptimal action
                    if true_best_action in actions and len(actions) > 1:
                        actions.remove(true_best_action)
                    human_choice = np.random.choice(actions)
                    s = self.move(current_position, human_choice)
                    while s == current_position and len(actions) > 1:
                        actions.remove(human_choice)
                        human_choice = np.random.choice(actions)
                        s = self.move(current_position, human_choice)
            else:
                # Choose optimal action
                human_choice = true_best_action

        return accept, detect, human_choice

    def reward(self, augmented_state, robot_action, human_action=None):
        """
        Calculate the reward in the current step
        :param augmented_state: The current augmented world state after executing the robot and human actions
        :param robot_action: The last action that the robot agent chose
        :param human_action: The last action that the human player chose
        :return: The reward of this step after executing the robot and human actions
        """
        position, last_position, human_slippery, robot_slippery = augmented_state[:4]
        # Get reward based on the optimality of the human action and the turn number
        curr_row = position // self.ncol
        curr_col = position % self.ncol
        last_row = last_position // self.ncol
        last_col = last_position % self.ncol
        reward = -1
        detect = None
        if human_action:
            human_accept, detect, human_choice = human_action
        if detect == 1:
            reward = -2  # Penalty for using the detection function
        # Penalty for falling into a hole (Evaluated after robot action)
        if self.desc[curr_row, curr_col] in b'HS' or \
                (self.desc[last_row, last_col] in b'HS' and position == 0 and self.move(last_position,
                                                                                        robot_action[1]) != 0) or \
                (self.desc[last_row, last_col] in b'HS' and robot_action[0] == 0):
            reward = -10
        elif self.desc[curr_row, curr_col] in b'G':
            reward = 30  # Bonus for reaching the goal
        return reward

    def get_human_action(self, robot_action=None):
        raise NotImplementedError

    def get_robot_action(self, world_state, robot_assistance_mode=0):
        # Robot's recommended action with or without explanations
        # For the purpose of data collection, the robot will follow static_take_control policies
        position = world_state[0]
        last_position = world_state[1]
        robot_slippery = world_state[3]

        curr_row = position // self.ncol
        curr_col = position % self.ncol

        # Interrupt
        if robot_assistance_mode == CONDITION['interrupt'] or robot_assistance_mode == CONDITION['interrupt_w_explain']:
            # wait in the same state
            self.interrupted = 1
            if self.desc[curr_row, curr_col] == b"H":
                self.interrupted = 2

            # Useless if we store the previous position
            last_human_action = self.get_last_action(position, last_position)
            if last_human_action - 2 >= 0:
                undo_action = last_human_action - 2
            else:
                undo_action = last_human_action + 2
            self.robot_action = undo_action
            return robot_assistance_mode, undo_action

        # Take over control
        elif robot_assistance_mode == CONDITION['control'] or robot_assistance_mode == CONDITION['control_w_explain']:
            last_human_action = self.get_last_action(position, last_position)
            s_previous = last_position
            shortest_path = self.find_shortest_path(self.desc, robot_slippery, s_previous, self.ncol)
            if len(shortest_path) < 2:
                actions = [RIGHT, DOWN, LEFT, UP]
                actions.remove(last_human_action)
                robot_action = actions.pop()
                next_s = self.move(s_previous, robot_action)

                while len(actions) > 0 and (
                       self.desc[next_s // self.ncol, next_s % self.ncol] in b'HS' or next_s == s_previous):

                    robot_action = actions.pop()
                    next_s = self.move(s_previous, robot_action)
            else:
                best_action = shortest_path[1][1]
                robot_action = best_action
                next_s = self.move(s_previous, best_action)

                # Choose another action if the best action will lead to failure
                actions = [RIGHT, DOWN, LEFT, UP]
                actions.remove(best_action)
                if last_human_action in actions:
                    actions.remove(last_human_action)
                while len(actions) > 0 and (
                        self.desc[next_s // self.ncol, next_s % self.ncol] in b'HS' or next_s == s_previous):

                    robot_action = actions.pop()
                    next_s = self.move(s_previous, robot_action)
                    if len(actions) == 0 and (
                            self.desc[next_s // self.ncol, next_s % self.ncol] in b'HS' or next_s == s_previous):
                        return 0, None

                if robot_action == last_human_action:
                    return 0, None

            self.interrupted = 1
            if self.desc[curr_row, curr_col] == b"H":
                self.interrupted = 2
            self.last_interrupt = [position, last_human_action]
            return robot_assistance_mode, robot_action

        # No interruption
        else:
            self.robot_action = None
            return robot_assistance_mode, None

    def get_action_space(self, agent):
        if agent == "robot":
            return self.robot_action_space
        else:
            return self.human_action_space

    def get_observation_space(self, agent):
        if agent == "robot":
            return self.robot_observation_space
        else:
            raise NotImplementedError

    def reset(self):
        """ Reset the environment, including the user and robot's knowledge of slippery regions
            :return world_state: (type: List) [position, last_position, ...]
        """
        self.s = 0  # Reset the agent location to the start grid
        next_human_slippery, next_robot_slippery = self.detect_slippery_region(0, {i for i in self.hole},
                                                                               {i for i in self.hole}, (), ())
        self.world_state = [0, 0, next_human_slippery, next_robot_slippery, set(), set()]

        return [self.s, 0, next_human_slippery, next_robot_slippery, set(), set()]

    def isTerminal(self, world_state):
        """
        Checks if the current world_state is a terminal state (i.e., either user found the code, or ran out of max turns)
        :param world_state: [position, last_position, ...]
        :return: returns true if world_state is terminal
        """
        position, last_position, human_slippery, robot_slippery, human_err, robot_err = world_state
        curr_col = position // self.ncol
        curr_row = position % self.ncol
        if self.desc[curr_col, curr_row] in b'G':
            # Reached the Goal successfully -> Terminate the episode
            self.running = False
            return True
        return False

    def render(self, map):
        """
        Render the map as a list of letters (text-based visualization)
        :param map: (type: String) Layout of the FrozenLake Grid
        """
        desc = map.tolist()
        row, col = self.s // self.ncol, self.s % self.ncol
        desc = [[c.decode("utf-8") for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        print("\n".join("".join(line) for line in desc))