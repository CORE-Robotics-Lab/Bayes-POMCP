import numpy as np
import os
import random
import copy
import math
from pomcp_solvers.root_node import *
from pomcp_solvers.robot_action_node import *
from pomcp_solvers.human_action_node import *


class POMCPSolver:
    def __init__(self, epsilon, env, root_action_node, num_iter, c, gamma=0.99):
        """
        Initializes instance of the POMCP solver for learning a robot policy

        :param epsilon: (type: float) tolerance factor to terminate rollout
        :param env: (type: Environment) Instance of the Mastermind environment
        :param root_action_node: (type: RootNode or Human Action Node)
        :param num_iter: (type: int) the number of trajectories or simulations in each search
        :param c: (type: float) Exploration constant for the UCT algorithm
        """
        self.epsilon = epsilon
        self.env = env
        self.root_action_node = root_action_node
        self.num_iter = num_iter
        self.c = c
        self.gamma = gamma
        random.seed(env.seed)
        np.random.seed(env.seed)
        os.environ['PYTHONHASHSEED'] = str(env.seed)

    def search(self):
        """
        Starting point for the POMCP framework. Samples / simulates num_iter trajectories and carries out the search
        :return: (type: np array) optimal robot action (based on the tree so far)
        """
        for _ in range(self.num_iter):
            sample_augmented_state = copy.deepcopy(self.root_action_node.sample_state())
            self.simulate(sample_augmented_state, self.root_action_node, 0)

        return self.root_action_node.optimal_robot_action(c=0)  # No exploration for now...

    def rollout(self, augmented_state, robot_action, action_node, depth):
        """
        Calls the rollout helper function (recursive rollout till certain depth) and adds new robot and human nodes
        created by the current rollout to the tree

        :param augmented_state: (type: list) the augmented state (world state + latent human states) before robot and human action
        :param robot_action: (type: int)  the starting robot action
        :param action_node: the action node / history (denoted as h) from where the rollout starts
        :param depth: (type: int) the current depth in the tree
        :return: (type: float) returns rollout value
        """
        human_action = self.env.get_rollout_observation(augmented_state, robot_action)
        value = self.rollout_helper(
            copy.deepcopy(augmented_state), robot_action, human_action, depth)
        next_augmented_state = self.env.augmented_state_transition(copy.deepcopy(augmented_state), robot_action, human_action)

        # Create new robot node and human action nodes
        new_robot_action_node = RobotActionNode(self.env)
        new_robot_action_node.update_visited()
        new_robot_action_node.update_value(value)

        new_human_action_node = HumanActionNode(self.env)
        new_human_action_node.update_belief(next_augmented_state)
        new_human_action_node.update_visited()
        new_human_action_node.update_value(value)

        # Add the newly created nodes to the tree
        new_robot_action_node.human_node_children[human_action[1]*4 + human_action[2]] = new_human_action_node
        new_robot_action_node.position = augmented_state[0]
        action_node.robot_node_children[robot_action[0]] = new_robot_action_node

        return value

    def rollout_helper(self, augmented_state, robot_action, human_action, depth):
        """
        Carries out the recursive rollout process

        :param augmented_state: (type: list) the augmented state (world state + latent human states) before robot and human action
        :param robot_action: (type: np array)  the starting robot action
        :param human_action: (type: int) the current human action
        :param depth: (type: int) the current depth in the tree
        :return: (type: float) returns rollout value
        """

        # Returns 0 if max depth has been reached
        if math.pow(self.gamma, depth) < self.epsilon:
            return 0

        # Returns the env reward upon reaching the terminal state
        world_state = augmented_state[:6]

        second_augmented_state = self.env.augmented_state_transition(augmented_state, None, human_action)

        # next robot action --> sample from a uniform distribution
        if human_action[1] == 1:
            next_robot_action_type = 0
        else:
            next_robot_action_type = self.env.robot_action_space.sample()[0]

        next_robot_action = self.env.get_robot_action(second_augmented_state[:6], next_robot_action_type)

        next_augmented_state = self.env.augmented_state_transition(second_augmented_state, next_robot_action, None)

        if self.env.isTerminal(next_augmented_state[:6]):
            return 0

        next_human_action = self.env.get_rollout_observation(next_augmented_state, next_robot_action)

        # Recursive rollout
        return self.env.reward(next_augmented_state, next_robot_action, next_human_action) + self.gamma * self.rollout_helper(next_augmented_state,
                                                                                                     next_robot_action,
                                                                                                     next_human_action, depth + 1)

    def simulate(self, augmented_state, action_node, depth):
        """
        1. Simulates a trajectory from the start state down the search tree by picking the optimal action according to
           the tree policy (UCT) at each point in the tree and simulating observations (i.e., human actions).
        2. Incrementally builds the search tree (after every rollout) and updates the statistics of the visited nodes
           (the value and visitation count)
        3. Returns the value achieved from simulation

        :param augmented_state: (type: list) the augmented state (world state + latent human states) before robot and human action
        :param action_node: (type: human action node object) Current human action node for rollout
        :param depth: (type: int) depth of the search tree
        :return: (type: float) value from the current simulation
        """
        # Returns 0 upon reaching the max depth
        if math.pow(self.gamma, depth) < self.epsilon:
            return 0

        # Update belief
        action_node.update_belief(augmented_state)

        # Returns the env reward upon reaching the terminal state
        world_state = augmented_state[:6]

        if world_state[0] == world_state[1]:
            robot_action_type = 0
        else:
            # Finds optimal robot action based on the UCT policy
            robot_action_type = action_node.optimal_robot_action(self.c)
        robot_action = self.env.get_robot_action(world_state, robot_assistance_mode=robot_action_type)
        robot_action_node = action_node.robot_node_children[robot_action[0]]

        second_augmented_state = self.env.augmented_state_transition(augmented_state, robot_action, None)

        if self.env.isTerminal(second_augmented_state[:6]):
            return 0

        # If the robot action node is not in the tree, then returns the rollout value
        if robot_action_node == "empty":
            rollout_value = self.rollout(second_augmented_state, robot_action, action_node, depth)
            return rollout_value

        # Simulate human action
        human_action = self.env.get_rollout_observation(second_augmented_state, robot_action)
        next_augmented_state = self.env.augmented_state_transition(second_augmented_state, robot_action, human_action)

        next_action_node = robot_action_node.human_node_children[human_action[1]*4 + human_action[2]]


        # Creates new node if the next human action node is empty
        if next_action_node == "empty":
            new_human_action_node = HumanActionNode(self.env)
            next_action_node = robot_action_node.human_node_children[human_action[1]*4 + human_action[2]] = new_human_action_node

        # Compute value from recursive rollouts
        curr_reward = self.env.reward(second_augmented_state, robot_action, human_action)
        value = curr_reward + self.gamma * self.simulate(
            next_augmented_state,
            next_action_node, depth + 1)

        # Backups / update statistics
        robot_action_node.update_visited()
        robot_action_node.update_value(value)
        next_action_node.update_visited()
        next_action_node.update_value(value)

        return value
