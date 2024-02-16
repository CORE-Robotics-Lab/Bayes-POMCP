"""
Implementation of Bayes-POMCP for testing with simulated human models.
"""
import os
import sys

sys.path.append(os.getcwd())
import argparse
from pomcp_solvers.solver import *
from pomcp_solvers.root_node import *
from pomcp_solvers.robot_action_node import *
from pomcp_solvers.human_action_node import *
from pomcp_solvers.dynamic_users import *
import time
from collections import defaultdict
import json


def write_json(path, data, indent=4):
    class npEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.int32):
                return int(obj)
            return json.JSONEncoder.default(self, obj)

    with open(path, 'w') as file:
        json.dump(data, file, indent=indent, cls=npEncoder)


class Driver:
    def __init__(self, env, solver, num_steps, simulated_human, update_belief=True):
        """
        Initializes a driver : uses particle filter to maintain belief over hidden states,
        and uses POMCP to determine the optimal robot action

        :param env: (type: Environment) Instance of the FrozenLake environment
        :param solver: (type: POMCPSolver) Instance of the POMCP Solver for the robot policy
        :param num_steps: (type: int) number of actions allowed -- I think it's the depth of search in the tree
        :param simulated_human: (type: SimulatedHuman) the simulated human model
        :param update_belief: (type: bool) if set to True performs BA-POMCP, else it's regular POMCP
        """
        self.env = env
        self.solver = solver
        self.num_steps = num_steps
        self.simulated_human = simulated_human
        self.update_belief = update_belief
        self.num_world_states = len(self.env.world_state)
        self.num_total_states = self.num_world_states + 2  # Should change this to 1, as we are not looking at capability

    def invigorate_belief(self, current_human_action_node, parent_human_action_node, robot_action, human_action, env):
        """
        Invigorates the belief space when a new human action node is created
        Updates the belief to match the world state, whenever a new human action node is created
        :param current_human_action_node: Current human action node is the hao node.
        :param parent_human_action_node: Parent human action node is the h node (root of the current search tree).
        :param robot_action: Robot action (a) taken after parent node state
        :param human_action: Human action (o) in response to robot action
        :param env: gym env object to determine current world state of the Frozen Lake
        :return:
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
        :param human_action_node:
        :param env:
        :return:
        """
        if len(human_action_node.belief) == 0:
            print("Node belief is empty!!!")
            return

        # Update the belief (i.e., all particles) in the current node to match the current world state
        if human_action_node.belief[0][:self.num_world_states] != env.world_state:
            human_action_node.belief = [env.world_state[:] + [belief[-1]] for belief in human_action_node.belief]

    def updateBeliefTrust(self, human_action_node, human_action):
        """
        Updates the human trust parameter in the robot's belief based on the human's action
        :param human_action_node:
        :param human_action:
        :return:
        """
        human_accept, detect, human_choice_idx = human_action  # human accept: 0:no-assist, 1:accept, 2:reject

        for belief in human_action_node.belief:
            if human_accept != 0:  # In case of robot assistance
                # Update trust
                # index 0 of the particle is acceptance count, and index 1 is rejection count
                belief[self.num_world_states][
                    human_accept - 1] += 1

    def execute(self, round_num, render_game_states=False):
        """
        Executes one round of search with the POMCP policy
        :param round_num: (type: int) the round number of the current execution
        :return: (type: float) final reward from the environment
        """
        robot_actions = []
        human_actions = []
        all_states = []

        # create a deep copy of the env and the solver
        env = copy.deepcopy(self.env)
        solver = copy.deepcopy(self.solver)

        final_env_reward = 0

        # Initial human action
        robot_action = (0, None)  # No interruption (default assumption since human takes the first action)
        human_action = self.simulated_human.simulateHumanAction(env.world_state, robot_action)

        # Here we are adding to the tree as this will become the root for the search in the next turn
        human_action_node = HumanActionNode(env)
        # This is where we call invigorate belief... When we add a new human action node to the tree
        self.invigorate_belief(human_action_node, solver.root_action_node, robot_action, human_action, env)
        solver.root_action_node = human_action_node
        env.world_state = env.world_state_transition(env.world_state, robot_action, human_action)
        all_states.append(env.world_state[0])
        curr_reward = env.reward(env.world_state, (0, None), human_action)
        final_env_reward += curr_reward

        human_actions.append(human_action)

        for step in range(self.num_steps):
            t = time.time()
            if human_action[1] == 1:
                robot_action_type = 0
            else:
                robot_action_type = solver.search()  # One iteration of the POMCP search  # Here the robot action indicates the type of assistance

            robot_action = env.get_robot_action(env.world_state[:6], robot_action_type)
            robot_action_node = solver.root_action_node.robot_node_children[robot_action[0]]

            if robot_action_node == "empty":
                # We're not adding to the tree though here
                # It doesn't matter because we are going to update the root from h to hao
                robot_action_node = RobotActionNode(env)

            # Update the environment
            env.world_state = env.world_state_transition(env.world_state, robot_action, None)
            robot_action_node.position = env.world_state[0]

            all_states.append(env.world_state[0])
            if render_game_states:
                env.render(env.desc)

            # We use the real observation / human action (i.e., from the simulated human model)

            # Note here that it is not the augmented state
            # (the latent parameters are already defined in the SimulatedHuman model I think)
            human_action = self.simulated_human.simulateHumanAction(env.world_state, robot_action)
            human_action_node = robot_action_node.human_node_children[human_action[1] * 4 + human_action[2]]

            curr_reward = env.reward(env.world_state, robot_action, human_action)
            final_env_reward += curr_reward

            # Terminates if goal is reached
            if env.isTerminal(env.world_state):
                break

            if human_action_node == "empty":
                # Here we are adding to the tree as this will become the root for the search in the next turn
                human_action_node = robot_action_node.human_node_children[
                    human_action[1] * 4 + human_action[2]] = HumanActionNode(env)
                # This is where we call invigorate belief... When we add a new human action node to the tree
                self.invigorate_belief(human_action_node, solver.root_action_node, robot_action, human_action, env)

            # Update the environment
            solver.root_action_node = human_action_node  # Update the root node from h to hao
            env.world_state = env.world_state_transition(env.world_state, robot_action, human_action)

            # Updates the world state in the belief to match the actual world state
            self.updateBeliefWorldState(human_action_node, env)

            # Updates robot's belief of the human capability based on human action
            if self.update_belief:
                self.updateBeliefTrust(human_action_node, human_action)  # For now I'm updating every turn.

            if render_game_states:
                env.render(env.desc)

            robot_actions.append(robot_action)
            human_actions.append(human_action)

        return final_env_reward, human_actions, robot_actions, all_states


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--belief', type=int, default=0, help='Set to 1 to run Bayes-POMCP')
    parser.add_argument('--exp_num', type=int, default=1, help='1 for static trust experiments,'
                                                               '2 for static expertise experiments'
                                                               '3 for dynamic user experiments')

    args = parser.parse_args()

    seeds = [0, 1, 2]
    # To visualize rollout:
    user_data = defaultdict(lambda: defaultdict())

    # Human latent parameters (set different values for each test)
    true_trust = [(20, 80), (40, 60), (50, 50), (60, 40), (80, 20)]
    epsilon_vals = [0.5]
    true_capability = 0.85  # fixed - parameter (not used currently) at the start of the study

    # Initialize constants for setting up the environment
    num_choices = 3

    # factors for POMCP (also used in the environment for get_observations which uses UCT for human policy)
    gamma = 0.99  # gamma for terminating rollout based on depth in MCTS
    c = 20  # 400  # exploration constant for UCT (taken as R_high - R_low)
    e = 0.1  # For epsilon-greedy policy
    epsilon = math.pow(gamma, 30)  # tolerance factor to terminate rollout
    num_iter = 500
    num_steps = 100
    update_belief = args.belief == 1  # set to true for BA-POMCP otherwise it's just regular POMCP
    human_type = "epsilon_greedy" if update_belief else "random"

    # Executes num_tests of experiments
    num_test = len(true_trust)

    for SEED in seeds:
        print("*********************************************************************")
        print("Executing SEED {}......".format(SEED))
        print("*********************************************************************")

        for map_num in [4, 5, 7, 10, 11]:
            print("*********************************************************************")
            print("Executing MAP {}......".format(map_num))
            print("*********************************************************************")

            random.seed(SEED)
            np.random.seed(SEED)
            os.environ['PYTHONHASHSEED'] = str(SEED)

            mean_rewards = []
            std_rewards = []
            all_rewards = []

            for n in range(num_test):

                # Robot's belief of human parameters
                all_initial_belief_trust = []
                for _ in range(1000):
                    all_initial_belief_trust.append([1, 1])

                # Setup Driver
                map = MAPS["MAP" + str(map_num)]
                foggy = FOG["MAP" + str(map_num)]
                human_err = HUMAN_ERR["MAP" + str(map_num)]
                robot_err = ROBOT_ERR["MAP" + str(map_num)]
                env = FrozenLakeEnv(render_mode="human", desc=map, foggy=foggy, human_err=human_err,
                                    robot_err=robot_err, seed=SEED, human_type=human_type, update_belief=update_belief)

                # Executes num_rounds of search (calibration)
                num_rounds = 1
                total_env_reward = 0

                rewards = []
                for i in range(num_rounds):
                    # Re-initialize the human model and belief for each round...
                    initial_belief = []

                    # Reset the environment to initialize everything correctly
                    env.reset()
                    init_world_state = env.world_state

                    for b in range(len(all_initial_belief_trust)):
                        initial_belief.append(init_world_state[:] + [all_initial_belief_trust[b]])

                    root_node = RootNode(env, initial_belief)
                    solver = POMCPSolver(epsilon, env, root_node, num_iter, c)

                    simulated_human = SimulatedHuman(env, true_trust=true_trust[n],
                                                     type="epsilon_greedy",
                                                     epsilon=0.25,
                                                     )  # This does not change

                    driver = Driver(env, solver, num_steps, simulated_human, update_belief=update_belief)

                    # We should only change the true state for every round (or after every termination)
                    driver.env.reset()
                    env_reward, human_actions, robot_actions, all_states = driver.execute(i)
                    print(env_reward)
                    rewards.append(env_reward)
                    total_env_reward += env_reward

                    # To visualize the rollout...
                    history = []
                    for t in range(len(human_actions) - 1):
                        history.append({"human_action": list(human_actions[t]),
                                        "robot_action": list(robot_actions[t])})

                all_rewards.append(rewards)
                mean_rewards.append(np.mean(rewards, dtype=np.int64))
                std_rewards.append(np.std(rewards))

            print("===================================================================================================")
            print("===================================================================================================")
            print(mean_rewards, std_rewards)
            print("===================================================================================================")
            print(f"{np.mean(mean_rewards):.3f}, {np.std(mean_rewards) / np.sqrt(10):.3f}")
            print("===================================================================================================")

            all_rewards = np.array(all_rewards)
            print(all_rewards)

            combined_array = np.column_stack((mean_rewards, std_rewards))

            # Specify the CSV file path
            if update_belief:
                csv_file_path = 'frozen_lake/files/bayes_pomcp/'
            else:
                csv_file_path = f'frozen_lake/files/pomcp/'

            # Save the 2D array as a CSV file
            if not os.path.exists(csv_file_path):
                os.makedirs(csv_file_path)

            np.savetxt(csv_file_path + f'output_MAP_{map_num}_SEED_{SEED}_500.csv', combined_array, delimiter=',',
                       fmt='%.3f')

