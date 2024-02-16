"""
Script to evaluate different baselines for comparing human-robot performance
Baselines:
- Random (Robot chooses a random policy to assist user)
- Inverse Reward POMCP
- Static Policy (always provides one type of assistance: assist / assist + explanations)
- No assistance
"""
import os
import sys
sys.path.append(os.getcwd())
import time
from pomcp_solvers.simulated_human import *
from visualize_rollouts import view_rollout


class RandomAgent:
    def __init__(self, num_actions):
        """
        Initializes a robot agent choosing random actions
        :param num_actions: (type: int) number of available actions
        """
        self.num_actions = num_actions

    def get_action(self, env):
        """
        Return a random robot action following random policy

        :param env: (type: Environment) Instance of the FrozenLake environment
        :return: (type: int) a random robot action index
        """
        return np.random.choice(self.num_actions)


class StaticAgent:
    def __init__(self, fixed_action=1):
        """
        Initializes a robot agent only having a fixed action
        :param fixed_action: (type: int) a fixed action index that the robot can choose
        """
        self.robot_action = fixed_action

    def get_action(self, env):
        """
        Return a fixed robot action

        :param env: (type: Environment) Instance of the FrozenLake environment
        :return: (type: int) a fixed robot action index
        """
        return self.robot_action


class NoAssistAgent:
    def __init__(self):
        """
        Initializes a robot agent that does nothing
        """
        self.robot_action = 0

    def get_action(self, env):
        """
        Return a fixed robot action

        :param env: (type: Environment) Instance of the FrozenLake environment
        :return: (type: int) a robot action index (=0) that means not taking actions
        """
        return self.robot_action


class HeuristicAgent:
    def __init__(self, type):
        """
        Initializes a robot agent using a heuristic policy to choose actions (interrupt/take control)
        :param type: (type: int) The action type of intervention (interrupt/take control + explanation/no explanation)
        """
        self.robot_action = 0
        self.type = type
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
        elif len(current_path) > 1 and len(last_path) > 1 and len(last_path) <= len(current_path) and \
                self.num_interrupt < 3:
            self.robot_action = self.type
            self.num_interrupt += 1
        else:
            self.robot_action = 0

        return self.robot_action


def execute(round_num, num_steps, env, human_agent, robot_agent=None):
    """
    Executes one round of search with the simulated human agent and the robot agent we choose
    :param round_num: (type: int) the round number of the current execution
    :param num_steps: (type: int) the number of steps to execute
    :param human_agent: (type: SimulatedHuman) the simulated human agent
    :param robot_agent: the robot agent (RandomAgent / StaticAgent / NoAssistAgent / HeuristicAgent)
    :return final_env_reward: (type: float) final reward from the environment
    :return human_actions: (type: List(int)) a list of executed human actions
    :return robot_actions: (type: List(int)) a list of executed robot actions
    """
    robot_actions = []
    human_actions = []
    num_actions = 3

    if robot_agent is None:
        robot_agent = RandomAgent(num_actions)

    final_env_reward = 0

    human_action = human_agent.simulateHumanAction(env.world_state, (0, None))
    robot_action = (0, None)

    # update the environment
    env.world_state = env.world_state_transition(env.world_state, None, human_action)
    curr_reward = env.reward(env.world_state, (0, None), human_action)
    final_env_reward += curr_reward

    human_actions.append(human_action)

    for step in range(num_steps):
        if robot_action[0] or human_action[1]:
            robot_action_type = 0  # Cannot interrupt twice successively
        else:
            robot_action_type = robot_agent.get_action(env)

        robot_action = env.get_robot_action(env.world_state, robot_action_type)

        # update the environment
        env.world_state = env.world_state_transition(env.world_state, robot_action, None)

        human_action = human_agent.simulateHumanAction(env.world_state, robot_action)

        curr_reward = env.reward(env.world_state, robot_action, human_action)
        final_env_reward += curr_reward

        if env.isTerminal(env.world_state):
            break

        # update the environment
        env.world_state = env.world_state_transition(env.world_state, None, human_action)
        robot_actions.append(robot_action)
        human_actions.append(human_action)

    # Final env reward is calculated based on the true state of the tiger and what the human finally decided to do
    # The step loop terminates if the env terminates

    return final_env_reward, human_actions, robot_actions


if __name__ == '__main__':
    # Initialize constants for setting up the environment
    max_steps = 100
    num_choices = 3

    # Human latent parameters (set different values for each test)
    true_trust = [(20, 80), (40, 60), (60, 40), (80, 20)]
    epsilon_vals = [0.5]

    true_capability = 0.85  # fixed - parameter (currently unused) at the start of the study

    # factors for POMCP (also used in the environment for get_observations which uses UCT for human policy)
    gamma = 0.99
    c = 10  # Exploration bonus
    beta = 0.9

    # Executes num_tests of experiments
    num_test = len(true_trust)

    for map_num in [4, 5, 7, 10, 12]:
        print("*********************************************************************")
        print("Executing MAP {}......".format(map_num))
        print("*********************************************************************")

        # Set appropriate seeds
        for SEED in [0]:
            random.seed(SEED)
            np.random.seed(SEED)
            os.environ['PYTHONHASHSEED'] = str(SEED)

            mean_rewards = []
            std_rewards = []
            all_rewards = []

            for n in range(num_test):

                # Setup Driver
                map = MAPS["MAP" + str(map_num)]
                foggy = FOG["MAP" + str(map_num)]
                human_err = HUMAN_ERR["MAP" + str(map_num)]
                robot_err = ROBOT_ERR["MAP" + str(map_num)]

                env = FrozenLakeEnv(render_mode="human", desc=map, foggy=foggy, human_err=human_err,
                                    robot_err=robot_err, seed=SEED, human_type="epsilon_greedy")

                # Reset the environment to initialize everything correctly
                robot_policy = HeuristicAgent(type=1)

                # Executes num_rounds of search (calibration)
                num_rounds = 10
                total_env_reward = 0

                rewards = []
                for i in range(num_rounds):
                    # Re-initialize the human model for each round...
                    simulated_human = SimulatedHuman(env, true_trust=true_trust[n],
                                                     type="epsilon_greedy",
                                                     epsilon=epsilon_vals[0])

                    # We should only change the true state for every round (or after every termination)
                    env.reset()

                    env_reward, human_actions, robot_actions = execute(round_num=i, num_steps=max_steps, env=env,
                                         human_agent=simulated_human, robot_agent=robot_policy)
                    rewards.append(env_reward)
                    total_env_reward += env_reward

                    # To visualize the rollout...

                    history = []
                    for t in range(len(human_actions) - 1):
                        history.append({"human_action": list(human_actions[t]),
                                        "robot_action": list(robot_actions[t])})

                    user_data = {"mapOrder": [map_num],
                                 str(i): {
                                     "history": history
                                    }
                                 }



                all_rewards.append(rewards)
                mean_rewards.append(np.mean(rewards))
                std_rewards.append(np.std(rewards))
            print("===================================================================================================")
            print("===================================================================================================")
            print(mean_rewards, std_rewards)
            print("===================================================================================================")
            print(f"{np.mean(mean_rewards):.3f}, {np.std(mean_rewards)/np.sqrt(10):.3f}")
            print("===================================================================================================")

            all_rewards = np.array(all_rewards)

            # Combine the arrays horizontally to create a 2D array
            combined_array = np.column_stack((mean_rewards, std_rewards))

            # Specify the CSV file path
            csv_file_path = f'frozen_lake/files/heuristic_test/output_{map_num}.csv'

            # Save the 2D array as a CSV file
            np.savetxt(csv_file_path, combined_array, delimiter=',', fmt='%.3f')