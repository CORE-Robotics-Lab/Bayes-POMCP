from pomcp_solvers.simulated_human import *
import time

from frozen_lake.frozen_lake_interface import FrozenLakeEnvInterface

order = [4, 8, 6]
heuristic_order = [0, 1]  # First one is the order of interrupting agent, second is the order of taking control agent.
random.shuffle(heuristic_order)
CONDITION = {
    'practice': [0, 1, 2, 3],
    'pomcp': [order[0], order[0] + 1],
    'pomcp_inverse': [order[1], order[1] + 1],
    'interrupt': [order[2] + heuristic_order[0]],
    'take_control': [order[2] + heuristic_order[1]]
}


def view_rollout(user_data, rollout_idx=4, SEED=0):
    """
    Visualize the human and robot actions in one round in the interface
    :param user_data: (type: json) the recorded data of playing history from a user
    :param rollout_idx: (type: int) the round number of the rollout
    :param SEED: (type: int) the random seed
    :return:
    """
    # Ignore the practice rollouts
    map_num = user_data["mapOrder"][rollout_idx]
    rollout = user_data[str(rollout_idx)]["history"]
    map = MAPS["MAP" + str(map_num)]
    foggy = FOG["MAP" + str(map_num)]

    human_err = HUMAN_ERR["MAP" + str(map_num)]
    robot_err = ROBOT_ERR["MAP" + str(map_num)]

    round_num = 1

    # Create Env based on the first rollout after practice rounds...
    env = FrozenLakeEnvInterface(desc=map, foggy=foggy, human_err=human_err, robot_err=robot_err,
                                 render_mode="human", true_human_trust=(5, 50),
                                 beta=0.9, c=20, gamma=0.99, seed=SEED,
                                 human_type="epsilon_greedy", round=round_num)

    # Reset the environment to initialize everything correctly
    env.reset(round_num=round_num)

    detection_num = 0
    step = 0

    for t in range(len(rollout)):
        curr_human_action = rollout[t]['human_action']
        curr_robot_action = rollout[t]['robot_action']

        print(curr_human_action, curr_robot_action)

        is_accept, detecting, action = curr_human_action
        if detecting:
            # TODO: Render detection action
            env.render(round_num, None, None, env.world_state)
            time.sleep(2.5)
            detection_num += 1
            step += 1  # one extra step penalty for using detection

        step += 1

        # Account for human action transition (not visualized)...
        env.world_state = env.world_state_transition(env.world_state, None, curr_human_action)
        env.world_state = env.world_state_transition(env.world_state, curr_robot_action, None)
        # Visualize after robot action
        env.render(round_num=round_num, human_action=curr_human_action, robot_action=curr_robot_action,
                   world_state=env.world_state)
        if curr_robot_action[0] > 0:
            time.sleep(2.5)
            # input("Press enter to continue")
        time.sleep(2.5)

