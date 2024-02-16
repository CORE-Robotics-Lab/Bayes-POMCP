import numpy as np
from scipy.stats import beta


def get_user_action_from_beta(beta_params, curr_robot_action=(0, None)):
    robot_action_type = curr_robot_action[0]
    acceptance_prob = np.random.beta(beta_params[0], beta_params[1])
    detect_grid_prob = 0.2
    e = np.random.uniform()
    if e < acceptance_prob:
        human_action = 0  # Accept
        accept = True
    else:
        if robot_action_type == 0:
            human_action = 1  # Detect
        else:
            # For take control or interrupt
            if np.random.uniform() < detect_grid_prob:
                human_action = 1  # Detect
            else:
                human_action = 2  # Oppose

    return human_action


def prediction_counts_after_belief_update(particle_set, curr_robot_action):
    prediction_counts = np.array([0, 0, 0])
    num_particles = particle_set.shape[0]

    predicted_human_actions = np.zeros((num_particles, 1))

    # Random draw...
    random_samples = np.random.uniform(0, 1, num_particles).reshape(-1, 1)
    acceptance_probs = np.random.beta(particle_set[:, 0], particle_set[:, 1]).reshape(-1, 1)
    if curr_robot_action == 0:
        predicted_human_actions[random_samples > acceptance_probs] = 2

    else:
        # Random draw for choosing between oppose and detect
        temp = np.zeros((num_particles, 1))
        temp[random_samples > acceptance_probs] = 1
        random_samples = random_samples * temp

        # Threshold for oppose vs detect
        threshold = 0.9
        predicted_human_actions[np.logical_and(random_samples > 0.01, random_samples < threshold)] = 1
        predicted_human_actions[random_samples > threshold] = 2

    idxs, count_vals = np.unique(predicted_human_actions, return_counts=True)
    prediction_counts[idxs.astype(int)] = count_vals
    return prediction_counts


def distance_from_true_action(user_action, predicted_action):
    human_accept, detect, direction = user_action
    # 2: reject/oppose; 1: detect; 0:accept
    if detect:
        final_action = 1
    elif human_accept == 2:
        final_action = 2
    else:
        final_action = 0
    return 1 if final_action == predicted_action else 0


def get_entropy(beta_params):
    return beta.entropy(beta_params[0], beta_params[1])