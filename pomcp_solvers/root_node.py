import random as rand


class RootNode:
    def __init__(self, env, belief):
        """
        Initializes the special root node of the search tree.

        :param env: (type: Environment object) environment in which the robot and human operate
        :param belief: (type: List) the belief set of augmented state particles
        """
        self.type = "root"
        self.env = env
        self.belief = belief
        # List of robot action nodes children
        self.robot_node_children = self.init_children()

    def init_children(self):
        """
        Initializes all the robot node children of the root node to "empty".

        :return children: (type: List) initialized robot node children
        """
        # Initialize empty children for each robot action
        # For now only considering three types of robot actions -- No-assist, assist with and without explanations
        children = ["empty"] * self.env.robot_action_space.nvec[0]
        return children

    def optimal_robot_action(self, c):
        """
        Returns the optimal robot action to take from this node.

        :param c: (type: float) exploration constant
        :return: (type: list) optimal robot action
        """
        values = []
        for child in self.robot_node_children:
            if child == "empty":
                values.append(c)
            else:
                values.append(child.augmented_value(c))

        return values.index(max(values))

    def sample_state(self):
        """
        Randomly sample an initial augmented state from the current belief.

        :return: (type: list) sampled augmented state
        """
        return rand.choice(self.belief)

    def update_visited(self):
        """
        Does not keep/update the number of times of visiting this node.
        """
        pass

    def update_value(self, reward):
        """
        Does not keep/update value of the root.

        :param reward: (type: float) the immediate reward just received
        """
        pass

    def update_belief(self, augmented_state):
        """
        Does not update belief of the root.

        :param augmented_state: (type: list) the augmented state visiting this node
        """
        pass

    def get_children_values(self):
        """
        Returns the values of the robot children nodes of this node.

        :return values: (type: list) values of robot children nodes
        """
        values = [0] * len(self.robot_node_children)
        for i, child in enumerate(self.robot_node_children):
            if child != "empty":
                values[i] = child.value

        return values
