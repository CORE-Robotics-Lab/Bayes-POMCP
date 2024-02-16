import random as rand


class HumanActionNode:
    def __init__(self, env):
        """
        Initializes a human action node.

        :param env: (type: Environment object) environment in which the robot and human operate
        """
        self.type = "human"
        self.env = env
        self.robot_node_children = self.init_children()
        self.value = 0
        self.visited = 0
        self.belief = []

    def init_children(self):
        """
        Initializes all the robot node children of this node to "empty".

        :return children: (type: List) initialized robot node children
        """
        # Initialize empty children for each robot action
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

    def update_value(self, reward):
        """
        Updates the value of the search node.

        :param reward: (type: float) the immediate reward just received
        """
        self.value += (reward - self.value) / self.visited

    def update_visited(self):
        """
        Increments the number of times of visiting this node.

        """
        self.visited += 1

    def update_belief(self, augmented_state):
        """
        Add new augmented state perticle to the current belief set.

        :param augmented_state: (type: List) the augmented state visiting this node
        """
        self.belief.append(augmented_state)

    def sample_state(self):
        """
        Samples an augmented state from the current belief set.

        :return: (type: List) a sampled augmented state
        """
        if len(self.belief) == 0:
            print('wrong!!!')
            print(self.env.s)
        return rand.choice(self.belief)

    def get_children_values(self):
        """
        Returns the values of the robot children nodes of this node.

        :return values: (type: List) values of robot children nodes
        """
        values = [0] * len(self.robot_node_children)
        for i, child in enumerate(self.robot_node_children):
            if child != "empty":
                values[i] = child.value

        return values
