class RobotActionNode:
    def __init__(self, env):
        """
        Initializes a robot action node.

        :param env: (type: Environment object) environment in which the robot and human operate
        """
        self.type = "robot"
        self.env = env
        self.value = 0
        self.visited = 0
        self.position = None
        self.human_node_children = self.init_children()

    def init_children(self):
        """
        Initializes all the human node children of this node to "empty".

        :return children: (type: list) initialized human node children
        """
        # Initialize empty children for each human action
        # For now only considering the choice of the human (0-1) x (0-3).
        # TODO: Should I include whether the human accepted or rejected as a separate child here??
        children = ["empty"] * (self.env.human_action_space.nvec[2] * self.env.human_action_space.nvec[1])
        return children

    def augmented_value(self, c):
        """
        Returns the augmented value (value + exploration bonus) of taking this robot action.

        :return: (type: float) augmented value of robot action
        """
        return self.value + float(c) / self.visited

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
