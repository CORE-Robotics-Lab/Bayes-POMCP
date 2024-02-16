from abc import ABC

from frozen_lake.frozen_lake_env import FrozenLakeEnv
import pygame
import os
from typing import Optional

# First four rounds are demo + practice
CONDITION = {
    'practice': [0, 1, 2, 3],
}


class FrozenLakeEnvInterface(FrozenLakeEnv, ABC):
    """
    FrozenLake user interface
    """
    metadata = {
        "render_modes": ["human", "ansi", "rgb_array"],
        "render_fps": 4,
    }

    def render(self, round_num, human_action, robot_action, world_state, end_detecting=0, truncated=False,
               timeout=False):
        """
        Render the user interface using pygame
        :param round_num: (type: int): The round number of game the user is playing
        :param human_action: (type: tuple): The action the human user chooses
        :param robot_action: (type: tuple): The action the robot agent chooses
        :param world_state: (type: list): A list that contains the current world state information
                                        (current position, last position, slippery regions, etc.)
        :param end_detecting: (type: int): If the human is about to exit the detection mode
        :param truncated: (type: bool): If the robot is stepping on an ice hole
        :param timeout: (type: bool): If the human user runs out of the step number
        :return:
        """
        window_width = self.window_size[0]
        window_height = self.window_size[1]
        map_size = self.window_size[0] - 256 * 2
        if robot_action:
            robot_type, robot_direction = robot_action
        else:
            robot_type, robot_direction = None, None
        if human_action:
            _, detecting, human_direction = human_action
        else:
            detecting, human_direction = None, None
        position, last_position, human_slippery, robot_slippery, human_err, robot_err = world_state

        class TextBox(pygame.sprite.Sprite):
            def __init__(self, surface):
                pygame.sprite.Sprite.__init__(self)
                self.initFont()
                self.image = None
                self.group = None
                self.ncol = 8
                self.system_rect = pygame.Rect(0, 0, 256, map_size)
                self.robot_rect = pygame.Rect(256 + map_size, 0, 256, map_size)
                self.system_top = map_size / 2
                self.robot_top = 100
                self.robot_left = map_size + 256
                self.system_left = map_size + 256

                self.initImage(surface)
                self.initGroup()

            def initFont(self):
                pygame.font.init()
                self.font = pygame.font.Font(pygame.font.match_font('calibri'), 20)
                self.font_bold = pygame.font.Font(pygame.font.match_font('calibri', bold=True), 20)

            def initImage(self, surface):
                self.image = surface
                self.image.fill((255, 255, 255), rect=self.system_rect)
                self.image.fill((255, 255, 255), rect=self.robot_rect)

            def setText(self, robot=None, system=None):
                tmp = pygame.display.get_surface()

                if system is not None:
                    x_pos = self.system_left + 5
                    y_pos = self.system_top + 5
                    x = self.font_bold.render("System:", True, (0, 0, 0))
                    tmp.blit(x, (x_pos, y_pos))
                    y_pos += 20
                    words = system.split(' ')
                    for t in words:
                        # print(t, x_pos)
                        if t == 'NOT':
                            x = self.font.render(t + " ", True, (255, 0, 0))
                        elif t in ["ENTER", "SPACE", "BACKSPACE"]:
                            x = self.font_bold.render(t + " ", True, (0, 0, 0))
                        else:
                            x = self.font.render(t + " ", True, (0, 0, 0))
                        if x_pos + x.get_width() < self.image.get_width() - 5:
                            tmp.blit(x, (x_pos, y_pos))
                            x_pos += x.get_width()
                        else:
                            x_pos = self.system_left + 5
                            y_pos += 20
                            tmp.blit(x, (x_pos, y_pos))
                            x_pos += x.get_width()
                pygame.draw.line(tmp, (0, 0, 0), (self.robot_left, self.system_top),
                                 (self.robot_left + 256, self.system_top))

                if robot is not None:
                    x_pos = self.robot_left + 5
                    y_pos = self.robot_top + 5
                    x = self.font_bold.render("Robot:", True, (0, 0, 0))
                    tmp.blit(x, (x_pos, y_pos))
                    y_pos += 20
                    words = robot.split(' ')
                    for t in words:
                        if t == 'n':
                            x = self.font.render("", True, (0, 0, 0))
                            x_pos = self.robot_left + 5
                            y_pos += 36
                        elif t not in ["slippery", "region.", "hole.", "longer", "way."]:
                            x = self.font.render(t + " ", True, (0, 0, 0))
                        else:
                            x = self.font_bold.render(t + " ", True, (0, 0, 0))
                        if x_pos + x.get_width() < self.image.get_width() - 5:
                            tmp.blit(x, (x_pos, y_pos))
                            x_pos += x.get_width()
                        else:
                            x_pos = self.robot_left + 5
                            y_pos += 20
                            tmp.blit(x, (x_pos, y_pos))
                            x_pos += x.get_width()

            def initGroup(self):
                self.group = pygame.sprite.GroupSingle()
                self.group.add(self)

        if self.window_surface is None:
            pygame.init()
            pygame.display.init()
            pygame.display.set_caption("Frozen Lake")
            self.window_surface = pygame.display.set_mode(self.window_size)

        assert (
                self.window_surface is not None
        ), "Something went wrong with pygame. This should never happen."

        if self.clock is None:
            self.clock = pygame.time.Clock()
        if self.hole_img is None:
            file_name = os.path.join(os.getcwd(), "frozen_lake/img/hole_new.png")
            self.hole_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.cracked_hole_img is None:
            file_name = os.path.join(os.getcwd(), "frozen_lake/img/cracked_hole_1.png")
            self.cracked_hole_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.goal_img is None:
            file_name = os.path.join(os.getcwd(), "frozen_lake/img/goal.png")
            self.goal_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.start_img is None:
            file_name = os.path.join(os.getcwd(), "frozen_lake/img/stool.png")
            self.start_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.elf_images is None:
            elfs = [
                os.path.join(os.getcwd(), "frozen_lake/img/robot{}.png".format(int(round_num / 2))),
                os.path.join(os.getcwd(), "frozen_lake/img/robot{}.png".format(int(round_num / 2))),
                os.path.join(os.getcwd(), "frozen_lake/img/robot{}.png".format(int(round_num / 2))),
                os.path.join(os.getcwd(), "frozen_lake/img/robot{}.png".format(int(round_num / 2))),
            ]
            self.elf_images = [
                pygame.transform.scale(pygame.image.load(f_name), self.cell_size)
                for f_name in elfs
            ]

        if self.fog_img is None:
            file_name = os.path.join(os.getcwd(), "frozen_lake/img/ice.png")
            self.fog_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )

        if self.smoke_img is None:
            file_name = os.path.join(os.getcwd(), "frozen_lake/img/steam_2.png")
            self.smoke_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )

        if self.slippery_img is None:
            file_name = os.path.join(os.getcwd(), "frozen_lake/img/slippery_1.png")
            self.slippery_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )

        desc = self.desc.tolist()
        foggy = self.fog.tolist()

        for y in range(self.nrow):
            for x in range(self.ncol):
                pos = (x * self.cell_size[0] + 256, y * self.cell_size[1])
                rect = (*pos, *self.cell_size)

                if foggy[y][x] == b"F":
                    self.window_surface.blit(self.fog_img, pos)
                else:
                    self.window_surface.fill((255, 255, 255), rect=rect)
                if (y, x) in human_slippery:
                    self.window_surface.blit(self.slippery_img, pos)
                    if foggy[y][x] == b"F":
                        self.window_surface.blit(self.smoke_img, pos)
                if desc[y][x] == b"H":
                    self.window_surface.blit(self.hole_img, pos)
                elif desc[y][x] == b"G":
                    self.window_surface.blit(self.goal_img, pos)
                elif desc[y][x] == b"B":
                    self.window_surface.blit(self.start_img, pos)

                pygame.draw.rect(self.window_surface, (180, 200, 230), rect, 1)

        # Add legend
        self.window_surface.fill((255, 255, 255), rect=(0, map_size, window_width, window_height - map_size))
        pygame.draw.line(self.window_surface, (0, 0, 0), (0, map_size),
                         (window_width, map_size))
        pygame.font.init()
        font_bold = pygame.font.Font(pygame.font.match_font('calibri', bold=True), 25)
        x = font_bold.render("Legend", True, (0, 0, 0))
        self.window_surface.blit(x, (10, map_size + 20))

        # Ice
        left_pos = 10 + x.get_width() + 20
        top_pos = map_size + 20
        font = pygame.font.Font(pygame.font.match_font('calibri'), 25)
        x = font.render("Non-fog", True, (0, 0, 0))
        self.window_surface.blit(x, (left_pos, top_pos))
        self.window_surface.blit(self.smoke_img, (left_pos + x.get_width() + 5, top_pos))
        pygame.draw.rect(self.window_surface, (180, 200, 230),
                         ((left_pos + x.get_width() + 5, top_pos), self.cell_size), 1)

        # Fog
        font = pygame.font.Font(pygame.font.match_font('calibri'), 25)
        x = font.render("       Fog", True, (0, 0, 0))
        self.window_surface.blit(x, (left_pos, top_pos + 120))
        self.window_surface.blit(self.fog_img, (left_pos + x.get_width() + 5, top_pos + 120))
        pygame.draw.rect(self.window_surface, (180, 200, 230),
                         ((left_pos + x.get_width() + 5, top_pos + 120), self.cell_size), 1)

        # Slippery
        left_pos += 120 + x.get_width()
        font = pygame.font.Font(pygame.font.match_font('calibri'), 25)
        x = font.render("Slippery", True, (0, 0, 0))
        self.window_surface.blit(x, (left_pos, top_pos))
        x_p = font.render("(No fog)", True, (0, 0, 0))
        self.window_surface.blit(x_p, (left_pos, top_pos + x.get_height()))
        self.window_surface.blit(self.slippery_img, (left_pos + x.get_width() + 5, top_pos))
        pygame.draw.rect(self.window_surface, (180, 200, 230),
                         ((left_pos + x.get_width() + 5, top_pos), self.cell_size), 1)

        x = font.render("Slippery", True, (0, 0, 0))
        self.window_surface.blit(x, (left_pos, top_pos + 120))
        x_p = font.render("(Fog)", True, (0, 0, 0))
        self.window_surface.blit(x_p, (left_pos, top_pos + x.get_height() + 120))
        self.window_surface.blit(self.slippery_img, (left_pos + x.get_width() + 5, top_pos + 120))
        self.window_surface.blit(self.smoke_img, (left_pos + x.get_width() + 5, top_pos + 120))
        pygame.draw.rect(self.window_surface, (180, 200, 230),
                         ((left_pos + x.get_width() + 5, top_pos + 120), self.cell_size), 1)

        # Hole
        left_pos += 120 + x.get_width()
        font = pygame.font.Font(pygame.font.match_font('calibri'), 25)
        x = font.render("Hole", True, (0, 0, 0))
        self.window_surface.blit(x, (left_pos, top_pos))
        self.window_surface.blit(self.hole_img, (left_pos + x.get_width() + 5, top_pos))
        pygame.draw.rect(self.window_surface, (180, 200, 230),
                         ((left_pos + x.get_width() + 5, top_pos), self.cell_size), 1)

        # Goal
        left_pos += 120 + x.get_width()
        font = pygame.font.Font(pygame.font.match_font('calibri'), 25)
        x = font.render("Goal", True, (0, 0, 0))
        self.window_surface.blit(x, (left_pos, top_pos))
        self.window_surface.blit(self.goal_img, (left_pos + x.get_width() + 5, top_pos))

        # paint the elf
        bot_row, bot_col = position // self.ncol, position % self.ncol
        cell_rect = (bot_col * self.cell_size[0] + 256, bot_row * self.cell_size[1])
        last_action = human_direction
        elf_img = self.elf_images[0]

        # Robot notification
        textbox = TextBox(self.window_surface)
        ACTIONS = ["LEFT", "DOWN", "RIGHT", "UP"]
        human_action_name = None
        robot_action_name = None
        if last_action is not None:
            human_action_name = ACTIONS[last_action]
        if robot_direction is not None:
            robot_action_name = ACTIONS[robot_direction]
        if timeout:
            textbox.setText(
                system="You've run out of the step number. You failed. "
                       "Please finish the survey and ask the experimenter to start a new game.")
        elif detecting and human_direction is None:
            self.window_surface.blit(elf_img, cell_rect)
            textbox.setText(system="You're entering detection mode. Press arrow keys to check the surrounding grids.")
        elif end_detecting == 1:
            self.window_surface.blit(elf_img, cell_rect)
            textbox.setText(system="You're exiting detection mode and back to navigation mode.")
        elif end_detecting == 2:
            self.window_surface.blit(elf_img, cell_rect)
            textbox.setText(
                system="You're out of attempts for using the detection sensor. "
                       "Press BACKSPACE again to exit detection mode.")
        elif detecting:
            self.window_surface.blit(elf_img, cell_rect)
            s = self.move(position, human_direction)
            if desc[s // self.ncol][s % self.ncol] in b'SH':
                is_slippery = True
            else:
                is_slippery = False
            left = (s % self.ncol) * self.cell_size[0] + 256
            top = (s // self.ncol) * self.cell_size[1]
            if is_slippery:
                pygame.draw.rect(self.window_surface, (255, 0, 0),
                                 pygame.Rect(left, top, self.cell_size[0], self.cell_size[1]), 4)

                textbox.setText(
                    system="The region you're detecting is NOT safe! Press BACKSPACE again to exit detection mode.")
            else:
                pygame.draw.rect(self.window_surface, (0, 255, 0),
                                 pygame.Rect(left, top, self.cell_size[0], self.cell_size[1]), 4)

                textbox.setText(
                    system="The region you're detecting is safe. Press BACKSPACE again to exit detection mode.")
        else:
            if robot_type == 1:  # interrupt
                system_prompt = "Press ENTER and then make your next choice."
                robot_prompt = "Your last choice was {}. I chose to stay. Please choose an action again.".format(
                    human_action_name)

            elif robot_type == 2:  # control
                system_prompt = "Press ENTER and then make your next choice."
                robot_prompt = "Your last choice was {}. My action was {}.".format(
                    human_action_name, robot_action_name)

            elif robot_type == 3:  # interrupt_w_explain
                last_row = last_position // self.ncol
                last_col = last_position % self.ncol
                if (desc[last_row][last_col] in b'S' and (
                        (last_row, last_col) not in self.robot_err or ((last_row, last_col) in robot_err))) or \
                        (desc[last_row][last_col] in b'F' and (
                                (last_row, last_col) in self.robot_err and ((last_row, last_col) not in robot_err))):
                    system_prompt = "Press ENTER and then make your next choice."
                    robot_prompt = "Your last choice was {}. I chose to stay. " \
                                   "n Going {} might step into a slippery region. " \
                                   "Please choose an action again. ".format(
                                    human_action_name, human_action_name)
                elif desc[last_row][last_col] in b'H':
                    system_prompt = "Press ENTER and then make your next choice."
                    robot_prompt = "Your last choice was {}. I chose to stay. n Going {} will step into a hole. " \
                                   "Please choose an action again. ".format(
                                    human_action_name, human_action_name)
                else:
                    system_prompt = "Press ENTER and then make your next choice."
                    robot_prompt = "Your last choice was {}. I chose to stay. n Going {} might take a longer way. " \
                                   "Please choose an action again.".format(
                                    human_action_name, human_action_name)

            elif robot_type == 4:  # control_w_explain
                last_row = last_position // self.ncol
                last_col = last_position % self.ncol
                if (desc[last_row][last_col] in b'S' and (
                        (last_row, last_col) not in self.robot_err or ((last_row, last_col) in robot_err))) or \
                        (desc[last_row][last_col] in b'F' and (
                                (last_row, last_col) in self.robot_err and ((last_row, last_col) not in robot_err))):
                    system_prompt = "Press ENTER and then make your next choice."
                    robot_prompt = "Robot: Your last choice was {}. My action was {}. " \
                                   "n Going {} might step into a slippery region. ".format(
                        human_action_name, robot_action_name, human_action_name)
                elif desc[last_row][last_col] in b'H':
                    system_prompt = "Press ENTER and then make your next choice."
                    robot_prompt = "Your last choice was {}. My action was {}. n Going {} will step into a hole. ".format(
                        human_action_name, robot_action_name, human_action_name)
                else:
                    system_prompt = "Press ENTER and then make your next choice."
                    robot_prompt = "Your last choice was {}. My action was {}. n Going {} might take a longer way. ".format(
                        human_action_name, robot_action_name, human_action_name)

            else:
                robot_prompt = "Your last choice was {}. I followed your choice.".format(human_action_name)

            if truncated:
                last_row, last_col = last_position // self.ncol, last_position % self.ncol
                if robot_type == 0:
                    hole_row, hole_col = last_row, last_col
                else:
                    hole_row, hole_col = self.inc(last_row, last_col, robot_direction)
                last_cell_rect = (hole_col * self.cell_size[0] + 256, hole_row * self.cell_size[1])
                if desc[hole_row][hole_col] in b"H":
                    last_cell_rect = (hole_col * self.cell_size[0] + 256, hole_row * self.cell_size[1])
                    print(hole_row, hole_col)
                else:
                    for r in [-1, 0, 1]:
                        for c in [-1, 0, 1]:
                            hole_row = last_row + r
                            hole_col = last_col + c
                            if 0 <= hole_row < self.ncol and 0 <= hole_col < self.ncol:
                                if desc[hole_row][hole_col] in b'H':
                                    last_cell_rect = (hole_col * self.cell_size[0] + 256, hole_row * self.cell_size[1])
                                    print(hole_row, hole_col)
                                    break
                self.window_surface.blit(self.cracked_hole_img, last_cell_rect)
                if round_num in CONDITION['practice']:
                    system_prompt = "You failed. Please press ENTER to restart."
                    robot_prompt = "Your last choice was {}. I followed your choice. I slipped into a hole.".format(
                        human_action_name, robot_action_name)
                    textbox.setText(robot=robot_prompt, system=system_prompt)
                    self.window_surface.blit(self.elf_images[0], (256 + map_size, 0))
                elif robot_type in [2, 4]:  # taking control
                    system_prompt = "You failed. Please press ENTER to restart."
                    robot_prompt = "Your last choice was {}. My action was {}. I slipped into a hole.".format(
                        human_action_name, robot_action_name)
                    textbox.setText(robot=robot_prompt, system=system_prompt)
                    self.window_surface.blit(self.elf_images[0], (256 + map_size, 0))
                else:
                    system_prompt = "You failed. Please press ENTER to restart."
                    robot_prompt = "Your last choice was {}. I followed your choice. I slipped into a hole.".format(
                        human_action_name)
                    textbox.setText(robot=robot_prompt, system=system_prompt)
                    self.window_surface.blit(self.elf_images[0], (256 + map_size, 0))

            elif desc[bot_row][bot_col] == b"G":
                if round_num in CONDITION['practice']:
                    system_prompt = "You successfully reached the goal.Please ask the experimenter to start a new game."
                    robot_prompt = "Your last choice was {}. I followed your choice.".format(
                        human_action_name)
                    textbox.setText(robot=robot_prompt, system=system_prompt)
                    self.window_surface.blit(self.elf_images[0], (256 + map_size, 0))
                elif robot_type == 2 or robot_type == 4:
                    system_prompt = "You successfully reached the goal.Please ask the experimenter to start a new game."
                    if human_action_name == robot_action_name:
                        robot_prompt = "Your last choice was {}. I followed your choice.".format(
                            human_action_name, robot_action_name)
                    else:
                        robot_prompt = "Your last choice was {}. My action was {}.".format(
                            human_action_name, robot_action_name)
                    textbox.setText(robot=robot_prompt, system=system_prompt)
                    self.window_surface.blit(self.elf_images[0], (256 + map_size, 0))
                else:
                    system_prompt = "You successfully reached the goal.Please ask the experimenter to start a new game."
                    robot_prompt = "Your last choice was {}. I followed your choice.".format(
                        human_action_name)
                    textbox.setText(robot=robot_prompt, system=system_prompt)
                    self.window_surface.blit(self.elf_images[0], (256 + map_size, 0))
            else:
                if not timeout:
                    self.window_surface.blit(elf_img, cell_rect)
                if robot_type:
                    textbox.setText(robot=robot_prompt, system=system_prompt)
                    self.window_surface.blit(self.elf_images[0], (256 + map_size, 0))
                elif human_direction != None:
                    textbox.setText(
                        robot="Your last choice was {}. I followed your choice.".format(human_action_name))
                    self.window_surface.blit(self.elf_images[0], (256 + map_size, 0))
                else:
                    textbox.setText()

        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
            round_num=0
    ):
        '''
        Reset the interface for a new round of game
        :param seed: (type:int) random seed
        :param round_num: (type: int) the next round number after resetting (for display purposes)
        :return: the world state after resetting the game
        '''
        super().reset()

        self.render(round_num, None, None, self.world_state)
        return self.world_state

    def close(self):
        if self.window_surface is not None:
            pygame.display.quit()
            pygame.quit()


class InverseFrozenLakeEnv(FrozenLakeEnvInterface):
    """
    Change the reward for the Adversarial Bayes-POMCP Agent
    """

    def reward(self, augmented_state, robot_action, human_action=None):
        """
        Return the reward if using an inverse reward metrics.
        Used for the Adversarial Bayes-POMCP Agent
        :param augmented_state: Augmented world state
        :param human_action: The action the human user chooses
        :param robot_action: The action the robot agent chooses
        :return: the inverse round reward
        """
        position, last_position, human_slippery, robot_slippery = augmented_state[:4]
        # Get reward based on the optimality of the human action and the turn number
        # TODO: add penalty if robot takes control etc.
        curr_row = position // self.ncol
        curr_col = position % self.ncol
        last_row = last_position // self.ncol
        last_col = last_position % self.ncol
        reward = 1
        detect = None
        if human_action:
            human_accept, detect, human_choice = human_action
        if detect == 1:
            reward = 2
        if self.desc[curr_row, curr_col] in b'HS' or \
                (self.desc[last_row, last_col] in b'HS' and position == 0 and self.move(last_position,
                                                                                        robot_action[1]) != 0) or \
                (self.desc[last_row, last_col] in b'HS' and robot_action[0] == 0):
            reward = 10
        elif self.desc[curr_row, curr_col] in b'G':
            reward = -30
        return reward
