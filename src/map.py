import random
from copy import deepcopy

CHEESE_CELL = 2
BLOCK_CELL = 1
FREE_CELL = 0

NORTH = (-1, 0)
WEST = (0, -1)
SOUTH = (1, 0)
EAST = (0, 1)
STAY = (0, 0)

ACTIONS = [NORTH, SOUTH, EAST, WEST, STAY]
# ACTIONS = [NORTH, SOUTH, EAST, WEST]


class Map(object):
    def __init__(self, file_config=None):
        if file_config is not None:
            with open(file_config, 'r') as config_map:
                self.height, self.width = config_map.readline().strip().split()
                self.height = int(self.height)
                self.width = int(self.width)

                self.position_obstacles = []
                self.position_cheeses = []
                self.maze = []
                for i in range(self.height):
                    line = config_map.readline().strip().split()
                    self.maze.append([])
                    for j in range(self.width):
                        self.maze[i].append(int(line[j]))

                        if int(line[j]) == BLOCK_CELL:
                            self.position_obstacles.append([i, j])
                        if int(line[j]) == CHEESE_CELL:
                            self.position_cheeses.append([i, j])

                self.A = int(config_map.readline().strip())

                pos_x_mouse, pos_y_mouse = config_map.readline().strip().split()
                self.position_mouse = [int(pos_x_mouse), int(pos_y_mouse)]

                pos_x_cat, pos_y_cat = config_map.readline().strip().split()
                self.position_cat = [int(pos_x_cat), int(pos_y_cat)]

        else:
            self.__generate_random_map()

        self.back_up_position_cheeses = deepcopy(self.position_cheeses)
        self.back_up_position_mouse = deepcopy(self.position_mouse)
        self.back_up_position_cat = deepcopy(self.position_cat)
        self.back_up_maze = deepcopy(self.maze)

        self.reward_moving = -0.1 * len(self.position_cheeses)
        self.reward_pick_up_cheese = 10.0
        self.reward_win = 10.0 * (len(self.position_cheeses) + 1)
        self.reward_lose = -10.0 * len(self.position_cheeses)

        # self.reward_moving = -0.01
        # self.reward_pick_up_cheese = 0.0
        # self.reward_win = 10.0
        # self.reward_lose = -10.0
        # self.reward_win = 10.0 * len(self.back_up_position_cheeses)
        # self.reward_lose = -10.0 * len(self.position_cheeses)

    def __generate_random_map(self):
        width = random.randint(5, 10)
        height = random.randint(5, 10)
        maze = [['#' for _ in range(width)] for _ in range(height)]
        visited = [[False for _ in range(width)] for _ in range(height)]

        queue = [(height // 2, width // 2)]
        direction = ACTIONS

        visited[height // 2][width // 2] = True
        maze[height // 2][width // 2] = ' '

        while queue:
            current_cell = queue.pop(0)
            x, y = current_cell

            next_cells = []
            for d in direction:
                new_x = x + d[0]
                new_y = y + d[1]
                if 0 <= new_y < width and 0 <= new_x < height and not visited[new_x][new_y]:
                    visited[new_x][new_y] = True
                    next_cells.append((new_x, new_y))
                else:
                    continue

            random.shuffle(next_cells)
            if next_cells:
                if random.randint(0, 100) > 50:
                    choice_cell = random.choices(next_cells, k=max(0, len(next_cells) - 1))
                else:
                    choice_cell = next_cells

                for new_x, new_y in choice_cell:
                    maze[new_x][new_y] = ' '
                    queue.append((new_x, new_y))

        position_obstacles = []
        free_cells = []
        for i in range(height):
            for j in range(width):
                if maze[i][j] == '#':
                    position_obstacles.append([i, j])
                else:
                    free_cells.append([i, j])

        position_cheeses = random.choices(free_cells, k=random.randint(1, len(free_cells) // 8))

        position_mouse = None
        position_cat = None
        max_distance = -1
        free_cells = [x for x in free_cells if x not in position_cheeses]

        for cell_mouse in free_cells:
            for cell_cat in free_cells:
                if abs(cell_cat[0] - cell_mouse[0]) + abs(cell_cat[1] - cell_mouse[1]) > max_distance:
                    max_distance = abs(cell_cat[0] - cell_mouse[0]) + abs(cell_cat[1] - cell_mouse[1])
                    position_cat = cell_cat
                    position_mouse = cell_mouse

        self.width = width
        self.height = height
        self.position_obstacles = position_obstacles
        self.position_mouse = position_mouse
        self.position_cat = position_cat
        self.position_cheeses = position_cheeses
        self.maze = [[int([i, j] in self.position_obstacles) + 2 * int([i, j] in self.position_cheeses)
                      for j in range(self.width)] for i in range(self.height)]

        self.A = min(self.width, self.height)

    def print_map(self, file_name):

        with open(file_name, 'wt') as output_file:
            output_file.writelines(str(self.height) + " " + str(self.width) + "\n")

            for line_maze in self.maze:
                for i in line_maze:
                    output_file.writelines(str(i) + " ")
                output_file.write("\n")

            output_file.writelines(str(self.A) + "\n")
            output_file.writelines(str(self.position_mouse[0]) + " " + str(self.position_mouse[1]) + "\n")
            output_file.writelines(str(self.position_cat[0]) + " " + str(self.position_cat[1]) + "\n")

    # functii pentru Q-Learning, SARSA
    def get_init_state(self) -> tuple:
        return tuple(self.position_mouse)

    def reset(self):
        self.position_cheeses = deepcopy(self.back_up_position_cheeses)
        self.position_cat = deepcopy(self.back_up_position_cat)
        self.position_mouse = deepcopy(self.back_up_position_mouse)
        self.maze = deepcopy(self.back_up_maze)

    def get_available_actions(self, state, funct=None) -> list:
        available_actions = []

        for action in ACTIONS:
            new_x = state[0] + action[0]
            new_y = state[1] + action[1]

            if funct is None:
                if 0 <= new_y < self.width and 0 <= new_x < self.height and self.maze[new_x][new_y] != BLOCK_CELL:
                    available_actions.append(action)
            else:

                # print(new_x, new_y)

                if funct(new_x, new_y):
                    available_actions.append(action)

        return available_actions

    def get_available_actions_mouse(self):
        funct = lambda x, y: 0 <= y < self.width and 0 <= x < self.height and self.maze[x][y] != BLOCK_CELL and \
                             [x, y] != self.position_cat

        return self.get_available_actions(self.position_mouse, funct)

    def __update_cat_position(self):
        distance = abs(self.position_mouse[0] - self.position_cat[0]) + \
                   abs(self.position_mouse[1] - self.position_cat[1])

        if distance > self.A:
            # mutarea ramndom
            action = random.choice(self.get_available_actions(self.position_cat))
            self.position_cat = [self.position_cat[0] + action[0], self.position_cat[1] + action[1]]

        else:
            # mutarea folosind bfs
            target = deepcopy(self.position_mouse)
            start = deepcopy(self.position_cat)

            queue = [target]
            visited = [[False for _ in range(self.width)] for _ in range(self.height)]
            visited[target[0]][target[1]] = True

            while queue:
                current_cell = queue.pop(0)

                next_actions = self.get_available_actions(current_cell)
                random.shuffle(next_actions)

                for action in next_actions:
                    next_cell = [current_cell[0] + action[0], current_cell[1] + action[1]]

                    if next_cell == start:
                        self.position_cat = deepcopy(current_cell)
                        return

                    if not visited[next_cell[0]][next_cell[1]]:
                        queue.append(next_cell)
                        visited[next_cell[0]][next_cell[1]] = True

            pass

    def apply_action(self, action):

        new_state = [self.position_mouse[0] + action[0], self.position_mouse[1] + action[1]]
        reward = self.reward_moving
        self.position_mouse = deepcopy(new_state)

        self.__update_cat_position()

        # if new_state in self.back_up_position_cheeses:
        #     reward = -10.0

        if self.position_cat == self.position_mouse:
            reward = self.reward_lose
        elif self.maze[self.position_mouse[0]][self.position_mouse[1]] == CHEESE_CELL:
            self.position_cheeses.remove(self.position_mouse)
            self.maze[self.position_mouse[0]][self.position_mouse[1]] = FREE_CELL

            if len(self.position_cheeses) == 0:
                reward = self.reward_win
            else:
                reward = self.reward_pick_up_cheese

        return tuple(new_state), reward

    def is_final(self, score) -> bool:
        return len(self.position_cheeses) == 0 or self.position_cat == self.position_mouse \
               or score < 2 * self.reward_lose

    def win(self) -> bool:
        return len(self.position_cheeses) == 0

    def draw(self, screen, image_cat, image_mouse, image_cheese, image_wall, width_cell, height_cell):
        screen.blit(image_cat, (self.position_cat[1] * width_cell, self.position_cat[0] * height_cell))
        screen.blit(image_mouse, (self.position_mouse[1] * width_cell, self.position_mouse[0] * height_cell))

        for position_cheese in self.position_cheeses:
            screen.blit(image_cheese, (position_cheese[1] * width_cell, position_cheese[0] * height_cell))

        for position_obstacle in self.position_obstacles:
            screen.blit(image_wall, (position_obstacle[1] * width_cell, position_obstacle[0] * height_cell))
