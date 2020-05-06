from src.map import *
from src.q_learning import max_first
from src.display import Display


class Game:
    @staticmethod
    def run_simulation(game_map: Map, Q: dict, epsilon_greedy=max_first):
        display = Display(game_map)
        game_map.reset()
        state = deepcopy(game_map.get_init_state())
        run = True
        score = 0

        display.render()
        while run:
            action = epsilon_greedy(Q, state, game_map.get_available_actions(state), 0.3, "softmax")
            state, _ = game_map.apply_action(action)
            run = not Display.is_close() and not game_map.is_final(score)
            display.render()

        Display.close_display()

        if game_map.win():
            print("Game Win!!!")
        else:
            print("Game Lose!!!")
