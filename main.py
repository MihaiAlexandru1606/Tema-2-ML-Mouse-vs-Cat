from src.eval import *
from src.map import *
from src.display import *
from src.q_learning import *
from src.play import Game

if __name__ == '__main__':
    game_map = Map("config/mini_map.txt")

    Q, _, _, win, _, _ = q_learning(game_map, epsilon_greedy=max_first, strategy='default', epsilon=0.9,
                                    training_episode=10000, verbose=False)
    print(win)
    Game.run_simulation(game_map, Q, epsilon_greedy=explore_exploitation)
