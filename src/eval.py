from matplotlib import pyplot as plt
import multiprocessing
import plotly.graph_objects as go
import pandas as pd

from src.q_learning import *
from src.map import Map


class Eval(object):
    @staticmethod
    def plot_score(list_maps):
        param = [
            ("random", random_choice_action, "default"),
            ("max first", max_first, 'default'),
            ("Explore : default", explore, 'default'), ("Explore : softmax", explore, 'softmax'),
            ("Explore : uct", explore, 'uct'),
            ('Explore-exploitation : default', explore_exploitation, 'default'),
            ('Explore-exploitation : softmax', explore_exploitation, 'softmax')
        ]

        for name_map, number_episodes in list_maps:
            game_map = Map(name_map)
            for p in param:
                _, train_scores, eval_scores, _, _, _ = q_learning(game_map=game_map, training_episode=number_episodes,
                                                                   epsilon_greedy=p[1], strategy=p[2])
                plt.clf()
                fig = plt.figure()
                plt.plot(
                    np.linspace(1, len(train_scores), len(train_scores)),
                    np.convolve(train_scores, [0.2, 0.2, 0.2, 0.2, 0.2], "same"),
                    linewidth=1.0, color="blue", label='Training'
                )
                plt.plot(
                    np.linspace(10, len(train_scores), len(eval_scores)),
                    eval_scores, linewidth=2.0, color="red", label='Evaluation'
                )
                plt.title("Score evolution: " + p[0] + " Map " + name_map)
                plt.ylabel("Average score")
                plt.xlabel("Episode")
                plt.legend()
                plt.savefig(
                    "plot/" + "Score_evolution" + p[0] + "_Map_" + name_map.split('/')[-1].split('.')[0] + ".png")

    @staticmethod
    def run_games(Q: dict, game_map: Map, number_run):
        win = 0
        for _ in range(number_run):
            game_map.reset()
            score = 0
            state = deepcopy(game_map.get_init_state())

            while not game_map.is_final(score):
                action = max_first(Q, state, game_map.get_available_actions(state))
                state, reward = game_map.apply_action(action)
                score += reward

            win += int(game_map.win())
        return win / number_run

    @staticmethod
    def test_hyperparameter(args):
        name_map, number_episodes, name_epsilon_greedy, epsilon_greedy, strategy = args
        learning_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        discovering_factors = [0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.5]
        game_map = Map(name_map)
        hearder_table = ["D.F. / L.R.", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7"]

        win_ant = []
        win_eval = []

        for discovering_factor in discovering_factors:
            w_ant = [discovering_factor]
            w_eval = [discovering_factor]

            for learning_rate in learning_rates:
                Q = {}
                Q, _, _, win_games, _, _ = q_learning(Q=Q, game_map=game_map, training_episode=number_episodes,
                                                      epsilon_greedy=epsilon_greedy, strategy=strategy,
                                                      learning_rate=learning_rate,
                                                      discovering_factor=discovering_factor)
                w_ant.append(round(win_games, 4))
                w_eval.append(round(Eval.run_games(Q, game_map, number_episodes), 2))

            win_ant.append(w_ant)
            win_eval.append(w_eval)

        df1 = pd.DataFrame(win_ant, columns=hearder_table)
        df1.to_excel("xlsx/Hyperparameters_{}_{}_Training.xlsx".format(name_epsilon_greedy, name_map.split('/')[-1].split('.')[0]))

        df2 = pd.DataFrame(win_eval, columns=hearder_table)
        df2.to_excel("xlsx/Hyperparameters_{}_{}_Evaluation.xlsx".format(name_epsilon_greedy, name_map.split('/')[-1].split('.')[0]))

    @staticmethod
    def compare_hyperparameter(list_maps):
        param = [
            ("random", random_choice_action, "default"),
            ("max_first", max_first, 'default'),
            ("explore_default", explore, 'default'), ("explore_softmax", explore, 'softmax'),
            ("explore_uct", explore, 'uct'),
            ('explore_exploitation_default', explore_exploitation, 'default'),
            ('explore_exploitation_softmax', explore_exploitation, 'softmax')
        ]
        input_thread = []

        for name_map, number_episodes in list_maps:
            for p in param:
                input_thread.append((name_map, number_episodes, p[0], p[1], p[2]))
        pool = multiprocessing.Pool()
        pool.map(Eval.test_hyperparameter, input_thread)

    @staticmethod
    def eval_score_max_first(list_maps):

        for name_map, number_episodes in list_maps:
            game_map = Map(name_map)
            _, _, _, _, q_values, _ = q_learning(game_map=game_map, epsilon_greedy=max_first,
                                                 training_episode=number_episodes)
            plt.clf()
            fig = plt.figure()
            plt.plot(range(1, len(q_values) + 1), q_values)
            plt.title("Max Fist : evolution q values Map : " + name_map.split('/')[-1].split('.')[0])
            plt.xlabel("Number episodes")
            plt.ylabel("Mean Q Values")
            plt.savefig('plot/Max_Fist_evolution_q_values_Map_' + name_map.split('/')[-1].split('.')[0] + '.png')

    @staticmethod
    def compare_max_first_random(list_maps):

        for name_map, number_episodes in list_maps:
            game_map = Map(name_map)
            _, _, _, _, q_values_max, number_visited_max = q_learning(game_map=game_map, epsilon_greedy=max_first,
                                                                      training_episode=number_episodes)
            _, _, _, _, q_values_random, number_visited_random = q_learning(game_map=game_map,
                                                                            epsilon_greedy=random_choice_action,
                                                                            training_episode=number_episodes)

            plt.clf()
            fig = plt.figure()
            plt.plot(range(1, len(q_values_max) + 1), q_values_max, color='yellow', label='Max Fist')
            plt.plot(range(1, len(q_values_random) + 1), q_values_random, color='blue', label="Random")
            plt.title("Evolution q values, Map : " + name_map.split('/')[-1].split('.')[0])
            plt.xlabel("Number episodes")
            plt.ylabel("Mean Q Values")
            plt.legend()
            plt.savefig('plot/Evolution_q_values_Map_' + name_map.split('/')[-1].split('.')[0] + '.png')

            plt.clf()
            fig = plt.figure()
            plt.plot(range(1, len(number_visited_max) + 1), number_visited_max, color='yellow', label='Max Fist')
            plt.plot(range(1, len(number_visited_random) + 1), number_visited_random, color='blue', label='Random')
            plt.title("Len Q, Map : " + name_map.split('/')[-1].split('.')[0])
            plt.xlabel("Number episodes")
            plt.ylabel("Len Q")
            plt.legend()
            plt.savefig('plot/Len_Q_Map_' + name_map.split('/')[-1].split('.')[0] + '.png')

    @staticmethod
    def compare_q_learning_sarsa(list_maps):
        param = [
            ("max first", max_first, 'default'),
            ("Explore : softmax", explore, 'softmax'),
            ('Explore-exploitation : softmax', explore_exploitation, 'softmax')
        ]

        for name_map, number_episodes in list_maps:
            game_map = Map(name_map)
            for p in param:
                _, _, _, _, q_values_q, number_q = q_learning(game_map=game_map, training_episode=number_episodes,
                                                              epsilon_greedy=p[1], strategy=p[2])

                _, _, _, _, q_values_sarsa, number_sarsa = sarsa(game_map=game_map,
                                                                 training_episode=number_episodes,
                                                                 epsilon_greedy=p[1], strategy=p[2])

                plt.clf()
                fig = plt.figure()
                plt.plot(range(1, len(q_values_q) + 1), q_values_q, color='green', label='Q-Learning')
                plt.plot(range(1, len(q_values_sarsa) + 1), q_values_sarsa, color='red', label="SARSA")
                plt.title("Evolution q values : " + p[0] + " Map : " + name_map.split('/')[-1].split('.')[0])
                plt.xlabel("Number episodes")
                plt.ylabel("Mean Q Values")
                plt.legend()
                plt.savefig('plot/Q_Learning_vs_SARSA_Value_' + p[0] + '_' + name_map.split('/')[-1].split('.')[0] + '.png')

                plt.clf()
                fig = plt.figure()
                plt.plot(range(1, len(number_q) + 1), number_q, color='yellow', label='Q-Learning')
                plt.plot(range(1, len(number_sarsa) + 1), number_sarsa, color='blue', label='SARSA')
                plt.title("Len Q " + p[0] + " Map : " + name_map.split('/')[-1].split('.')[0])
                plt.xlabel("Number episodes")
                plt.ylabel("Len Q")
                plt.legend()
                plt.savefig('plot/Q_Learning_vs_SARSA_Len_' + p[0] + '_' + name_map.split('/')[-1].split('.')[0] + '.png')
