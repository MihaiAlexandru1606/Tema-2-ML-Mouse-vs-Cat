import sys
import random
from math import sqrt, log
import operator
import numpy as np
from scipy.special import softmax
from copy import deepcopy
from statistics import mean

from src.display import Display

Q_VALUE = 0
NUMBER_VISITED = 1


def uct_strategy(Q, state, actions):
    actions_info = []
    reward_all = 0
    number_visited = 0

    random.shuffle(actions, random=np.random.random)
    for action in actions:
        if (state, action) in Q:
            actions_info.append((action, Q[(state, action)][Q_VALUE], Q[(state, action)][NUMBER_VISITED]))
            reward_all += Q[(state, action)][Q_VALUE]
            number_visited += Q[(state, action)][NUMBER_VISITED]
        else:
            actions_info.append((action, 0, 1))
    if reward_all == 0:
        reward_all = 1

    return max([(info[0], info[1] + 1 / sqrt(2) * sqrt(2 * log(number_visited)) / info[2]) for info in
                actions_info], key=operator.itemgetter(1))[0]


def softmax_strategy(Q, state, actions):
    q_list = []
    for action in actions:
        if (state, action) in Q:
            q_list.append(Q[(state, action)][Q_VALUE])
        else:
            q_list.append(0)

    q_list = np.array(q_list)
    prob = softmax(q_list)

    return actions[np.random.choice(range(len(actions)), p=prob)]


def max_first(Q, state, actions, _=None, __=None):
    random.shuffle(actions)

    current_max = -sys.maxsize
    return_actions = []

    for action in actions:
        utility = Q.get((state, action), (0, 0))[Q_VALUE]

        if current_max < utility:
            current_max = utility
            return_actions = [action]
        elif current_max == utility:
            return_actions.append(action)

    return random.choice(return_actions)


def random_choice_action(_, __, actions, ___=None, ____=None):
    random.shuffle(actions)

    return random.choice(actions)


"""
Bine cred ca mai de graba este o exploreare
"""


def explore(Q, state, actions, _=None, strategy="default"):
    possible_action = []

    for action in actions:
        if (state, action) not in Q:
            possible_action.append(action)

    if possible_action:
        random.shuffle(possible_action)
        return random.choice(possible_action)

    """ in functie de tipul de strategie ales, daca s-au visitat toti vecini"""
    if strategy == "default":
        min_visited = sys.maxsize
        for action in actions:
            if Q[(state, action)][NUMBER_VISITED] < min_visited:
                possible_action = [action]
            elif Q[(state, action)][NUMBER_VISITED] == min_visited:
                possible_action.append(action)

        random.shuffle(possible_action)
        return random.choice(possible_action)

    elif strategy == 'uct':
        return uct_strategy(Q, state, actions)
    elif strategy == 'softmax':
        return softmax_strategy(Q, state, actions)

    print("GGGGGGGGGGGGGGG")
    sys.exit(-1)


def explore_exploitation(Q, state, actions, epsilon, strategy="default"):
    if strategy == 'default':
        if np.random.uniform(0, 1) > epsilon:
            return max_first(Q, state, actions)
        else:
            return explore(Q, state, actions)
    elif strategy == 'softmax':
        return softmax_strategy(Q, state, actions)


def print_q(Q: dict, height, width, actions):
    from prettytable import PrettyTable

    x = PrettyTable()
    x.field_names = ["State"] + list(map(lambda tup: " ".join(map(str, tup)), actions))
    for i in range(height):
        for j in range(width):
            list_l = [" ".join([str(i), str(j)])]
            for action in actions:
                list_l.append(round(Q.get(((i, j), action), (0.0, 0.0))[0], 4))

            x.add_row(list_l)

    print(x)


def q_learning(game_map, epsilon_greedy, Q={}, learning_rate=0.1, discovering_factor=0.9, epsilon=0.3,
               training_episode=1000, evaluation_episode=10, strategy="default", verbose=False):
    train_scores = []
    eval_scores = []
    initial_state = game_map.get_init_state()
    win_game = 0
    q_values = []
    number_cell_visited = []

    if verbose:
        display = Display(game_map)

    for train_ep in range(1, training_episode + 1):
        score = 0
        state = deepcopy(initial_state)
        game_map.reset()

        if verbose:
            from src.map import ACTIONS
            import sys
            print_q(Q, game_map.height, game_map.width, ACTIONS)
            display.render()
            sys.stdin.readline()

            if display.is_close():
                sys.exit(0)

        while not game_map.is_final(score):

            actions = game_map.get_available_actions_mouse()
            action = epsilon_greedy(Q, state, actions, epsilon, strategy)

            # apply action and get the next state and the reward
            next_state, reward = game_map.apply_action(action)
            score += reward

            state_q_value, number_visited = Q.get((state, action), (0, 0))
            action_next_state = max_first(Q, next_state, game_map.get_available_actions(next_state))
            next_state_q_value, _ = Q.get((next_state, action_next_state), (0, 0))

            q_value = state_q_value + learning_rate * (reward + discovering_factor * next_state_q_value - state_q_value)
            Q[(state, action)] = (q_value, number_visited + 1)
            state = next_state

            if verbose:
                from src.map import ACTIONS
                import sys
                print_q(Q, game_map.height, game_map.width, ACTIONS)
                display.render()
                sys.stdin.readline()

                if display.is_close():
                    sys.exit(0)

        train_scores.append(score)
        win_game += int(game_map.win())
        q_values.append(sum([x[Q_VALUE] for x in Q.values()]))
        number_cell_visited.append(len(Q))

        print("Episode {}/{}, score : {} win : {}".format(train_ep, training_episode, score, game_map.win()))

        if train_ep % evaluation_episode == 0:
            avg_score = mean(train_scores[train_ep - evaluation_episode: train_ep])

            eval_scores.append(avg_score)

    win_game /= training_episode

    return Q, train_scores, eval_scores, win_game, q_values, number_cell_visited


def sarsa(game_map, epsilon_greedy, Q={}, learning_rate=0.1, discovering_factor=0.9, epsilon=0.3,
          training_episode=1000, evaluation_episode=10, strategy="default", verbose=False):
    train_scores = []
    eval_scores = []
    initial_state = game_map.get_init_state()
    win_game = 0
    q_values = []
    number_cell_visited = []

    if verbose:
        display = Display(game_map)

    for train_ep in range(1, training_episode + 1):
        game_map.reset()
        score = 0

        state = deepcopy(initial_state)
        actions = game_map.get_available_actions_mouse()
        action = epsilon_greedy(Q, state, actions, epsilon, strategy)

        if verbose:
            from src.map import ACTIONS
            import sys
            print_q(Q, game_map.height, game_map.width, ACTIONS)
            display.render()
            sys.stdin.readline()

            if display.is_close():
                sys.exit(0)

        while not game_map.is_final(score):
            # apply action and get the next state and the reward
            next_state, reward = game_map.apply_action(action)
            score += reward

            state_q_value, number_visited = Q.get((state, action), (0, 0))
            action_next_state = epsilon_greedy(Q, next_state, game_map.get_available_actions(next_state), epsilon,
                                               strategy)
            next_state_q_value, _ = Q.get((next_state, action_next_state), (0, 0))

            q_value = state_q_value + learning_rate * (reward + discovering_factor * next_state_q_value - state_q_value)

            Q[(state, action)] = (q_value, number_visited + 1)
            state = next_state
            action = action_next_state

            if verbose:
                from src.map import ACTIONS
                import sys
                print_q(Q, game_map.height, game_map.width, ACTIONS)
                display.render()
                sys.stdin.readline()

                if display.is_close():
                    sys.exit(0)

        q_values.append(sum([x[Q_VALUE] for x in Q.values()]))
        win_game += int(game_map.win())
        train_scores.append(score)
        number_cell_visited.append(len(Q.keys()))

        print("Episode {}/{}, score : {} win : {}".format(train_ep, training_episode, score, game_map.win()))

        if train_ep % evaluation_episode == 0:
            avg_score = mean(train_scores[train_ep - evaluation_episode: train_ep])

            eval_scores.append(avg_score)

    win_game /= training_episode

    return Q, train_scores, eval_scores, win_game, q_values, number_cell_visited
