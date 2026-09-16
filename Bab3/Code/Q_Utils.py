# Example 3.22 - Q_Utils.py
# Modified from:
# https://amunategui.github.io/reinforcement-learning/index.html
# http://firsttimeprogrammer.blogspot.com/2016/09/getting-ai-smarter-with-q-learning.html
# http://mnemstudio.org/path-finding-q-learning-tutorial.htm
import numpy as np
import pylab as plt
import networkx as nx


def showgraph(points_list):
    G = nx.Graph()
    G.add_edges_from(points_list)
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos)
    plt.show()


def createRmat(MATRIX_SIZE, points_list, goal):
    R = np.matrix(np.ones(shape=(MATRIX_SIZE, MATRIX_SIZE)))
    R *= -1
    for point in points_list:
        print(point)
        if point[1] == goal:
            R[point] = 100
        else:
            R[point] = 0
        if point[0] == goal:
            R[point[::-1]] = 100
        else:
            R[point[::-1]] = 0
    R[goal, goal] = 100
    print(R)
    return R


def available_actions(R, state):
    current_state_row = R[state,]
    av_act = np.where(current_state_row >= 0)[1]
    return av_act


def sample_next_action(available_act):
    next_action = int(np.random.choice(available_act, 1))
    return next_action


def update(R, Q, current_state, action, gamma):
    max_index = np.where(Q[action,] == np.max(Q[action,]))[1]
    if max_index.shape[0] > 1:
        max_index = int(np.random.choice(max_index, size=1))
    else:
        max_index = int(max_index)
    max_value = Q[action, max_index]
    Q[current_state, action] = R[current_state, action] + gamma * max_value
    print('max_value', R[current_state, action] + gamma * max_value)
    if np.max(Q) > 0:
        return np.sum(Q / np.max(Q) * 100)
    return 0
