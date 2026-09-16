# Exercise 3.16 Eight-State Routing Problem for Q-Learning
import numpy as np
import pylab as plt
from Q_Utils import *


points_list = [(0, 1), (1, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7), (3, 6)]
goal = 7
showgraph(points_list)
MATRIX_SIZE = 8
R = createRmat(MATRIX_SIZE, points_list, goal)
Q = np.matrix(np.zeros([MATRIX_SIZE, MATRIX_SIZE]))
gamma = 0.8

scores = []
for _ in range(700):
    current_state = np.random.randint(0, int(Q.shape[0]))
    available_act = available_actions(R, current_state)
    action = sample_next_action(available_act)
    score = update(R, Q, current_state, action, gamma)
    scores.append(score)

print('Trained Q matrix:')
print(Q / np.max(Q) * 100)

current_state = 0
steps = [current_state]
while current_state != goal:
    next_step_index = np.where(Q[current_state,] == np.max(Q[current_state,]))[1]
    if next_step_index.shape[0] > 1:
        next_step_index = int(np.random.choice(next_step_index, size=1))
    else:
        next_step_index = int(next_step_index)
    steps.append(next_step_index)
    current_state = next_step_index

print('Most efficient path:')
print(steps)
plt.plot(scores)
plt.show()
