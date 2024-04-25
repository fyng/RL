from evaluation import *
import matplotlib.pyplot as plt
from maze import *

# TODO: intialize
lr = 0.1 # update to Q table
epsilon = 0.1 # chance of taking a random action

eval_step, eval_reward = [] , []
learning = True
while learning:
    # TODO: implement Q learning

    avg_step, avg_reward = evaluation(Maze(), current_Q_table)
    eval_step.append(avg_step)
    eval_reward.append(avg_reward)

f1, ax1 = plt.subplots()
ax1.plot(np.arange(0, 5000, 50), eval_step)

f2, ax2 = plt.subplots()
ax2.plot(np.arange(0, 5000, 50), eval_reward)