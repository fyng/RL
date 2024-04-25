import numpy as np
from maze import *

env = Maze()
init_state = env.reset()
state = init_state

# set to no slip
env.slip = 0

Q_val = np.load('Qval.npy')
policy = np.load('policy.npy')

done = False
while not done:
    action = np.argmax(Q_val[state])
    # action = policy[state]
    reward, state, done = env.step(state, action)
    print('value: ', Q_val[state], np.argmax(Q_val[state]))
    env.plot(state, action)
