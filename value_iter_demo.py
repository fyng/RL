import numpy as np
from maze import *

env = Maze()
init_state = env.reset()
state = init_state

Q_val = np.load('Qval.npy')

done = False
while not done:
    # FIXME: 
    action = np.argmax(Q_val[state])
    reward, state, done = env.step(state, action)
    env.plot(state, action)
