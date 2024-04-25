import numpy as np
from maze import *

# initialize maze
env = Maze()
init_state = env.reset()
state = init_state

# initialize reward
Q_val = np.zeros((env.snum, env.anum))

discount = 0.9

max_iter = 1000

opt_state = env.cell2idx[env.goal_pos]*8 + 7
print(opt_state)

for i in range(max_iter):
    Q_val_init = Q_val.copy()
    for state in range(env.snum):
        if state == opt_state:
            # don't update the value of the optimal state
            # can remove this if presetting optimal state to 3 and taking the max
            Q_val[state] = 0
            continue
        for action in range(env.anum):
            reward, next_state, done = env.step(state, action)
            if done:
                # if done, update Q value to the reward
                Q_val[state, action] = max(discount * reward, Q_val[state, action])
            else:
                # if not, discount the value of the next state
                next_val = np.max(Q_val[next_state])
                Q_val[state, action] = max(discount * next_val, Q_val[state, action])

    # if np.allclose(Q_val, Q_val_init):
    #     print('Converged at iteration', i+1)
    #     break

np.save('Qval.npy', Q_val)

V = np.max(Q_val, axis=1)
print(V.shape)
print(V[0])

np.save('value.npy', V)


