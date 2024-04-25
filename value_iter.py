import numpy as np
from maze import *

# initialize maze
env = Maze()
init_state = env.reset()
state = init_state

discount = 0.9

# turn off slip and calculate expectation instead
slip = env.slip
env.slip = 0

value = np.zeros(env.snum)
goal = env.cell2idx[env.goal_pos]*8 + 7

# value iteration
delta = 1
while delta > 1e-8:
    delta = 0
    for state in range(env.snum):
        v_prev = value[state]

        v_new = 0
        for action in range(env.anum):
            # don't slip
            reward, next_state, done = env.step(state, action)
            val = discount * value[next_state] + reward

            # slip
            reward_slip, next_state_slip, done_slip = env.step(state, ACTMAP[action])
            val_slip = discount * value[next_state_slip] + reward_slip

            v_a = (1 - slip) * val + slip * val_slip
            v_new = max(v_new, v_a)
        
        value[state] = v_new
        delta = max(delta, abs(v_prev - value[state]))

# output optimal policy
policy = np.zeros(env.snum)
Q_val = np.zeros((env.snum, env.anum))
for state in range(env.snum):
    v = np.zeros(env.anum)
    for action in range(env.anum):
        reward, next_state, done = env.step(state, action)
        reward_slip, next_state_slip, done_slip = env.step(state, ACTMAP[action])

        val = discount * value[next_state] + reward
        val_slip = discount * value[next_state_slip] + reward_slip

        v[action] = (1 - slip) * val + slip * val_slip
        Q_val[state, action] = (1 - slip) * val + slip * val_slip

    policy[state] = np.argmax(v)

policy = np.argmax(Q_val, axis = 1)


np.save('value.npy', value)
np.save('Qval.npy', Q_val)
np.save('policy.npy', policy)



