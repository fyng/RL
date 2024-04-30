from evaluation import *
import matplotlib.pyplot as plt
from maze import *
from tqdm import tqdm

def rmse(predictions, targets):
    return np.sqrt(np.mean((predictions-targets)**2))


lr = 0.6 
epsilon = 0.05
max_iter = 5000
discount = 0.9
eval_step = []
eval_reward = []
eval_error = []

env = Maze()
env.reset()

Q_table = np.zeros((env.snum, env.anum))
Q_optim = np.load('Qval.npy')

for i in tqdm(range(max_iter)):  
    for state in range(env.snum):
        for action in range(env.anum):
            reward, next_state, done = env.step(state, action)

            # epsilon greedy
            if np.random.rand() < epsilon:
                next_action = np.random.randint(env.anum)
            else:
                next_action = np.argmax(Q_table[next_state])

            # update Q table
            Q_table[state, action] = (1 - lr) * Q_table[state, action] + lr * (reward + discount * np.max(Q_table[next_state, next_action]))

    if (i+1) % 50 == 0:
        avg_step, avg_reward = evaluation(Maze(), Q_table)
        eval_step.append(avg_step)
        eval_reward.append(avg_reward)
    error = rmse(Q_table, Q_optim)
    eval_error.append(error)

print(len(eval_step), len(eval_reward), len(eval_error))

f1, axes = plt.subplots(1, 3, figsize=(12, 5))
axes[0].plot(range(0, max_iter, 50), eval_step)
axes[0].set_title('Average steps to goal in an episode')

axes[1].plot(range(0, max_iter, 50), eval_reward)
axes[1].set_title('Average reward in an episode')

axes[2].plot(eval_error)
axes[2].set_title('RMSE between Q values')
plt.tight_layout()
plt.savefig(f'plots/Q_lr{lr}_ep{epsilon}.png')