from evaluation import *
import matplotlib.pyplot as plt
from maze import *
from tqdm import tqdm
import gymnasium as gym

STATE_RES = 10 # 10 points per state dimension
lr = 0.1 
epsilon = 0.05
max_iter = 5000
discount = 0.99
eval_step = []
eval_reward = []
eval_error = []

# https://gymnasium.farama.org/environments/classic_control/acrobot/
env = gym.make('Acrobot-v1')
name = env.unwrapped.spec.id
env.action_space.seed(123) # set seed for reproducibility

obs_space_dims = env.observation_space.shape[0]
action_space_dims = env.action_space.n
low = env.observation_space.low
high = env.observation_space.high

def rmse(predictions, targets):
    return np.sqrt(np.mean((predictions-targets)**2))

def get_action_egreedy(values ,epsilon):
	# Implement epsilon greedy action policy
	actions = values.shape[0]
	if np.random.rand() < epsilon:
		return np.random.randint(actions)
	else:
		return np.argmax(values)

def state2idx(state, low, high, obs_space_dims):
    state = 2 * (state - low) / (high - low) - 1 # rescale
    state_digit = np.digitize(state, np.linspace(-1, 1, STATE_RES)) - 1 # digitize
    shape = tuple([STATE_RES]*obs_space_dims)
    idx = np.ravel_multi_index(state_digit, shape)

def evaluation(env, Q_table, step_bound = 100, num_itr = 10):
	"""
	Semi-greedy evaluation for discrete state and discrete action spaces and an episodic environment.

	Input:
		env : an environment object. 
		Q : A numpy array. Q values for all state and action pairs. 
			Q.shape = (the number of states, the number of actions)
		step_bound : the maximum number of steps for each iteration
		num_itr : the number of iterations

	Output:
		Total number of steps taken to finish an episode (averaged over num_itr trials)
		Cumulative reward in an episode (averaged over num_itr trials)

	"""
	total_step = 0 
	total_reward = 0 
	itr = 0 
	while(itr<num_itr):
		step = 0
		np.random.seed()
		state,_ = env.reset()
		reward = 0.0
		done = False
		while((not done) and (step < step_bound)):
			state_idx = state2idx(state, low, high, obs_space_dims)
			action = get_action_egreedy(Q_table[state_idx], 0.05)
			r, state_n, done, _ = env.step(action)
			state = state_n
			reward += r
			step +=1
		total_reward += reward
		total_step += step
		itr += 1
	return total_step/float(num_itr), total_reward/float(num_itr)

Q_table = np.zeros((obs_space_dims * STATE_RES, action_space_dims))

for i in tqdm(range(max_iter)):  
    for state_idx in range(obs_space_dims * STATE_RES):
        # state_idx = state2idx(state, low, high, obs_space_dims)
        for action in range(action_space_dims):
            reward, next_state, done, _ = env.step(state, action)
            next_state_idx = state2idx(next_state, low, high, obs_space_dims)
            # epsilon greedy
            if np.random.rand() < epsilon:
                next_action = np.random.randint(env.anum)
            else:
                next_action = np.argmax(Q_table[next_state_idx])

            # update Q table
            Q_table[state_idx, action] = (1 - lr) * Q_table[state_idx, action] + lr * (reward + discount * np.max(Q_table[next_state_idx, next_action]))

    if (i+1) % 50 == 0:
        avg_step, avg_reward = evaluation(Maze(), Q_table)
        eval_step.append(avg_step)
        eval_reward.append(avg_reward)
    # error = rmse(Q_table, Q_optim)
    # eval_error.append(error)

print(len(eval_step), len(eval_reward), len(eval_error))

f1, axes = plt.subplots(1, 3, figsize=(12, 5))
axes[0].plot(range(0, max_iter, 50), eval_step)
axes[0].set_title('Average steps to goal in an episode')

axes[1].plot(range(0, max_iter, 50), eval_reward)
axes[1].set_title('Average reward in an episode')

# axes[2].plot(eval_error)
# axes[2].set_title('RMSE between Q values')
plt.tight_layout()
plt.savefig(f'plots/Q_{name}.png')