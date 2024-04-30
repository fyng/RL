from evaluation import *
import matplotlib.pyplot as plt
from maze import *
from tqdm import tqdm
import gymnasium as gym

def rmse(predictions, targets):
    return np.sqrt(np.mean((predictions-targets)**2))

def get_action_egreedy(values ,epsilon):
	# Implement epsilon greedy action policy
	actions = values.shape[0]
	if np.random.rand() < epsilon:
		return np.random.randint(actions)
	else:
		return np.argmax(values)

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
		state = env.reset()
		reward = 0.0
		done = False
		while((not done) and (step < step_bound)):
			action = get_action_egreedy(Q_table[state], 0.05)
			r, state_n, done, _ = env.step(state,action)
			state = state_n
			reward += r
			step +=1
		total_reward += reward
		total_step += step
		itr += 1
	return total_step/float(num_itr), total_reward/float(num_itr)

def discretize_state(env, state):
    # FIXME!
    # state: [cos(theta1) sin(theta1) cos(theta2) sin(theta2) theta_dot1 theta_dot2]

    """
    Discretize the continuous state into discrete state.
    States are normalized to mean 1 and variance 0

    Input:
        env : an environment object
        state : a tuple of continuous state
            [cos(theta1) sin(theta1) cos(theta2) sin(theta2) theta_dot1 theta_dot2]
            

    Output:
        discrete_state : a tuple of discrete state
    """
    discrete_state = ()
    for i in range(len(state)):
        discrete_state += (int((state[i]-env.state_bounds[i][0])/env.state_interval[i]),)
    return discrete_state


lr = 0.6 
epsilon = 0.05
max_iter = 5000
discount = 0.9
eval_step = []
eval_reward = []
eval_error = []

# https://gymnasium.farama.org/environments/classic_control/acrobot/
env = gym.make('Acrobot-v1')
observation = env.reset()
env.action_space.seed(123) # set seed for reproducibility

Q_table = np.zeros((env.snum, env.anum))
Q_optim = np.load('Qval.npy')

for i in tqdm(range(max_iter)):  
    # TODO: implement Q learning

    for state in range(env.snum):
        for action in range(env.anum):
            reward, next_state, done, _ = env.step(state, action)

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