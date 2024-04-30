# Introduction
In this project, I implemented reinforcement learning methods including value iteration, Q-learning and policy gradient on a variety of environments. Reproducible code is available as a [git repository](https://github.com/fyng/SLAM)

# Maze
## Problem Statement
Given an maze, determine the optimal policy (next move given current location in the maze) to 1) maximize the number of flags captured while 2) reaching the end as quickly as possible

The maze is a 4x5 grid, where a square can be empty contains obstables, or flags. There is a designated starting and goal square. The agent can take a step at every timepoint (UP, DOWN, LEFT, RIGHT). There is a 0.1 probability that the agent slips and take the next counterclockwise action (e.g. if slipped while going UP, go RIGHT instead). When the agent encounters an obstacle, it bounces back to its current square. 

## Approach 1: Value Iteration
Starting from a state $s$, the optimal value of the state $s$ can be calculated deterministically as a function of the optimal value of the next state $s'$ and the action taken $a$, using the Bellman Equation. 
$$V(s) = \max_a \sum_{s', r} p(s',r|s,a)[\gamma V(s') + R(s,a)]$$

The challenge is that the optimal value of the next state $V(s')$ is unknown. Luckily and perhaps surprisingly, we can randomly initialize $V(s) \forall s \in S$, and iteratively update each $V(s)$. Given sufficient iterations, this is guaranteed to converge to the optimal value function $V^{*}(s)$.

Once the optimal value function $V^{*}(s)$ is found, the optimal Q-function $Q^{*}(s,a)$ can be found by exploring all possible action starting from state $s: a \in A(s)$ and calculating the optimal value for each state-action pair $(s,a)$. This can be achieved by performing a single forward pass over all states, and the permissable action from each state. 

## Results
Taking a discount factor of $\gamma = 0.9$, the optimal policy will produce the following trajectory:
START -> DOWN -> RIGHT -> RIGHT -> UP -> DOWN -> RIGHT -> DOWN -> DOWN -> UP -> UP -> RIGHT -> UP

The Q-function that produced this set of trajectory is in `Qval.npy`.

> The effect of discount factor $\gamma$:
> We choose the hyperparameter $\gamma \in (0,1)$, which is a discount factor on future value in order to force finite trajectories. In the problem statement we used $\gamma=0.9$, and the corresponding trajectory captured *2 flags*
> It is important to note that this is result is not agnostic to parameter choice. E.g. choosing a less aggressive discount $\gamma = 0.99$ will yield a trajectory that captures *all 3 flags*. This is an intuitive result since the future reward is valued higher. 


## Approach 2: Q-learning
Instead of iteratively updating the value function $V(s)$, Q-learning updates the Q-function $Q(s,a)$. 

## Results: 
The model maintains a $55m \times 55m$ occupancy grid map at $5cm$ resolution. The value of a grid is its log-probability of being occupied by an obstacle. At each time point, the mounted LIDAR scans the environment and the light ray passes through unoccupied empty space until it hits the first obstacle along its path and bounces back. Each LIDAR beam is mapped to (1) a single occupied grid coordinates, and (2) a series of empty grid coordinates via `getMapCellsFromRay_fclad()`. 

For each time point, a contact updates the log-probability of a grid coordinate by +1, while a passthrough updates the log-probability of a grid coordinate by -0.1. Therefore, a LIDAR beam will need to pass through a coordinate for 10 subsequent timepoints to clear a obstacle registration. The absolute log-probability of each grid is capped at 15, which prevents the map from becoming too confident.

>**NOTE ON LIDAR DATA**:
 Each LIDAR beam consists of $n$ (angle, distance) tuples undergoes preprocessing before being used in mapping. First, LIDAR beams with distance < 0.1 meters are discarded as this might be reflecting off the robot itself. The maximum spec-ed range of the LIDAR is 30 meters, so any beams with distance > 30 meters is also removed. Lastly, the (angle, distance) tuples are translated from the robot frame to the world frame. 


## 3. SLAM
The intuition of SLAM is that adding noise to the position and orientation of particles can correct for rolling with slipping or non-optimal parameter choice for the robot. In my implementation, 200 particles are seeded. Noise is added to the particles every $n$ iterations. Noise is only added to $\theta$, which will effectively add noise to $x, y$ as the particles independently evolve. In particular, I chose to scale the noise as a function of the current $d\theta$. This helps correct for unoptimal robot width - if the robot is not turning, there is no $\theta$ error, but the $\theta$ error increases when the turn is sharp. The scaling factor is parameterized by `theta_scale`

Each particle has a weight, which is updated at each time step by the correspondance between the particle's map and the current best map. The correspondence is measured by the log-probability of the pixel that is detected as occupied. 
$$corr = \sigma(\sum_{i \in{\text{lidar obstacles}}} \text{occupancy grid}_i)$$
The better the correspondance, the higher the sum. The value is passed through a sigmoid to ensure non-negativity, and multiplied to the weight of the previous timepoint before re-normalizing. 
$$w_{t+1} = w_t \times corr$$

The 'exploration' of the particles are determined by the reseeding interval. The intuition is that the LIDAR of the robot only covers the forward cone, so sufficient time needs to occur between reseeding to allow the robot to turn such that it will encounter previous obstacles in its field of view. However, if the reseeding interval is too long, the robot will completely deviate off course and loss its dead_reckoning. I experimented with constant reseeding intervals and found it best to dynamically set the reseeding interval by checking the number of effective particles. Resampling when $n_effective < 0.8$ works well.
$$\text{n\_{effective}} = \frac{\sum w^2}{(\sum w)^2}$$

During reseeding, the particles with higher overall correspondence have a higher probability of being selected. The probability is determined by the weight of the particle. The particles are sampled *with replacement* using the weights until 200 particles are reselected. After reseeding, all particle weights will be reset to be uniform

# Results
For the naive algorithm (odometry only), I used 730 mm as the width, which produced the most reasonable path traces. The SLAM algorithm further refined the alignment of wall features and reduced the "smearing effect" of wheels slipping during turns.

The naive algorithms maps poorly at width = 500 mm, but using the SLAM algorithm recovers the geometry of the mapping, although the overall rotation of the map with respect to the grid suffers.

### Maze: optimal path via value iteration


### Maze: Q-learning 


# Appendix:
## Value Iteration over Maze
value:  [4.88757666 5.35070957 5.29191057 5.94523286] 3
