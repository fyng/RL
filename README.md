# Usage
Install micromamba or mamba as the package manager. To install micromamba, refer to the [installation guide](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)

To install the classifier:
1. Clone the repo
```
git clone https://github.com/fyng/RL.git
cd RL
```

2. Create virtual environment
```
micromamba env create -f environment.yml
```
```
micromamba activate RL
```

3. Create directory `plots` for plots


4. Run model
- Question 1: Run `value_iter.py` to generate Q and V. Run `value_iter_demo.py` to visualize the optimal path through the maze.
- Question 2: Run `qlearning_maze.py` to generate plots comparing Q-learning against Q* found through value iteration in question 1.
- Question 3: Run `gym.py` for the REINFORCE algorithm. Uncomment the appropriate lines for Acrobat-V1 and Mountaincar-V0, and to select the models.

# Acknowledgement
ECE 5242 - Intelligent Autonomous Systems taught by Prof Daniel Lee & Travers Rhodes 