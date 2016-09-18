import matplotlib.pyplot as plt
import numpy as np

class Bandit:

    def __init__(self, arms):
        self.arm_mean = []
        for _ in range(arms):
            self.arm_mean.append(np.random.normal(10.0, 1.0))

    def num_arms(self):
        return len(self.arm_mean)

    def sample_arm(self, n):
        return np.random.normal(self.arm_mean[n], 1.0)

    def highest_mean(self):
        return max(self.arm_mean)

    def lowest_mean(self):
        return min(self.arm_mean)

class ActionHistory:

    def __init__(self):
        self.count = 0
        self.mean_reward = 0.0

    def sample(self, value):
        self.count += 1
        self.mean_reward += (value - self.mean_reward)/self.count

class Agent:

    def __init__(self, epsilon, bandit_arms):
        self.epsilon = epsilon
        self.arm_action_history = [ActionHistory() for _ in range(bandit_arms)]
        self.combined_action_history = ActionHistory()

    def sampleBandit(self, bandit):
        arm_n = self.egreedyArm(bandit)
        sample = bandit.sample_arm(arm_n)
        self.arm_action_history[arm_n].sample(sample)
        self.combined_action_history.sample(sample)

    def egreedyArm(self, bandit):
        if np.random.random() < self.epsilon:
            # Explore arm at random
            return np.random.randint(0, bandit.num_arms())
        else:
            # Exploit best known arm at the moment
            return self.bestArm()

    def bestArm(self):
        max_val = max([a.mean_reward for a in self.arm_action_history])
        best_arm = self.arm_action_history.index(next(a for a in self.arm_action_history if a.mean_reward == max_val))
        return best_arm


NUM_ARMS = 100
NUM_SAMPLES = 1000

bandit = Bandit(NUM_ARMS)
print("Best arm mean=",bandit.highest_mean())
print("Lowest arm mean=",bandit.lowest_mean())

epsilons = [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.9, 1.0]
agents = []
agents_plot = [[] for _ in range(len(epsilons))]

for e in epsilons:
    agents.append(Agent(e, NUM_ARMS))

for n in range(NUM_SAMPLES):
    # Sample
    for n in range(len(agents)):
        agent = agents[n]
        agent.sampleBandit(bandit)
        # Add to plot
        agents_plot[n].append(agent.combined_action_history.mean_reward)

# Plot stuff
x = list(range(NUM_SAMPLES))
for n in range(len(agents)):
    plt.plot(x, agents_plot[n], linewidth=2, label=str(epsilons[n]))

plt.legend()
plt.show()
