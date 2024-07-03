import numpy as np
import tensorflow as tf
from tensorflow import keras

# Definujeme jednoduché prostredie
class Environment:
    def __init__(self):
        self.steps_left = 10

    def get_observation(self):
        return np.array([self.steps_left])

    def get_actions(self):
        return [0, 1]

    def is_done(self):
        return self.steps_left == 0

    def action(self, action):
        if self.is_done():
            raise Exception("Hra skončila")
        self.steps_left -= 1
        reward = 1.0 if action == 1 else -1.0
        return reward

# Vytvoríme model
model = keras.Sequential([
    keras.layers.Dense(5, input_shape=(1,), activation='relu'),
    keras.layers.Dense(2, activation='linear')
])

# Skompilujeme model
model.compile(optimizer='adam', loss='mse')

# Nastavíme parametre učenia
gamma = 0.99
epsilon = 0.1

# Trénovacia slučka
num_episodes = 1000

for episode in range(num_episodes):
    env = Environment()
    total_reward = 0
    observations = []
    actions = []
    rewards = []

    while not env.is_done():
        observation = env.get_observation()
        if np.random.random() < epsilon:
            action = np.random.choice(env.get_actions())
        else:
            q_values = model.predict(observation[np.newaxis])[0]
            action = np.argmax(q_values)

        reward = env.action(action)
        next_observation = env.get_observation()

        observations.append(observation)
        actions.append(action)
        rewards.append(reward)

        total_reward += reward

    # Vypočítame Q-hodnoty
    q_values = []
    q_value = 0
    for reward in reversed(rewards):
        q_value = reward + gamma * q_value
        q_values.insert(0, q_value)

    # Trénujeme model
    observations = np.array(observations)
    q_values = np.array(q_values)
    target_q_values = model.predict(observations)
    target_q_values[np.arange(len(actions)), actions] = q_values
    model.fit(observations, target_q_values, verbose=0)

    if episode % 100 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward}")

# Test naučeného modelu
env = Environment()
total_reward = 0
while not env.is_done():
    observation = env.get_observation()
    q_values = model.predict(observation[np.newaxis])[0]
    action = np.argmax(q_values)
    reward = env.action(action)
    total_reward += reward

print(f"Test Reward: {total_reward}")
