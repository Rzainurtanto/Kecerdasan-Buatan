# Example 3.23 OpenAI Gym CartPole
# Gunakan Gymnasium karena Gym yang lama tidak kompatibel dengan NumPy 2.x
import gymnasium as gym


env = gym.make('CartPole-v1', render_mode='human')
for i_episode in range(20):
    observation, info = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break

env.close()
