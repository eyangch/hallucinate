import gymnasium as gym
import time
import numpy as np
from PIL import Image
import random

async def play_actions(num_iter):
    env = gym.make("CartPole-v1", render_mode="rgb_array")

    observation, _ = env.reset()

    t0 = time.time()

    data = []
    keys = []

    for i in range(num_iter):
        # Choose an action: 0 = push cart left, 1 = push cart right
        if random.random() < 0.7:
            action = env.action_space.sample()  # Random action for now - real agents will be smarter!
        elif observation[2] < 0:
            action = 0
        else:
            action = 1
        keys.append(action)

        # Take the action and see what happens
        observation, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            env.reset()

        img = Image.fromarray(env.render()).resize((192, 192))
        env_render = np.array(img)
        data.append(env_render)
        #print(env_render.shape)

        #env.render()

        if terminated or truncated:
            env.reset()
        
    env.close()

    data = np.array(data)
    keys = np.array(keys)

    print(time.time() - t0)

    return data, keys

if __name__ == "__main__":
    import asyncio
    asyncio.run(play_actions(1000))