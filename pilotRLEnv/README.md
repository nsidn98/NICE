# pilotRL environment
An [OpenAI gym](https://gym.openai.com/docs/) type environment for scheduing pilots to training events.

Usage:

Only have to use the `reset()` and `step()` method to access everything related to the environment. 
```python
import numpy as np
# if running from the main folder
from pilotRLEnv.env import PilotRLEnv 
from pilotRLEnv.config import args
# config.args has the configuration for all the parameters for the environment

env = PilotRLEnv(args)

for episode in range(10):
    state = env.reset()
    done = False
    episode_reward = 0
    while not done:
        # sample a pilot from all the valid pilots for the current event
        action = env.sampleAction()
        # take a step in the environment
        next_state, reward, done, info = env.step(action)
        episode_reward += reward
    print(f"Episode reward in episode:{episode} is {episode_reward:.3f}")
```
