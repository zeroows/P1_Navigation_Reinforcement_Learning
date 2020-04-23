import argparse
import torch
import numpy as np
from unityagents import UnityEnvironment

from agent import Agent, QNet

DEFAULT_ENV_NAME = "./Banana_Linux/Banana.x86_64"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env", default=DEFAULT_ENV_NAME,
                        help="Environment name to use, default=" + DEFAULT_ENV_NAME)
    parser.add_argument("-m", "--model", required=False,
                        help="Model file to load to retain")
    args = parser.parse_args()

    print("Testing Trained agent ...")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    env = UnityEnvironment(file_name=args.env)

    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # obtain initial observation
    env_info = env.reset(train_mode=False)[brain_name]
    # number of agents in the environment
    print('Number of agents:', len(env_info.agents))
    # number of actions
    action_size = brain.vector_action_space_size
    print('Number of actions:', action_size)
    # observation space size
    state = env_info.vector_observations[0]
    state_size = len(state)
    print('States have length:', state_size)

    agent = Agent(state_size, action_size=action_size, batch_size=64)
    agent.policy_net.load_state_dict(torch.load('target_model.pth'))

    for i in range(3):
        env_info = env.reset(train_mode=False)[brain_name]
        state = env_info.vector_observations[0] 
        score = 0
        for j in range(200):
            action = agent.act(state)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            agent.step(state, action, reward, next_state, done)
            score += reward
            state = next_state
            if done:
                break
        print("Episode {} is done total score is {}".format(i+1, score))        
    env.close()


if __name__ == "__main__":
    main()