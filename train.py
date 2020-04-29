
import argparse
import torch
import numpy as np
from unityagents import UnityEnvironment
from collections import deque

from agent import Agent, QNet

DEFAULT_ENV_NAME = "./Banana_Linux/Banana.x86_64"
DEFAULT_EPISODES = 2000


def train_dqn(agent, env, brain_name, n_episodes=2000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    last_100_scores = deque(maxlen=100)  # last 100 scores
    scores = []
    eps = eps_start

    print("training the model for: ", n_episodes, " episodes")
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        score = 0
        done = False
        while not done:
            action = agent.act(state, eps)
            env_info = env.step(int(action))[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            agent.step(state, action, reward, next_state, done)
            score += reward
            state = next_state

        last_100_scores.append(score)
        scores.append(score)
        eps = max(eps_end, eps_decay*eps)
        print('\rEpisode {}\tEpisode Score: {:.2f}'.format(
            i_episode, score), end="")

        mean_last_100_scores = np.mean(last_100_scores)

        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Episodes Score: {:.2f}'.format(
                i_episode, mean_last_100_scores))

        if mean_last_100_scores >= 15.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(
                i_episode, mean_last_100_scores))
            torch.save(agent.policy_net.state_dict(), 'pmodel.pth')
            torch.save(agent.target_net.state_dict(), 'model.pth')
            break

    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env", default=DEFAULT_ENV_NAME,
                        help="Environment name to use, default=" + DEFAULT_ENV_NAME)
    parser.add_argument("-m", "--model", required=False,
                        help="Model file to load to retain")
    parser.add_argument("-ec", "--episodes_count",
                        required=False, help="Number of episodes to train")
    args = parser.parse_args()

    print("Training ...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    env = UnityEnvironment(file_name=args.env)

    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # obtain initial observation
    env_info = env.reset(train_mode=True)[brain_name]
    # number of agents in the environment
    print('Number of agents:', len(env_info.agents))
    # number of actions
    action_size = brain.vector_action_space_size
    print('Number of actions:', action_size)
    # observation space size
    state = env_info.vector_observations[0]
    state_size = len(state)
    print('States have length:', state_size)

    agent = Agent(state_size, action_size=action_size,
                  batch_size=64, update_every=4)

    # if args.model:
    #     print("loading pretrained model: ", args.model)
    #     weights = torch.load(args.model)
    #     agent.policy_net.load_state_dict(weights)

    if args.episodes_count:
        train_dqn(agent, env, brain_name, int(args.episodes_count))
    else:
        train_dqn(agent, env, brain_name)


if __name__ == "__main__":
    main()
