
import argparse
import torch
import numpy as np
from unityagents import UnityEnvironment


from agent import Agent, QNet


DEFAULT_ENV_NAME = "./Banana_Linux/Banana.x86_64"
DEFAULT_EPISODES = 5000

def train_dqn(agent, env, brain_name, n_episodes=2000, max_t=100, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    scores = []
    eps = eps_start
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0] 
        score = 0
        for t in range(max_t):
            action = int(agent.act(state, eps))
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            agent.step(state, action, reward, next_state, done)
            score += reward
            state = next_state
            if done:
                break
        scores.append(score)
        eps = max(eps_end, eps_decay*eps)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores)), end="")
        
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores)))

        if i_episode % 1000 == 0:
            print('\nSaving target net {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores)))
            torch.save(agent.target_net.state_dict(), 'target_model.pth')

        if np.mean(scores) >= 13.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores)))
            torch.save(agent.policy_net.state_dict(), 'policy_model.pth')
            torch.save(agent.target_net.state_dict(), 'target_model.pth')
            break
    return scores

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env", default=DEFAULT_ENV_NAME,
                        help="Environment name to use, default=" + DEFAULT_ENV_NAME)
    parser.add_argument("-m", "--model", required=False, help="Model file to load to retain")
    parser.add_argument("-ec", "--episodes_count", default=DEFAULT_EPISODES, help="Number of episodes to train")
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

    agent = Agent(state_size, action_size=action_size, batch_size=64, update_every=4)
    
    if args.model:
        print("loading pretrained model: ", args.model)
        weights = torch.load(args.model)
        agent.policy_net.load_state_dict(weights)
        agent.target_net.load_state_dict(weights)
    print("training the model for: ", args.episodes_count, " episodes")
    train_dqn(agent, env, brain_name, int(args.episodes_count))

if __name__ == "__main__":
    main()