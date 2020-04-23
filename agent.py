from collections import namedtuple, deque
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tensorboardX import SummaryWriter


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class QNet(torch.nn.Module):
    """ The model to be used in training the agent
    ====
    state_size: the env state size \n
    action_size: the number actions in the env
    """
    def __init__(self, state_size, action_size):
        """ The model to be used in training the agent
        ====
        state_size: the env state size \n
        action_size: the number actions in the env
        """
        super(QNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_size, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Memory:
    def __init__(self, action_size, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.learnt = namedtuple("Learnt", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        e = self.learnt(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(DEVICE)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(DEVICE)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(DEVICE)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(DEVICE)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(DEVICE)

        return (states, actions, rewards, next_states, dones)
    def __len__(self):
        return len(self.memory)

class Agent():
    def __init__(self, state_size, action_size, batch_size, gamma=0.999, upr=1e-3, update_every=4):
        """This is the agent to be trained

            Param
            ===
            state_size: env state size \n
            action_size: actions size \n
            batch_size: minibatch size \n
            net: pretrainded model \n
            gamma: discount factor (0.999 defualt) \n
            upr: update policy rate \n
            update_every: when to update the network (4 defualt) \n
        """
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.update_e = update_every
        self.writer = SummaryWriter(comment="-Agent-")
        self.upr = upr

        self.policy_net = QNet(state_size, action_size).to(DEVICE)
        self.target_net = QNet(state_size, action_size).to(DEVICE)

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=3e-3)

        # Replay memory
        self.memory = Memory(action_size, int(1e4), 64)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # Save to  memory
        self.memory.add(state, action, reward, next_state, done)
        # Learn every update_every time steps.
        if self.t_step % self.update_e == 0:
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)

        self.t_step = self.t_step + 1

    def act(self, state, eps=0.):
        """
            Return an actions for given state.
            Params
            
            state (array_like): current state
            eps (float): epsilon
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
        self.policy_net.eval()
        with torch.no_grad():
            action_values = self.policy_net(state)
        self.policy_net.train()
        rand = np.random.random()
        # Epsilon-greedy action selection
        if rand > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return np.random.choice(np.arange(self.action_size))

    def learn(self, learnt, gamma):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """

        states, actions, rewards, next_states, dones = learnt

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.target_net(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.policy_net(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)


        self.writer.add_scalar("loss", loss, self.t_step)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update target network
        self.update_target(self.policy_net, self.target_net, self.upr)

    def update_target(self, policy_model, target_model, uip):
        """Update target model parameters.
        Params
        ======
            policy_model: weights from policy will be copied from
            target_model: weights will be copied to
            uip: update interpolation parameter
        """
        for target_param, policy_param in zip(target_model.parameters(), policy_model.parameters()):
            target_param.data.copy_(uip*policy_param.data + (1.0-uip)*target_param.data)