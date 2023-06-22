import torch 
import torch.nn as nn
import torch.optim as optim
from  ReplayBufferMultiAgent import ReplayBuffer
from MultiAgentNetwork import Qnetwork,  SimpleNetwork , SimpleDiffrentLossFunction , SimpleNetworkWithDiffrentOptimizer , MoreLayerDiffrentLossFunction , MoreLayersNetwork , MoreLayersNetworkDiffrentOptimizer
import random 
import os 



class Agent:

    #fc1 need to become a list of  hyperparamter for each  agent  
    #loss types  nn.MSELoss() , nn.BCELoss(),nn.CrossEntropyLoss(),nn.KLDivLoss() ,nn.HingeEmbeddingLoss() , nn.SmoothL1Loss()
    def __init__(self, input_dimlsit, fc1_dimlsit, fc2_dimlist, fc3_dimlist , fc4_dimlist , n_actions, lrlist,losslist ,  batch_size, mem_size, gamma_list, num_agents):
        self.num_agents = num_agents
        self.agents = []

        


        Networks_list  = [ Qnetwork , MoreLayerDiffrentLossFunction , SimpleNetworkWithDiffrentOptimizer , MoreLayerDiffrentLossFunction , MoreLayersNetwork , MoreLayersNetworkDiffrentOptimizer]
        self.gamma_list = gamma_list

        for index  in range(num_agents):

            input_dim = input_dimlsit[index]
            fc1_dim =   fc1_dimlsit[index]
            fc2_dim =   fc2_dimlist[index]
            fc3_dim =   fc3_dimlist[index]
            fc4_dim =   fc4_dimlist[index]
            lr = lrlist[index]
            loss = losslist[index]

            agent_mem = ReplayBuffer(mem_size, input_dim, n_actions)
            #every agent will  have his own network  
            agent_network = Qnetwork(input_dim, fc1_dim, fc2_dim, fc3_dim ,fc4_dim ,n_actions, lr, loss)
            gamma = gamma_list[index]

            agent = {
                'mem': agent_mem,
                'network': agent_network,
                'epsilon': 0,
                'n_games': 0,
                'gamma': gamma  # Assign gamma value to agent

            }

            self.agents.append(agent)
            self.batch_size = batch_size


    def choose_action(self, states):
        actions = []
        for agent_index, agent in enumerate(self.agents):
            #increase the exploartion  with some exploitation  of course  

            epsilon = 1000 - agent['n_games']
            final_move = [0, 0, 0]
            if random.randint(0, 1000) < epsilon:
                move = random.randint(0, 2)
                final_move[move] = 1
            else:
                state = states[agent_index]
                state_tensor = torch.tensor(state, dtype=torch.float)
                prediction = agent['network'](state_tensor)
                move = torch.argmax(prediction).item()
                final_move[move] = 1
            actions.append(final_move)
        return actions

    def short_mem(self, states, next_states, actions, rewards, dones):
        for agent_index, agent in enumerate(self.agents):
            agent['mem'].store_transition(states[agent_index], next_states[agent_index],
                                        actions[agent_index], rewards[agent_index], dones[agent_index])
            agent['n_games'] += 1

            agent['epsilon'] = 100 - agent['n_games']

        self.learn()

    def long_mem(self):
        for agent in self.agents:
            if self.batch_size < agent['mem'].mem_cntr:
                self.learn()

    def save(self, agent_idx , Zeitpunkt):
        file_name=f'Agent{agent_idx}TakenAt{Zeitpunkt}.pth'
        model_folder_path = f'./Save/SaveEachXSteps/{agent_idx}'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        agent = self.agents[agent_idx]
        file_name = os.path.join(model_folder_path, file_name)
        file_name_agent = f'{file_name}_agent_{agent_idx}'
        torch.save(agent['network'].state_dict(), file_name_agent)

    def learn(self):
        for agent_index, agent in enumerate(self.agents):
            states, next_states, actions, rewards, dones = agent['mem'].sample_batch(self.batch_size)

            state_tensor = torch.tensor(states, dtype=torch.float)
            next_state_tensor = torch.tensor(next_states, dtype=torch.float)
            action_tensor = torch.tensor(actions, dtype=torch.long)
            reward_tensor = torch.tensor(rewards, dtype=torch.float)
            done_tensor = torch.tensor(dones)

            if len(state_tensor.shape) == 1:
                state_tensor = torch.unsqueeze(state_tensor, 0)
                next_state_tensor = torch.unsqueeze(next_state_tensor, 0)
                action_tensor = torch.unsqueeze(action_tensor, 0)
                reward_tensor = torch.unsqueeze(reward_tensor, 0)
                done_tensor = torch.unsqueeze(done_tensor, 0)

            pred = agent['network'](state_tensor)

            target = pred.clone()
            for idx in range(len(done_tensor)):
                Q_new = reward_tensor[idx]
                if not done_tensor[idx]:
                    Q_new = reward_tensor[idx] + agent['gamma'] * torch.max(agent['network'](next_state_tensor[idx]))

                target[idx][torch.argmax(action_tensor[idx]).item()] = Q_new

            agent['network'].optimizer.zero_grad()
            loss = agent['network'].loss(target, pred)
            loss.backward()
            agent['network'].optimizer.step()

