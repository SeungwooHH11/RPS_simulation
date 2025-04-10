import numpy as np
import pandas as pd
import os
import random
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

device = 'cuda'
np.random.seed(1)
random.seed(1)
torch.manual_seed(1)


class ConvLayer(nn.Module):
    def __init__(self, node_fea_len, edge_fea_len):
        super(ConvLayer, self).__init__()
        self.node_fea_len = node_fea_len
        self.edge_fea_len = edge_fea_len  # 여기서 fc layer 하나더 추가해도 될듯
        self.fc_full = nn.Linear(2 * self.node_fea_len + self.edge_fea_len,
                                 2 * self.node_fea_len).to(device)
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()
        self.alpha = nn.Parameter(torch.tensor(0.7,dtype=torch.float32))
        self.initialize_weights()

    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # He 초기화를 사용하여 가중치를 초기화합니다.
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                # 편향을 0으로 초기화합니다.
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                # 배치 정규화 레이어의 가중치를 초기화합니다.
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, node_in_fea, edge_fea, edge_fea_idx):
        N, M = edge_fea_idx.shape
        # convolution
        node_edge_fea = node_in_fea[edge_fea_idx, :]  # edge fea idx -> 시작 노드 -도착 노드

        total_nbr_fea = torch.cat([node_in_fea.unsqueeze(1).expand(N, M, self.node_fea_len), node_edge_fea, edge_fea],
                                  dim=2)

        total_gated_fea = self.fc_full(total_nbr_fea)
        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus(nbr_core)
        mask = torch.where(edge_fea_idx < 0, torch.tensor(0), torch.tensor(1))
        nbr_filter = nbr_filter * mask.unsqueeze(2)
        nbr_core = nbr_filter * mask.unsqueeze(2)
        nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1)
        
        out = self.softplus(self.alpha*node_in_fea + nbr_sumed)

        return out


class CrystalGraphConvNet(nn.Module):
    def __init__(self, orig_node_fea_len, orig_edge_fea_len, edge_fea_len, node_fea_len,
                 final_node_len, dis):
        super(CrystalGraphConvNet, self).__init__()
        self.embedding_n = nn.Linear(orig_node_fea_len, node_fea_len).to(device)
        self.embedding_e = nn.Linear(orig_edge_fea_len, edge_fea_len).to(device)
        self.dis=dis
        N=dis.shape[0]
        self.convs1 = ConvLayer(node_fea_len, edge_fea_len)
        self.convs2 = ConvLayer(node_fea_len, edge_fea_len)
        self.convs3 = ConvLayer(node_fea_len, edge_fea_len)
        self.final_layer = nn.Linear(node_fea_len,int(final_node_len/2)).to(device)
        self.conv_to_fc = nn.Linear(final_node_len*N,256).to(device)
        self.readout1 = nn.Linear(256, 128).to(device)
        self.readout2 = nn.Linear(128, 64).to(device)
        self.fc_out = nn.Linear(64, 1).to(device)
        self.DA_weight = nn.Parameter(torch.tensor(48/5,dtype=torch.float32))  # Learnable weight parameter
        self.DA_bias = nn.Parameter(torch.tensor(-28/5,dtype=torch.float32))
        self.DA_act=nn.Sigmoid()

        self.act_fun = nn.ELU()


        self.initialize_weights()
                     
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # He 초기화를 사용하여 가중치를 초기화합니다.
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                # 편향을 0으로 초기화합니다.
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                # 배치 정규화 레이어의 가중치를 초기화합니다.
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        
    def forward(self, node_fea, edge_fea, edge_fea_idx):
        node_fea = self.embedding_n(node_fea)  # N,fea
        edge_fea = self.embedding_e(edge_fea)
        node_fea = self.convs1(node_fea, edge_fea, edge_fea_idx)
        node_fea = self.convs2(node_fea, edge_fea, edge_fea_idx)
        node_fea = self.convs3(node_fea, edge_fea, edge_fea_idx)
        return node_fea

    def readout(self, node_fea):
        B,N,M=node_fea.shape
        node_fea = self.conv_to_fc(node_fea.view(B,-1))  # batch
        node_fea = self.act_fun(node_fea)
        node_fea = self.readout1(node_fea)
        node_fea = self.act_fun(node_fea)
        node_fea = self.readout2(node_fea)
        node_fea = self.act_fun(node_fea)
        out = self.fc_out(node_fea)
        return out


class MLP(nn.Module):
    def __init__(self, state_size, output_size):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output_size)
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # He 초기화를 사용하여 가중치를 초기화합니다.
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                # 편향을 0으로 초기화합니다.
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                # 배치 정규화 레이어의 가중치를 초기화합니다.
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
                
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class PPO(nn.Module):
    def __init__(self, learning_rate, lmbda, gamma, alpha, beta, epsilon, discount_factor,
                 location_num,transporter_type,dis):
        super(PPO, self).__init__()
        self.transporter_type=transporter_type
        self.node_fea_len = 32
        self.final_node_len=32
        self.edge_fea_len = 32
        self.gnn = CrystalGraphConvNet(orig_node_fea_len=int(2*self.transporter_type+1), orig_edge_fea_len=int(3+2*self.transporter_type), edge_fea_len=self.edge_fea_len, node_fea_len=self.node_fea_len, final_node_len=32, dis=dis)
        self.pi = MLP(32 + 2*int(3+2*self.transporter_type) + 5 + 5, 1).to(device)
        self.temperature = 1.0
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.lmbda = lmbda
        self.gamma = gamma
        self.alpha = alpha
        self.discount_factor = discount_factor
        self.beta = beta
        self.epsilon = epsilon

    def calculate_GNN(self, node_fea, edge_fea, edge_fea_idx):
        return self.gnn(node_fea, edge_fea, edge_fea_idx)

    def calculate_pi(self, state_gnn, node_fea, edge_fea, edge_fea_idx, distance, tp_type):
        # node_fea 9,13
        # edge_fea 9,3,5
        # edge_fea_idx 9, 3
        # distance 9,3
        # tp_type float=> 9 3 5
        action_variable = state_gnn[edge_fea_idx, :]
        edge_fea_tensor = edge_fea.repeat(1, 1, 2)

        distance_tensor = distance.unsqueeze(2).repeat(1, 1, 5)

        action_variable = torch.cat([action_variable, edge_fea_tensor], 2)

        action_variable = torch.cat([action_variable, distance_tensor], 2)
        action_variable = torch.cat(
            (action_variable, torch.full((edge_fea_idx.shape[0], edge_fea_idx.shape[1], 5), float(tp_type)/self.transporter_type).to(device)), dim=2)
        
        '''one_hot_vector = torch.zeros(self.transporter_type).to(device)
        
        # 현재 타입의 위치에 1을 할당
        one_hot_vector[int(tp_type)] = 1
        reshaped_tensor = one_hot_vector.expand(edge_fea_idx.shape[0], edge_fea_idx.shape[1], self.transporter_type)
        action_variable = torch.cat([action_variable, reshaped_tensor], 2)'''
        action_probability = self.pi(action_variable)
        return action_probability

    def get_action(self, node_fea, edge_fea, edge_fea_idx, mask, distance, tp_type):
        with torch.no_grad():
            N, M = edge_fea_idx.shape
            # print(edge_fea_idx)
            # print(edge_fea)
            # print(node_fea)
            state = self.calculate_GNN(node_fea, edge_fea, edge_fea_idx)
            # print(state)
            probs = self.calculate_pi(state, node_fea, edge_fea, edge_fea_idx, distance, tp_type)
            # print(probs) # type0 weight 작다
            logits_masked = probs - 1e8 * mask
            # print(logits_masked)
            prob = torch.softmax((logits_masked.flatten() - torch.max(logits_masked.flatten()))/self.temperature, dim=-1)
            m = Categorical(prob)
            action = m.sample().item()
            i = int(action / M)
            j = int(action % M)
            while edge_fea_idx[i][j] < 0:
                action = m.sample().item()
                i = int(action / M)
                j = int(action % M)
            return action, i, j, prob[action]

    def calculate_v(self, x):
        return self.gnn.readout(x)

    def update(self, data, probs, rewards, action, dones,step1,model_dir):
        num = 0
        ave_loss = 0
        en_loss = 0
        v_loss = 0
        p_loss = 0
        # data-> episode-> state [30,16,5]  node_fea (9,13),edge_fea (9,3,5),edge_fea_idx(9,3), distance (9,3) type (1), mask(9,3), 
        # probs [30*15] numpy
        # rewards [30*15]
        # action [30*15]
        # dones [30*15]
        probs = torch.tensor(probs, dtype=torch.float32).unsqueeze(1).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)

        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)

        tr = 0
        step = 0
        sw = 0
        for episode in data:
            for state in episode:

                # node_fea=torch.tensor(state[0],dtype=torch.float32).to(device)
                # edge_fea=torch.tensor(state[1],dtype=torch.float32).to(device)
                # edge_fea_idx=torch.tensor(state[2],dtype=torch.int32).to(device)
                # distance=torch.tensor(state[3],dtype=torch.float32).to(device)
                # type
                state_gnn = self.calculate_GNN(state[0], state[1], state[2])
                
                if step < len(episode) - 1:
                    #data(state):  node_fea (9,13),edge_fea (9,3,5),edge_fea_idx(9,3),  distance (9,3) type (1), mask(9,3)
                    #cal_pi:  state_gnn, node_fea, edge_fea, edge_fea_idx, distance, tp_type)
                    prob_a = self.calculate_pi(state_gnn, state[0], state[1], state[2], state[3], state[4])
                    mask = state[5]
                    logits_maksed = prob_a - 1e8 * mask
                    prob = torch.softmax((logits_maksed.flatten() - torch.max(logits_maksed.flatten()))/self.temperature, dim=-1)
                    pi_a = prob[int(action[sw])]
                    sw += 1
                    if tr == 0:
                        pi_a_total = pi_a.unsqueeze(0)
                    else:
                        pi_a_total = torch.cat([pi_a_total, pi_a.unsqueeze(0)])
                state_gnn = state_gnn.unsqueeze(0)
                if tr == 0:
                    state_GNN = state_gnn
                elif tr == 1:
                    next_state_GNN = state_gnn
                    state_GNN = torch.cat([state_GNN, state_gnn])
                elif step == 0:
                    state_GNN = torch.cat([state_GNN, state_gnn])
                elif step == len(episode) - 1:
                    next_state_GNN = torch.cat([next_state_GNN, state_gnn])
                else:
                    state_GNN = torch.cat([state_GNN, state_gnn])
                    next_state_GNN = torch.cat([next_state_GNN, state_gnn])
                tr += 1
                step += 1
            step = 0

        total_time_step = sw
        state_v = self.calculate_v(state_GNN)
        
        state_next_v = self.calculate_v(next_state_GNN)
        td_target = rewards + self.gamma * state_next_v * dones
        delta = td_target - state_v
        advantage_lst = np.zeros(total_time_step)
        advantage_lst = torch.tensor(advantage_lst, dtype=torch.float32).unsqueeze(1).to(device)
        i = 0
        for e,episode in enumerate(data):
            if e<1:
                advantage = 0.0
                for t in reversed(range(i, i + len(episode)-1)):
                    advantage = self.gamma * self.lmbda * advantage + delta[t][0]
                    advantage_lst[t][0] = advantage
                i += len(episode)-1
        
        ratio = torch.exp(torch.log(pi_a_total.unsqueeze(1)) - torch.log(probs))  # a/b == exp(log(a)-log(b))

        surr1 = ratio * advantage_lst
        surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage_lst
        loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(state_v, td_target.detach()) * self.alpha

        ave_loss = loss.mean().item()
        v_loss = (self.alpha * F.smooth_l1_loss(state_v, td_target.detach())).item()
        p_loss = -torch.min(surr1, surr2).mean().item()

        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()
        if step1 % 40 == 0:
            torch.save({
                'model_state_dict': self.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),

            }, model_dir+'trained_model' + str(step1) + '.pth')

        return ave_loss, v_loss, p_loss




