import numpy as np
import pandas as pd
import os
import random
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
TORCH_USE_CUDA_DSA=1
device = 'cuda'
np.random.seed(1)
random.seed(1)
torch.manual_seed(1)



# GCN Layer 정의
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features))
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, A_hat, X):
        return F.relu(torch.matmul(torch.matmul(A_hat, X), self.weight))

# GCNModel1 정의
class GCNModel1(nn.Module):
    def __init__(self, input_dim, hidden_dim,A_hat):
        super(GCNModel1, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.gcn1 = GCNLayer(hidden_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, hidden_dim)
        self.gcn3 = GCNLayer(hidden_dim, hidden_dim)
        self.init_weights()
        self.A=A_hat

    def init_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        for m in self.modules():
            if isinstance(m, GCNLayer):
                m.init_weights()

    def forward(self,  X):
        X=self.embedding(X)
        X = self.gcn1(self.A, X)/4.0
        X = self.gcn2(self.A, X)/4.0
        X = self.gcn3(self.A, X)/4.0
        
        return X

# GCNModel2 정의
class GCNModel2(nn.Module):
    def __init__(self, input_dim, hidden_dim,Up_A_hat, Down_A_hat, Right_A_hat, Left_A_hat):
        super(GCNModel2, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.gcn_layers = nn.ModuleList([GCNLayer(hidden_dim, hidden_dim) for _ in range(12)])
        self.a = nn.Parameter(torch.tensor(0.8, dtype=torch.float32))
        self.init_weights()
        self.U=Up_A_hat
        self.D=Down_A_hat
        self.R=Right_A_hat
        self.L=Left_A_hat

    def init_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        for layer in self.gcn_layers:
            layer.init_weights()

    def forward(self, X):
        X = self.embedding(X)
        for i in range(0, 12, 4):
            X = (
                self.gcn_layers[i](self.U, X) +
                self.gcn_layers[i+1](self.D, X) +
                self.gcn_layers[i+2](self.R, X) +
                self.gcn_layers[i+3](self.L, X) +
                self.a * X
            )/4.0
        return X


# Actor 네트워크 (어디에 배치할지 결정)
class Actor_net(nn.Module):
    def __init__(self, hidden_dim):
        super(Actor_net, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, int(hidden_dim/2))
        self.fc2 = nn.Linear(int(hidden_dim/2), int(hidden_dim/2))
        self.fc3 = nn.Linear(int(hidden_dim/2), int(hidden_dim/4))
        self.fc4 = nn.Linear(int(hidden_dim/4), 1)
        self.initialize_weights()


    def forward(self, X, mask):
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = F.relu(self.fc3(X))
        X = F.relu(self.fc4(X))
        
        X = X - mask * 1e8  #mask b, r*c, 1
        
        return F.softmax((X), dim=1)
        
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # He 초기화를 사용하여 가중치를 초기화합니다.
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                # 편향을 0으로 초기화합니다.
                nn.init.zeros_(m.bias)

# Critic 네트워크 (State Value 계산)
class Critic_net(nn.Module):
    def __init__(self, input_dim):
        super(Critic_net, self).__init__()
        self.fc1 = nn.Linear(input_dim, int(input_dim/4))
        self.fc2 = nn.Linear(int(input_dim/4), int(input_dim/8))
        self.fc3 = nn.Linear(int(input_dim/8), int(input_dim/16))
        self.fc4 = nn.Linear(int(input_dim/16), 1)
        self.initialize_weights()


    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = F.relu(self.fc3(X))
        X = F.relu(self.fc4(X))
        
        return X
        
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # He 초기화를 사용하여 가중치를 초기화합니다.
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                # 편향을 0으로 초기화합니다.
                nn.init.zeros_(m.bias)
            

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output_dim)
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # He 초기화를 사용하여 가중치를 초기화합니다.
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                # 편향을 0으로 초기화합니다.
                nn.init.zeros_(m.bias)
            
                
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class Placement(nn.Module):
    def __init__(self, feature_dim, hidden_dim, lookahead_block_num ,grid_size, learning_rate, lmbda, gamma, alpha, beta, epsilon, mod):
        super(Placement, self).__init__()
        self.r=grid_size[0]
        self.c=grid_size[1]
        adj_matrix,Up_A_hat,Down_A_hat,Left_A_hat,Right_A_hat=self.grid_to_adjacency_matrix(self.r, self.c)
        self.A=torch.tensor(adj_matrix,dtype=torch.float32).to(device)
        self.U=torch.tensor(Up_A_hat,dtype=torch.float32).to(device)
        self.D=torch.tensor(Down_A_hat,dtype=torch.float32).to(device)
        self.R=torch.tensor(Right_A_hat,dtype=torch.float32).to(device)
        self.L=torch.tensor(Left_A_hat,dtype=torch.float32).to(device)
        self.hidden_dim = hidden_dim
        self.input_dim=feature_dim
        self.mod=mod
        #self.embedding=nn.Linear()
        self.lookahead_block_num=lookahead_block_num
        if mod=='GCN1':
            self.gnn = GCNModel1(self.input_dim,self.hidden_dim,self.A).to(device)
        if mod=='GCN2':
            self.gnn = GCNModel2(self.input_dim,self.hidden_dim,self.U,self.D,self.R,self.L).to(device)
        self.Actor_net = Actor_net(self.hidden_dim+self.lookahead_block_num*feature_dim).to(device)
        #self.Critic_net = Critic_net(self.r*self.c*self.hidden_dim+self.lookahead_block_num*feature_dim).to(device)
        if mod=='MLP':
            self.Actor_net = MLP((self.r*self.c+self.lookahead_block_num)*feature_dim,self.r*self.c).to(device)
            self.Critic_net =MLP((self.r*self.c+self.lookahead_block_num)*feature_dim+1,1).to(device)
        
        self.temperature = 1.5
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.lmbda = lmbda
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon

    
    def grid_to_adjacency_matrix(self,n, m):
        size = n * m
        adj_matrix = np.zeros((size, size), dtype=np.float32)
        Up_A_hat = np.zeros((size, size), dtype=np.float32)
        Down_A_hat = np.zeros((size, size), dtype=np.float32)
        Right_A_hat = np.zeros((size, size), dtype=np.float32)
        Left_A_hat = np.zeros((size, size), dtype=np.float32)
        for i in range(n):
            for j in range(m):
                idx = i * m + j
                
                # 상 (i-1, j)
                if i > 0:
                    adj_matrix[idx][(i-1) * m + j] = 1
                    adj_matrix[(i-1) * m + j][idx] = 1
                    Up_A_hat[idx][(i-1) * m + j] = 1
                    Up_A_hat[(i-1) * m + j][idx] = 1
                    
    
                # 하 (i+1, j)
                if i < n - 1:
                    adj_matrix[idx][(i+1) * m + j] = 1
                    adj_matrix[(i+1) * m + j][idx] = 1
                    Down_A_hat[idx][(i+1) * m + j] = 1
                    Down_A_hat[(i+1) * m + j][idx] = 1
                    
                # 좌 (i, j-1)
                if j > 0:
                    adj_matrix[idx][i * m + (j-1)] = 1
                    adj_matrix[i * m + (j-1)][idx] = 1
                    Left_A_hat[idx][i * m + (j-1)] = 1
                    Left_A_hat[i * m + (j-1)][idx] = 1
                    
                # 우 (i, j+1)
                if j < m - 1:
                    adj_matrix[idx][i * m + (j+1)] = 1
                    adj_matrix[i * m + (j+1)][idx] = 1
                    Right_A_hat[idx][i * m + (j+1)] = 1
                    Right_A_hat[i * m + (j+1)][idx] = 1
    
    
        # Self-loop 추가 (GCN 학습 안정화)
        np.fill_diagonal(adj_matrix, 1)
        
    
        return adj_matrix,Up_A_hat,Down_A_hat,Left_A_hat,Right_A_hat


    def Locate(self,grids,blocks,masks,ans):
        b,r,c,f=grids.shape #blocks b, lookahead*featuredim
        blocks=blocks.reshape(b,-1)
        
        if self.mod=='GCN1' or self.mod=='GCN2':
            new_grids=grids.reshape(b,-1,f)
            blocks_expanded = blocks.unsqueeze(1).repeat(1, r*c, 1)  # (b, r*c, lookahead*featuredim)
            if ans!= None:
                output_vector=self.gnn(new_grids) #b,r*c,hidden_dim
                merged_tensor = torch.concat([output_vector, blocks_expanded], dim=2)
                output=self.Actor_net(merged_tensor,masks) # b, r*c, 1
                pi = output.squeeze(-1).gather(1, ans)
                return pi,ans,output
            else:
                with torch.no_grad():
                    output_vector=self.gnn(new_grids) #b,r*c,hidden_dim
                    merged_tensor = torch.concat([output_vector, blocks_expanded], dim=2)
                    output=self.Actor_net(merged_tensor,masks).squeeze(-1)
                    samples = torch.multinomial(output, 1) 
                    pi = output.gather(1, samples)
                    return pi,samples
        else:
            new_grids=grids.reshape(b,-1)
            if ans!=None:
                merged_tensor=torch.concat([new_grids, blocks], dim=1)
                output=self.Actor_net(merged_tensor) #b,r*c
                output=output-masks.squeeze(-1)* 1e8 #b, r*C
                output=F.softmax(output,dim=-1)
                pi = output.gather(1, ans)
                return pi, ans,output
            else:
                merged_tensor=torch.concat([new_grids, blocks], dim=1)
                output=self.Actor_net(merged_tensor)
                output=output-masks.squeeze(-1)* 1e8 #b*r*C
                output=F.softmax(output,dim=-1)
                samples = torch.multinomial(output, 1) 
                pi = output.gather(1, samples)
                return pi, samples
    
    def calculate_v(self, grids,blocks,block_lefts):
        b,r,c,f=grids.shape #blocks b, lookahead*featuredim
        blocks=blocks.reshape(b,-1)

        if self.mod=='GCN1' or self.mod=='GCN2':
            new_grids=grids.reshape(b,-1,f)
            output_vector=self.gnn(new_grids) #b,r*c,hidden_dim
            output_vector=output_vector.reshape(b,-1)
            merged_tensor = torch.cat([output_vector, blocks], dim=1)
            merged_tensor=torch.cat([merged_tensor,block_lefts],dim=1)
            state_values=self.Critic_net(merged_tensor) #b,1
        else:
            new_grids = grids.reshape(b,-1) #b r*c*  
            merged_tensor = torch.cat([new_grids, blocks], dim=1)
            merged_tensor = torch.cat([merged_tensor, block_lefts], dim=1)
            state_values=self.Critic_net(merged_tensor) #b 1
        return state_values
    def Actor_net_update(self,gridss,blockss,answerss,maskss):
        b, r, c, fea = gridss.shape
        gridss[:, :, :, 0] = gridss[:, :, :, 0] / 500.0
        blockss[:, :, 0] = blockss[:, :, 0] / 500.0
        grids = torch.tensor(gridss.reshape(-1,r,c,fea), dtype=torch.float32).to(device)
        answers=torch.tensor(answerss.reshape(-1,1),dtype=torch.int64).to(device)
        blocks = torch.tensor(blockss, dtype=torch.float32).to(device)
        masks = torch.tensor(maskss.reshape(b, r * c, 1), dtype=torch.float32).to(device)
        pi_cal, _,logits = self.Locate(grids, blocks, masks, answers)  # b,1

        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits.squeeze(-1), answers.squeeze(-1))
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()


        return loss.item()

    def update(self, gridss, blockss,block_leftss, actionss, rewardss,doness, maskss, probss, ep_len, step1, model_dir):
        ave_loss = 0
        en_loss = 0
        v_loss = 0
        p_loss = 0
        b,r,c,fea=gridss.shape
        #ep_len
        #gridss (b,r,c,fea)
        #blockss (b,num_overhead,fea)
        #actions (b)
        #rewards (b)
        #dones (b)
        #masks (b,r*c,1)
        #probs (b)
        gridss[:,:,:,0]=gridss[:,:,:,0]/500.0
        blockss[:,:,0]=blockss[:,:,0]/500.0
        grids=torch.tensor(gridss,dtype=torch.float32).to(device)
        block_leftss=torch.tensor(block_leftss.reshape(-1,1),dtype=torch.float32).to(device) #b,1
        blocks=torch.tensor(blockss,dtype=torch.float32).to(device)
        actions=torch.tensor(actionss.reshape(-1,1),dtype=torch.int64).to(device)
        probs = torch.tensor(probss.reshape(-1,1), dtype=torch.float32).to(device)
        rewards = torch.tensor(rewardss.reshape(-1,1), dtype=torch.float32).to(device)
        masks= torch.tensor(maskss.reshape(b,r*c,1),dtype=torch.float32).to(device)
        dones = torch.tensor(doness.reshape(-1,1), dtype=torch.float32).to(device)
        block_leftss=block_leftss/100.0
        # 0 0 0 1 0 0 0 0 1 len=4
        # 0 0 1 0 0 0 0 1 
        # 0 0 0 0 1 len=5

        temp_state_values=self.calculate_v(grids.reshape(-1,r,c,fea),blocks.reshape(b,-1,fea),block_leftss) #b,1
        state_values=temp_state_values # b,1
        next_state_values=torch.cat([temp_state_values[1:],temp_state_values[[0]]],dim=0) # b,1
        td_target = rewards + self.gamma * next_state_values * (1-dones) #b,1
        delta = td_target - state_values
        pi_cal,_,_=self.Locate(grids,blocks,masks,actions) #b,1
        
        advantage_lst = np.zeros((b,1))
        advantage_lst = torch.tensor(advantage_lst, dtype=torch.float32).to(device)
        i = 0
        for ep in ep_len:
            advantage = 0.0
            
            for t in reversed(range(i, i + ep)):
                advantage = self.gamma * self.lmbda * advantage + delta[t][0]
                advantage_lst[t][0] = advantage
            i += ep    
            
        ratio = torch.exp(torch.log(pi_cal) - torch.log(probs))  # a/b == exp(log(a)-log(b))

        surr1 = ratio * advantage_lst
        surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage_lst
        loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(state_values, td_target.detach()) * self.alpha

        ave_loss = loss.mean().item()
        v_loss = (self.alpha * F.smooth_l1_loss(state_values, td_target.detach())).item()
        p_loss = -torch.min(surr1, surr2).mean().item()

        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()
        if step1 % 1 == 0:
            torch.save({
                'model_state_dict': self.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),

            }, model_dir+'trained_model' + str(step1) + '.pth')

        return ave_loss, v_loss, p_loss






