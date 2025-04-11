import numpy as np
import random
from Modified_Retrieval2 import *
from Location_GNN import *
from Network_DAN import*
import math
from Location_Heuristic import *
import vessl
from collections import OrderedDict
# stockyard
# block_number, storaged, stocktime, weight(TP_type_one_hot), created time, source, stock, sink, stage
# block
# block_number, stocktime, weigth(TP_type_one_hot), created time, source, stock, sink, stage


# small problem
# problem, block: 40 pl:10 tp: 8
# 0 factory number
# 1 yard number
# 2 yard size
# 3 block number distribution per init yard
# 4 source block per day
# 5 storage_period_scale
# 6 ready high
# 7 gap
# 8 tardy high
# 9 TP capacity type
# 10 TP number
# 11 Weight distribution
# 12 Dis high
# 13 Dis low
# 14 TP speed
# 15 RT weight 15
# 16 RT time 30

class Simulate_yard:
    def __init__(self, input_list, PPO, Placement):
        self.factory_num = input_list[0]
        self.stock_yard_num = input_list[1]
        self.yard_size = input_list[2]  # (5,5)
        self.init_block_distribution = input_list[3]
        self.block_per_day = input_list[4]
        self.storage_period_scale = input_list[5]
        self.ready_high = input_list[6]  # 250
        self.gap = input_list[7]
        self.tardy_high = input_list[8]  # 300
        self.TP_weight_capacity = input_list[9]
        self.TP_number = input_list[10]
        self.weight_distribution = input_list[11]
        self.TP_capacity_type_length = len(self.TP_weight_capacity)
        self.stock_yard = np.zeros(
            (self.stock_yard_num, self.yard_size[0], self.yard_size[1], 8 + self.TP_capacity_type_length))
        self.total_reward = 0
        self.dis_high = input_list[12]  # 3500
        self.dis_low = input_list[13]  # 500
        self.location_number = self.factory_num + self.stock_yard_num
        upper_tri = np.random.uniform(self.dis_low, self.dis_high, (self.location_number, self.location_number))
        upper_tri = np.triu(upper_tri, 1)  # 대각선 아래 제거
        symmetric_matrix = upper_tri + upper_tri.T
        np.fill_diagonal(symmetric_matrix, 0)
        self.Dis = symmetric_matrix.copy()

        self.block_count = 0
        self.TP_speed = input_list[14]  # 120
        self.RT_weight = input_list[15]  # 15
        self.RT_time = input_list[16]
        self.stockyard_filled = np.zeros(self.stock_yard_num)
        self.lookahead_num = 1
        self.ppo = PPO
        self.Placement = Placement
        self.Total_block = []

    def one_hot_encode(self, weight, thresholds):
        return [1 if weight < t else 0 for t in thresholds]

    def Create_problem(self, simulation_day):
        self.stock_yard[:, :, :, :] = 0
        self.Total_block = []
        self.total_reward = 0
        self.block_count = 0
        temp_yard = -np.ones((self.stock_yard_num, self.yard_size[0] * self.yard_size[1]))
        yard_level = np.zeros(self.stock_yard_num)
        for yard_num in range(self.stock_yard_num):
            init_block_num = random.randint(self.init_block_distribution[0], self.init_block_distribution[1])
            self.stockyard_filled[yard_num] = init_block_num
            positions = np.random.choice(self.yard_size[0] * self.yard_size[1], init_block_num, replace=False)
            for in_block in range(init_block_num):
                x, y = divmod(positions[in_block], self.yard_size[0])
                time = int(np.random.exponential(scale=self.storage_period_scale))
                weight = np.random.randint(self.weight_distribution[0], self.weight_distribution[1])
                sink = np.random.randint(0, self.factory_num)
                yard_level[yard_num] += 1
                temp_yard[yard_num, int(yard_level[yard_num])] = time
                self.block_count += 1
                self.stock_yard[yard_num, x, y, 0] = self.block_count
                self.stock_yard[yard_num, x, y, 1] = 1
                self.stock_yard[yard_num, x, y, 2] = time
                embedded_weight = self.one_hot_encode(weight, self.TP_weight_capacity)
                self.stock_yard[yard_num, x, y, 3:3 + self.TP_capacity_type_length] = embedded_weight
                self.stock_yard[yard_num, x, y, 3 + self.TP_capacity_type_length] = 0
                self.stock_yard[yard_num, x, y, 4 + self.TP_capacity_type_length] = -1
                self.stock_yard[yard_num, x, y, 5 + self.TP_capacity_type_length] = yard_num + self.factory_num
                self.stock_yard[yard_num, x, y, 6 + self.TP_capacity_type_length] = sink
                self.stock_yard[yard_num, x, y, 7 + self.TP_capacity_type_length] = 1
        for day in range(simulation_day):
            created_block, yard_level, temp_yard = self.source(day, temp_yard, yard_level)
            self.Total_block.append(created_block)
            temp_yard = -np.sort(-temp_yard, axis=1)
            temp_yard = np.where(temp_yard <= 100, -1, temp_yard)
            yard_level = np.sum(temp_yard >= 0, axis=1)
            temp_yard = np.maximum(temp_yard - 100, -1)
            #print(yard_level)
        return self.stock_yard, self.Total_block

    def Run_simulation(self, simulation_day, scheduling_mode, init_yard=None, init_block=None, batch_step=10):
        if init_yard is None and init_block is None:
            _, _ = self.Create_problem(simulation_day)
        else:
            self.stock_yard = init_yard
            self.Total_block = init_block

        save_yard=self.stock_yard.copy()
        save_block=self.Total_block.copy()
        ave_reward = 0
        ave_ett = 0
        ave_tardy = 0
        ave_rt = 0
        action_list = []
        prob_list = []
        reward_list = []
        done_list = []
        data = []

        for btch in range(batch_step):
            self.stock_yard=save_yard.copy()
            self.Total_block=save_block.copy()
            for day in range(simulation_day):
                reward_sum, tardy_sum, ett_sum, rt_sum, event, episode, actions, probs, rewards, dones = self.day_schedule_pp(day, scheduling_mode)
                data.append(episode)
                ave_reward += reward_sum.item()
                ave_ett += ett_sum
                ave_rt += rt_sum
                ave_tardy += tardy_sum
                action_list = np.concatenate((action_list, actions))
                prob_list = np.concatenate((prob_list, probs))
                reward_list = np.concatenate((reward_list, rewards))
                done_list = np.concatenate((done_list, dones))
                self.stock_yard[:, :, :, 2] = np.maximum(self.stock_yard[:, :, :, 2] - 100, 0)
        return data, reward_list, done_list, prob_list, action_list, ave_reward/simulation_day/batch_step

    def source(self, created_day, temp_yard, yard_level):
        created_block_num = np.random.randint(self.block_per_day[0], self.block_per_day[1])
        # block
        # block num, stocktime, weigth(TP_type_one_hot), created time, source, stock, sink, stage
        source_factory = np.random.choice(self.factory_num, created_block_num, replace=True)
        # 이거 결정 여부 추후 논의
        # stock_yard=np.random.choice(self.stock_yard_num, created_block_num, replace=True)
        ##
        sink_factory = np.random.choice(self.factory_num, created_block_num, replace=True)
        created_block = np.zeros((created_block_num, 7 + self.TP_capacity_type_length))
        for block_num in range(created_block_num):
            time = int(max(np.random.exponential(scale=self.storage_period_scale), 101))
            weight = np.random.randint(self.weight_distribution[0], self.weight_distribution[1])
            self.block_count += 1
            created_block[block_num, 0] = self.block_count
            created_block[block_num, 1] = time
            embedded_weight = self.one_hot_encode(weight, self.TP_weight_capacity)
            created_block[block_num, 2:2 + self.TP_capacity_type_length] = embedded_weight
            created_block[block_num, 2 + self.TP_capacity_type_length] = created_day
            created_block[block_num, 3 + self.TP_capacity_type_length] = source_factory[block_num]
            do = True
            while do or yard_level[stock_yard] == self.yard_size[0] * self.yard_size[1] - 1:
                do = False
                stock_yard = np.random.randint(0, self.stock_yard_num)
            # stock_yard=np.argmin(yard_level)
            yard_level[stock_yard] += 1
            temp_yard[stock_yard, int(yard_level[stock_yard])] = time
            created_block[block_num, 4 + self.TP_capacity_type_length] = stock_yard + self.factory_num
            created_block[block_num, 5 + self.TP_capacity_type_length] = sink_factory[block_num]
            created_block[block_num, 6 + self.TP_capacity_type_length] = 0
        return created_block, yard_level, temp_yard

    def cal_retrieval(self, block_codes):
        Retrieval_array = np.zeros((len(block_codes), self.TP_capacity_type_length))
        for e1, block_info in enumerate(block_codes):
            for TP_capacity in range(self.TP_capacity_type_length):

                b, x, y = np.argwhere(self.stock_yard[:, :, :, 0] == block_info)[0]

                count, ispossible = Count_retrieval(self.stock_yard[b, :, :, 2:3 + self.TP_capacity_type_length].copy(),
                                                    TP_capacity, [x, y])
                if not ispossible:
                    count = self.yard_size[0] * self.yard_size[1] / 3
                Retrieval_array[e1, TP_capacity] = count

        return Retrieval_array

    def day_schedule_pp(self, created_day, scheduling_mode):
        created_block = self.Total_block[created_day].copy()
        condition = (self.stock_yard[..., 1] == 1) & (self.stock_yard[..., 2] <= 100)
        # stockyard
        # block_number, storaged, stocktime, weight(TP_type_one_hot), created time, source, stock, sink, stage
        count = np.sum(condition)

        #print('Average_stockyard_filled',self.stock_yard[...,1].mean())
        #print('Stock yard',self.stock_yard[0,:,:,1].sum(),self.stock_yard[1,:,:,1].sum(),self.stock_yard[2,:,:,1].sum(),self.stock_yard[3,:,:,1].sum())

        '''
        print('placement block',len(created_block))
        print('retrieval_block',count)
        print('')
        '''
        block_retrieve = self.stock_yard[..., [-3, -2]][condition].copy()
        block_placement = created_block[:, 3 + self.TP_capacity_type_length:5 + self.TP_capacity_type_length]
        Block_located_num = len(created_block)
        Retrieval_array = self.cal_retrieval(self.stock_yard[..., 0][condition])

        Block = np.zeros((Block_located_num + count, 6 + 2 * self.TP_capacity_type_length))
        Block[:Block_located_num, :2] = block_placement
        Block[Block_located_num:, :2] = block_retrieve
        for i in range(Block_located_num + count):
            Block[i, 2] = self.Dis[
                              int(Block[i, 0]), int(Block[i, 1])] / self.TP_speed / self.tardy_high  # processing time
            Block[i, 3] = np.random.randint(0, self.ready_high) / self.tardy_high  # ready time
            Block[i, 4] = np.random.randint(Block[i, 3] + self.gap, self.tardy_high) / self.tardy_high - Block[
                i, 2]  # tardy time
        # Block[Block_located_num:,5:5+self.TP_capacity_type_length] 여기에 적치장 벨류를 넣을지 말지 고민
        Block[Block_located_num:,
        5:5 + self.TP_capacity_type_length] = Retrieval_array * self.RT_weight / self.tardy_high
        Block[Block_located_num:, 5 + self.TP_capacity_type_length:5 + 2 * self.TP_capacity_type_length] = \
        self.stock_yard[..., 3:3 + self.TP_capacity_type_length][condition].copy()
        Block[:Block_located_num,
        5 + self.TP_capacity_type_length:5 + 2 * self.TP_capacity_type_length] = created_block[:,
                                                                                 2:2 + self.TP_capacity_type_length]
        # factory, yard 순서
        Block[:Block_located_num, -1] = created_block[:, 0]
        Block[Block_located_num:, -1] = self.stock_yard[..., 0][condition].copy()

        # dep arr pr ready due Re Re We We Bn
        Block = Block[Block[:, 0].argsort()]
        unique_values, counts = np.unique(Block[:, 0], return_counts=True)
        max_count = np.max(counts)
        edge_fea_idx = -np.ones((self.location_number, max_count))
        edge_fea = np.zeros((self.location_number, max_count, 4 + 2 * self.TP_capacity_type_length))
        step = 0
        node_in_fea = np.zeros((self.location_number, 2 * self.TP_capacity_type_length + 1))
        step_to_ij = np.zeros((self.location_number, max_count))
        for i in range(len(counts)):
            for j in range(max_count):
                if j < counts[i]:
                    edge_fea_idx[int(unique_values[i])][j] = int(Block[step, 1])
                    edge_fea[int(unique_values[i])][j] = Block[step, 2:].copy()
                    # edge_fea processing_time, ready_time, tardy_time, weight one hot encoding(self.Transporter_type) 3+self.Transporter_type
                    step_to_ij[int(unique_values[i])][j] = step
                    step += 1

        #
        for i in range(self.TP_capacity_type_length):
            node_in_fea[0, i * 2] = int(self.TP_number / self.TP_capacity_type_length)
        node_in_fea[self.factory_num:, -1] = self.stockyard_filled / self.yard_size[0] / self.yard_size[1]

        transporter = np.zeros((self.TP_number, 6))
        for i in range(self.TP_number):
            transporter[i, 0] = int(i * self.TP_capacity_type_length / self.TP_number)  # TP type
            transporter[i, 1] = 0  # TP heading point
            transporter[i, 2] = 0  # TP arrival left time
            transporter[i, 3] = 0  # empty travel time
            transporter[i, 4] = -1  # action i
            transporter[i, 5] = -1  # action j
        # print(np.round(Block,1))
        '''
        print('day start')
        for i in range(self.stock_yard_num):
            print(self.stock_yard[i,:,:,0])
            print('')
        '''
        reward_sum, tardy_sum, ett_sum, rt_sum, event, episode, actions, probs, rewards, dones = self.schedule(
            Block_located_num + count, self.TP_number, transporter, created_block, edge_fea_idx, node_in_fea, edge_fea,
            self.Dis, step_to_ij, scheduling_mode)
        '''
        print('day end')
        for i in range(self.stock_yard_num):
            print(self.stock_yard[i,:,:,0])
            print('')
        print('day end')
        for i in range(self.stock_yard_num):
            print(self.stock_yard[i,:,:,2])
            print('')
        '''
        return reward_sum, tardy_sum, ett_sum, rt_sum, event, episode, actions, probs, rewards, dones

    def Create_mask(self, grid, TP_capa):
        r, c, f = grid.shape
        mask = (grid[:, :, 3:3 + self.TP_capacity_type_length].sum(axis=2) > 0).astype(
            np.uint8)  # 첫 번째 열이 0 초과인 위치를 1로 설정
        mask = mask[:, :, np.newaxis].copy()
        mask = mask.reshape(r, c)
        rows, cols = len(mask), len(mask[0])
        visited = [[False] * cols for _ in range(rows)]  # 방문 여부 기록
        new_grid = [[1] * cols for _ in range(rows)]  # 모든 값을 1로 초기화

        # BFS를 위한 큐
        queue = deque()

        # Step 1: 첫 번째 행에서 0을 찾고 BFS 시작
        for x in range(cols):
            if mask[0][x] == 0:
                queue.append((0, x))
                visited[0][x] = True  # 방문 체크
                new_grid[0][x] = 0  # 그대로 유지

        # BFS 탐색 (상, 하, 좌, 우)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while queue:
            y, x = queue.popleft()

            for dy, dx in directions:
                ny, nx = y + dy, x + dx

                if 0 <= ny < rows and 0 <= nx < cols and not visited[ny][nx] and mask[ny][nx] == 0:
                    queue.append((ny, nx))
                    visited[ny][nx] = True
                    new_grid[ny][nx] = 0  # 유지
        new_grid = np.array(new_grid)
        need_retrieve = False
        if new_grid[0].sum() == cols:
            need_retrieve = True
            check_list = np.argwhere(mask == 0)
            check_num = np.zeros(len(check_list))
            valid = False
            for e, space in enumerate(check_list):
                count, ispossible = Count_retrieval(grid[:, :, 2:3 + self.TP_capacity_type_length].copy(), TP_capa,
                                                    space.copy())
                check_num[e] = count
                if not ispossible:
                    check_num[e] = np.inf
                else:
                    valid = True
            if not valid:
                return 0, 0, False
            min_value = check_num.min()
            index = np.argwhere(check_num == min_value).flatten()
            for i in index:
                new_grid[check_list[i, 0], check_list[i, 1]] = 0
        return new_grid, need_retrieve, True

    def schedule(self, B, T, transporter, block_created, edge_fea_idx, node_fea, edge_fea, dis, step_to_ij, mode):
        transporter = transporter.copy()
        block_created = block_created.copy()
        edge_fea_idx = edge_fea_idx.copy()
        node_fea = node_fea.copy()
        edge_fea = edge_fea.copy()
        event = []
        unvisited_num = B
        node_fea = torch.tensor(node_fea, dtype=torch.float32).to(device)
        edge_fea = torch.tensor(edge_fea, dtype=torch.float32).to(device)
        edge_fea_idx = torch.tensor(edge_fea_idx, dtype=torch.long).to(device)

        N = edge_fea_idx.shape[0]
        M = edge_fea_idx.shape[1]
        episode = []  # torch node_fea (9,13), edge_fea (9,3,5), edge_fea_idx(9,3), distance (9,3)
        probs = np.zeros(B)
        rewards = np.zeros(B)
        dones = np.ones(B)
        actions = np.zeros(B)
        tardiness = 0
        reward_sum = 0
        tardy_sum = 0
        ett_sum = 0
        rt_sum = 0
        step = 0
        time = 0
        prob = 0
        num_valid_coords = 10
        mask = np.ones((N, M, 1))
        agent = np.random.randint(0, int(T))  # 랜덤 트랜스포터 부터 지정
        node_fea[int(transporter[agent][1])][int(transporter[agent][0]) * 2] -= 1
        # for _ in range(3):
        while unvisited_num > 0:
            # print(unvisited_num)
            # transporter (T,3) TP type / heading point / TP arrival time
            start_location = transporter[agent][1]
            distance = torch.tensor(dis[int(start_location)] / self.TP_speed / self.tardy_high,
                                    dtype=torch.float32).unsqueeze(1).repeat(1, edge_fea_idx.shape[1]).to(
                device)  # (n, e)

            if mode == 'RL_full':  # tp type=2
                valid_coords = ((edge_fea_idx >= 0) & (1 == edge_fea[:, :, 3 + self.TP_capacity_type_length + int(
                    transporter[agent][0])])).nonzero(as_tuple=False)
                # mask 초기화
                mask = torch.ones((N, M, 1), device=edge_fea.device)
                # 벡터화로 mask 업데이트
                mask[valid_coords[:, 0], valid_coords[:, 1], 0] = 0
                temp_mask=mask.clone()
                # 가능한 반출 및 적치 마스킹
                if transporter[agent][0] < self.TP_capacity_type_length - 1:
                    for crd in valid_coords:
                        if crd[0] >= self.factory_num:  # 반출
                            target_block_code = edge_fea[crd[0], crd[1], -1].item()

                            b, x, y = np.argwhere(self.stock_yard[:, :, :, 0] == target_block_code)[0]

                            count, ispossible = Count_retrieval(
                                self.stock_yard[b, :, :, 2:3 + self.TP_capacity_type_length].copy(),
                                transporter[agent][0], [x, y].copy())
                            if not ispossible:
                                mask[crd[0], crd[1], 0] = 1
                        else:  # 적치
                            target_yard = edge_fea_idx[crd[0], crd[1]].item() - self.factory_num
                            new_grid, need_retrieve, valid = self.Create_mask(self.stock_yard[target_yard].copy(),
                                                                              TP_capa=transporter[agent][0])
                            if not valid:
                                mask[crd[0], crd[1], 0] = 1
                if mask.sum().item()==N*M:
                    mask=temp_mask.clone()
                episode.append(
                    [node_fea.clone(), edge_fea[:, :, :-1].clone(), edge_fea_idx.clone(), distance.clone(),
                     transporter[agent][0],
                     mask])

                action, i, j, prob = self.ppo.get_action(node_fea, edge_fea[:, :, :-1], edge_fea_idx, mask, distance,
                                                         transporter[agent][0])

            elif mode == 'RL_RHR':
                # masking action
                valid_coords = ((edge_fea_idx >= 0) & (0 == edge_fea[:, :, 3 + int(transporter[agent][0])])).nonzero()
                pt_average = np.zeros(valid_coords.shape[0])
                st_average = np.zeros(valid_coords.shape[0])
                pt = np.zeros(valid_coords.shape[0])
                for i in range(valid_coords.shape[0]):
                    n = valid_coords[i][0]
                    e = valid_coords[i][1]
                    pt_average[i] = edge_fea[n, e, 0]
                    st_average[i] = dis[int(start_location)][n] / 120 / tardy_high
                pt_a = pt_average.mean()
                st_a = st_average.mean()
                pri = np.zeros((6, valid_coords.shape[0]))
                mask = np.ones((N, M, 1))
                action_list = []
                for i in range(valid_coords.shape[0]):
                    n = valid_coords[i][0]
                    e = valid_coords[i][1]
                    pri[0][i] = max(dis[int(start_location)][n] / 120 / tardy_high, edge_fea[n, e, 1].item()) + \
                                edge_fea[n, e, 0].item()
                    pri[1][i] = dis[int(start_location)][n] / 120 / tardy_high
                    pri[2][i] = edge_fea[n, e, 1].item()
                    st = dis[int(start_location)][n] / 120 / tardy_high
                    pri[3][i] = -(1 / edge_fea[n, e, 0] * math.exp(-max(edge_fea[n, e, 2], 0) / pt_a) * math.exp(
                        -st / st_a)).item()
                    pri[4][i] = edge_fea[n, e, 2].item()
                    pri[5][i] = -(1 / edge_fea[n, e, 0] * (1 - (edge_fea[n, e, 2] / edge_fea[n, e, 0]))).item()
                for i in range(6):
                    value = np.unique(pri[i])
                    value1 = value[0]
                    for j in np.where(value1 == pri[i])[0]:
                        n = valid_coords[j][0].item()
                        e = valid_coords[j][1].item()
                        mask[n, e, 0] = 0
                    if len(value) > 1:
                        value2 = value[1]
                        for j in np.where(value2 == pri[i])[0]:
                            n = valid_coords[j][0].item()
                            e = valid_coords[j][1].item()
                            mask[n, e, 0] = 0

                mask = torch.tensor(mask).to(device)
                episode.append(
                    [node_fea.clone(), edge_fea.clone(), edge_fea_idx.clone(), distance.clone(), transporter[agent][0],
                     mask])

                action, i, j, prob = ppo.get_action(node_fea, edge_fea, edge_fea_idx, mask, distance,
                                                    transporter[agent][0])

            elif mode == 'RL_HR':
                # masking action
                valid_coords = ((edge_fea_idx >= 0) & (0 == edge_fea[:, :, 3 + int(transporter[agent][0])])).nonzero()

                pri = np.zeros((6, valid_coords.shape[0]))
                mask = np.ones((N, M, 1))
                action_list = []
                for i in range(valid_coords.shape[0]):
                    n = valid_coords[i][0]
                    e = valid_coords[i][1]
                    pri[0][i] = max(dis[int(start_location)][n] / 120 / tardy_high, edge_fea[n, e, 1].item()) + \
                                edge_fea[n, e, 0].item()
                    pri[1][i] = dis[int(start_location)][n] / 120 / tardy_high
                    pri[2][i] = edge_fea[n, e, 1].item()
                    pri[3][i] = -(1 / edge_fea[n, e, 0] * torch.exp(
                        -(edge_fea[n, e, 2]) / (torch.sum(edge_fea[:, :, 0]) / valid_coords.shape[0]))).item()
                    pri[4][i] = edge_fea[n, e, 2].item()
                    pri[5][i] = -(1 / edge_fea[n, e, 0] * (1 - (edge_fea[n, e, 2] / edge_fea[n, e, 0]))).item()
                for i in range(6):
                    value = np.unique(pri[i])
                    value1 = value[0]
                    for j in np.where(value1 == pri[i])[0]:
                        n = valid_coords[j][0].item()
                        e = valid_coords[j][1].item()
                        mask[n, e, 0] = 0
                mask = torch.tensor(mask).to(device)
                episode.append(
                    [node_fea.clone(), edge_fea.clone(), edge_fea_idx.clone(), distance.clone(), transporter[agent][0],
                     mask])

                action, i, j, prob = ppo.get_action(node_fea, edge_fea, edge_fea_idx, mask, distance,
                                                    transporter[agent][0])
            elif mode == 'Random':
                valid_coords = ((edge_fea_idx >= 0) & (1 == edge_fea[:, :, 3 + self.TP_capacity_type_length + int(
                    transporter[agent][0])])).nonzero()
                num_valid_coords = valid_coords.shape[0]
                action = random.randint(0, num_valid_coords - 1)
                i = valid_coords[action][0].item()
                j = valid_coords[action][1].item()

            elif mode == 'SSPT':  # PDR
                valid_coords = ((edge_fea_idx >= 0) & (1 == edge_fea[:, :, 3 + self.TP_capacity_type_length + int(
                    transporter[agent][0])])).nonzero()
                pt = np.zeros(valid_coords.shape[0])
                for i in range(valid_coords.shape[0]):
                    n = valid_coords[i][0]
                    e = valid_coords[i][1]
                    pt[i] = max(dis[int(start_location)][n] / self.TP_speed / self.tardy_high,
                                edge_fea[n, e, 1].item()) + edge_fea[n, e, 0].item()
                min_index = np.argmin(pt)  # 가장 작은 값의 인덱스 찾기

                # 같은 값이 여러 개인 경우 처리
                min_value = pt[min_index]
                same_value_indices = np.where(pt == min_value)[0]

                # 같은 값이 하나 이상인 경우
                if len(same_value_indices) > 1:
                    min_index = np.random.choice(same_value_indices)
                action = min_index
                i = valid_coords[action][0].item()
                j = valid_coords[action][1].item()

            elif mode == 'SET':

                valid_coords = ((edge_fea_idx >= 0) & (1 == edge_fea[:, :, 3 + self.TP_capacity_type_length + int(
                    transporter[agent][0])])).nonzero()
                pt = np.zeros(valid_coords.shape[0])
                for i in range(valid_coords.shape[0]):
                    n = valid_coords[i][0]
                    e = valid_coords[i][1]
                    pt[i] = dis[int(start_location)][n] / self.TP_speed / self.tardy_high
                min_index = np.argmin(pt)  # 가장 작은 값의 인덱스 찾기

                # 같은 값이 여러 개인 경우 처리
                min_value = pt[min_index]
                same_value_indices = np.where(pt == min_value)[0]

                # 같은 값이 하나 이상인 경우
                if len(same_value_indices) > 1:
                    min_index = np.random.choice(same_value_indices)
                action = min_index
                i = valid_coords[action][0].item()
                j = valid_coords[action][1].item()

            elif mode == 'SRT':
                valid_coords = ((edge_fea_idx >= 0) & (1 == edge_fea[:, :, 3 + self.TP_capacity_type_length + int(
                    transporter[agent][0])])).nonzero()
                pt = np.zeros(valid_coords.shape[0])
                for i in range(valid_coords.shape[0]):
                    n = valid_coords[i][0]
                    e = valid_coords[i][1]
                    pt[i] = edge_fea[n, e, 1].item()
                min_index = np.argmin(pt)  # 가장 작은 값의 인덱스 찾기

                # 같은 값이 여러 개인 경우 처리
                min_value = pt[min_index]
                same_value_indices = np.where(pt == min_value)[0]

                # 같은 값이 하나 이상인 경우
                if len(same_value_indices) > 1:
                    min_index = np.random.choice(same_value_indices)
                action = min_index
                i = valid_coords[action][0].item()
                j = valid_coords[action][1].item()

            elif mode == 'ATCS':
                valid_coords = ((edge_fea_idx >= 0) & (1 == edge_fea[:, :, 3 + self.TP_capacity_type_length + int(
                    transporter[agent][0])])).nonzero()
                pt_average = np.zeros(valid_coords.shape[0])
                st_average = np.zeros(valid_coords.shape[0])
                pt = np.zeros(valid_coords.shape[0])
                for i in range(valid_coords.shape[0]):
                    n = valid_coords[i][0]
                    e = valid_coords[i][1]
                    pt_average[i] = edge_fea[n, e, 0]
                    st_average[i] = dis[int(start_location)][n] / self.TP_speed / self.tardy_high
                pt_a = pt_average.mean()
                st_a = st_average.mean()

                for i in range(valid_coords.shape[0]):
                    n = valid_coords[i][0]
                    e = valid_coords[i][1]
                    st = dis[int(start_location)][n] / self.TP_speed / self.tardy_high
                    pt[i] = (1 / edge_fea[n, e, 0] * math.exp(-max(edge_fea[n, e, 2], 0) / pt_a) * math.exp(
                        -st / st_a)).item()
                max_index = np.argmax(pt)  # 가장 작은 값의 인덱스 찾기

                # 같은 값이 여러 개인 경우 처리
                max_value = pt[max_index]
                same_value_indices = np.where(pt == max_value)[0]

                # 같은 값이 하나 이상인 경우
                if len(same_value_indices) > 1:
                    max_index = np.random.choice(same_value_indices)
                action = max_index
                i = valid_coords[action][0].item()
                j = valid_coords[action][1].item()

            elif mode == 'MDD':
                valid_coords = ((edge_fea_idx >= 0) & (1 == edge_fea[:, :, 3 + self.TP_capacity_type_length + int(
                    transporter[agent][0])])).nonzero()
                pt = np.zeros(valid_coords.shape[0])
                for i in range(valid_coords.shape[0]):
                    n = valid_coords[i][0]
                    e = valid_coords[i][1]
                    pt[i] = edge_fea[n, e, 2].item()
                min_index = np.argmin(pt)  # 가장 작은 값의 인덱스 찾기

                # 같은 값이 여러 개인 경우 처리
                min_value = pt[min_index]
                same_value_indices = np.where(pt == min_value)[0]

                # 같은 값이 하나 이상인 경우
                if len(same_value_indices) > 1:
                    min_index = np.random.choice(same_value_indices)
                action = min_index
                i = valid_coords[action][0].item()
                j = valid_coords[action][1].item()

            elif mode == 'COVERT':
                valid_coords = ((edge_fea_idx >= 0) & (1 == edge_fea[:, :, 3 + self.TP_capacity_type_length + int(
                    transporter[agent][0])])).nonzero()
                pt = np.zeros(valid_coords.shape[0])
                for i in range(valid_coords.shape[0]):
                    n = valid_coords[i][0]
                    e = valid_coords[i][1]
                    pt[i] = -(1 / edge_fea[n, e, 0] * (1 - (edge_fea[n, e, 2] / edge_fea[n, e, 0]))).item()
                min_index = np.argmin(pt)  # 가장 작은 값의 인덱스 찾기

                # 같은 값이 여러 개인 경우 처리
                min_value = pt[min_index]
                same_value_indices = np.where(pt == min_value)[0]

                # 같은 값이 하나 이상인 경우
                if len(same_value_indices) > 1:
                    min_index = np.random.choice(same_value_indices)
                action = min_index
                i = valid_coords[action][0].item()
                j = valid_coords[action][1].item()

            transporter, edge_fea_idx, node_fea, edge_fea, event_list, ett, td, rt = self.do_action(transporter,
                                                                                                    edge_fea_idx.clone(),
                                                                                                    node_fea.clone(),
                                                                                                    edge_fea.clone(),
                                                                                                    agent, i, j, dis,
                                                                                                    time,
                                                                                                    step_to_ij,
                                                                                                    block_created, mask)
            if unvisited_num == 1:
                event_list.append(round(td, 3))
                event_list.append(round(ett, 3))
                event_list.append(round(td + ett, 3))
                event.append(event_list)
                tardy_sum += td
                ett_sum += ett
                rt_sum += rt
                reward = ett + td + rt
                reward_sum += reward
                actions[step] = action
                probs[step] = prob
                dones[step] = 0
                rewards[step] = reward
                episode.append(
                    [node_fea.clone(), edge_fea[:, :, :-1].clone(), edge_fea_idx.clone(), distance.clone(),
                     transporter[agent][0],
                     mask])
                break
            sw = 0  # do while

            temp_tardy = 0

            while (((num_valid_coords <= 0) | (sw == 0))):
                sw = 1

                next_agent, mintime = self.select_agent(transporter)

                transporter, edge_fea_idx, node_fea, edge_fea, tardiness, tardy = self.next_state(
                    transporter, edge_fea_idx, node_fea, edge_fea, tardiness, mintime, next_agent)
                agent = next_agent
                temp_tardy += tardy
                time += mintime

                valid_coords = ((edge_fea_idx >= 0) & (1 == edge_fea[:, :, 3 + self.TP_capacity_type_length + int(
                    transporter[agent][0])])).nonzero()
                num_valid_coords = valid_coords.shape[0]
                if num_valid_coords == 0:
                    transporter[agent][2] = float("inf")
            tardy_sum += td
            tardy_sum += temp_tardy
            rt_sum += rt
            ett_sum += ett

            reward = temp_tardy + ett + td + rt
            event_list.append(round(temp_tardy + td, 3))
            event_list.append(round(ett, 3))
            event_list.append(round(reward, 3))

            # event_list 현재 시간, ett,tardy,완료시간,tp,몇번,tardy,ett,reward

            event.append(event_list)
            actions[step] = action
            probs[step] = prob
            rewards[step] = reward
            unvisited_num -= 1

            reward_sum += reward
            step += 1

            # edge fea는 시간 /220 , 속도 100/80
            # dis 거리/4000
            # ready time이 0보다 작으면 0으로
            # tardiness는 그떄 발생한 정도 case 1,2,3 0보다 작으면

        # event_list 현재 시간, ett,tardy,완료시간,tp,몇번,tardy,ett,reward

        return reward_sum, tardy_sum, ett_sum, rt_sum, event, episode, actions, probs, rewards, dones

    # 각각 action과 next_state로 분리하자
    def do_action(self, transporter, edge_fea_idx, node_fea, edge_fea, agent, i, j, dis, time, step_to_ij, block_created, masked):

        # edge_fea      0                 1       2           3,4,                               5,6
        #          processing_time, ready_time, tardy_time, RT(self.Transporter_type) , weight one hot encoding(self.Transporter_type) 3+2*self.Transporter_type

        transporter[agent][3] = dis[int(transporter[agent][1]), i] / self.TP_speed / 1.5 / self.tardy_high
        ett = -dis[int(transporter[agent][1]), i] / self.TP_speed / 1.5 / self.tardy_high
        rt = -edge_fea[i, j, 3 + int(transporter[agent][0])].item()

        if i >= self.factory_num:  # 반출
            target_block_code = edge_fea[i, j, -1].item()
            b, x, y = np.argwhere(self.stock_yard[:, :, :, 0] == target_block_code)[0]
            # print('yard ',b)
            ispossible, rearrange_num, end_grid = Retrieval(self.stock_yard[b].copy(), transporter[agent][0], [x, y],
                                                            self.Placement, self.lookahead_num,
                                                            self.TP_capacity_type_length, 'Only retrieve')
            if not ispossible:
                ispossible, rearrange_num, end_grid = Retrieval(self.stock_yard[b].copy(),
                                                                self.TP_capacity_type_length - 1, [x, y],
                                                                self.Placement, self.lookahead_num,
                                                                self.TP_capacity_type_length, 'Only retrieve')
                #print('e1')
            self.stock_yard[b] = end_grid
        else:  # 적치
            target_block_code = edge_fea[i, j, -1].item()
            target_yard = edge_fea_idx[i][j].item() - self.factory_num
            mask, need_retrieval, possible = self.Create_mask(self.stock_yard[target_yard].copy(),
                                                              TP_capa=transporter[agent][0])
            if not possible:
                mask, need_retrieval, possible = self.Create_mask(self.stock_yard[target_yard].copy(),
                                                                  TP_capa=self.TP_capacity_type_length - 1)
                
            grid_tensor = torch.tensor(
                self.stock_yard[target_yard, :, :, 2:3 + self.TP_capacity_type_length].reshape(1, self.yard_size[0],
                                                                                               self.yard_size[1], -1),
                dtype=torch.float32).to(device)
            grid_tensor[:, :, 0] = grid_tensor[:, :, 0] / (500.0)
            # block
            # block_number, stocktime, weigth(TP_type_one_hot), created time, source, stock, sink, stage

            index = np.argwhere(block_created[:, 0] == target_block_code)

            index = index[0][0]
            blocks_vec = block_created[index, 1:2 + self.TP_capacity_type_length]
            block_tensor = torch.tensor(blocks_vec.reshape(1, self.lookahead_num, -1), dtype=torch.float32).to(device)
            block_tensor[:, :, 0] = block_tensor[:, :, 0] / (500.0)
            mask_tensor = torch.tensor(mask.reshape(1, -1, 1), dtype=torch.float32).to(device)

            prob, coord = self.Placement.Locate(grid_tensor, block_tensor, mask_tensor, ans=None)
            r = coord.item() // self.stock_yard[target_yard].shape[0]
            c = coord.item() % self.stock_yard[target_yard].shape[1]
            target_block = [r, c]
            # 적치
            # print('yard ',target_yard)
            # stockyard
            # block_number, storaged, stocktime, weight(TP_type_one_hot), created time, source, stock, sink, stage

            # print(self.stock_yard[target_yard,:,:,0])
            # print(block_created[index,0],' block stocked 예정 at',r,c)

            self.stock_yard[target_yard, r, c, 0] = block_created[index, 0]
            self.stock_yard[target_yard, r, c, 1] = 1
            self.stock_yard[target_yard, r, c, 2:] = block_created[index, 1:]
            self.stock_yard[target_yard, r, c, -1] = 1
            # print(self.stock_yard[target_yard,:,:,0])

            if need_retrieval:
                ispossible, rearrange_num, end_grid = Retrieval(self.stock_yard[target_yard].copy(),
                                                                transporter[agent][0], target_block.copy(),
                                                                self.Placement, self.lookahead_num,
                                                                self.TP_capacity_type_length, 'Not only retrieve')
                rt = -rearrange_num * self.RT_weight / self.tardy_high
                self.stock_yard[target_yard] = end_grid.copy()

        td = min(edge_fea[i, j, 2].item() - dis[
            int(transporter[agent][1]), i] / self.TP_speed / 1.5 / self.tardy_high - rt * self.RT_weight / self.RT_time,
                 0) - min(edge_fea[i, j, 2].item(), 0)
        transporter[agent][2] = (max(dis[int(
            transporter[agent][1]), i] / self.TP_speed / 1.5 / self.tardy_high + rt * self.RT_weight / self.RT_time,
                                     edge_fea[i][j][1].item()) + edge_fea[i][j][0].item())
        transporter[agent][1] = edge_fea_idx[i][j].item()
        transporter[agent][4] = i
        transporter[agent][5] = j
        event_list = [round(time, 3), round(transporter[agent][3] + time, 3),
                      round(edge_fea[i][j][2].item() + time + edge_fea[i][j][0].item(), 3),
                      round(transporter[agent][2] + time, 3), agent,
                      step_to_ij[i][j]]  # event_list 현재 시간, ett 끝 시간 ,tardy 끝 시간 ,완료 시간,tp, 몇번

        # 1 TP heading point
        # 2 TP arrival left time
        # 3 empty travel time
        # 4 action i
        # 5 action j

        node_fea[int(transporter[agent][1])][int(transporter[agent][0]) * 2 + 1] = (node_fea[
                                                                                        int(transporter[agent][1])][int(
            transporter[agent][0]) * 2 + 1] * node_fea[int(transporter[agent][1])][int(transporter[agent][0]) * 2] +
                                                                                    transporter[agent][2]) / (node_fea[
                                                                                                                  int(
                                                                                                                      transporter[
                                                                                                                          agent][
                                                                                                                          1])][
                                                                                                                  int(
                                                                                                                      transporter[
                                                                                                                          agent][
                                                                                                                          0]) * 2] + 1)
        node_fea[int(transporter[agent][1])][int(transporter[agent][0]) * 2] += 1
        edge_fea_idx[i][j] = -1
        if i >= self.factory_num:  # 반출 그래프 업데이트
            node_fea[i][-1] = node_fea[i][-1] - 1.0 / self.yard_size[0] / self.yard_size[1]
            for e, edge in enumerate(edge_fea[i]):
                if edge_fea_idx[i][e] != -1:
                    for TP_capacity in range(self.TP_capacity_type_length):

                        b, x, y = np.argwhere(self.stock_yard[:, :, :, 0] == edge_fea[i, e, -1].item())[0]
                        # edge_fea      0                 1       2           3,4,                               5,6
                        #          processing_time, ready_time, tardy_time, RT(self.Transporter_type) , weight one hot encoding(self.Transporter_type) 3+2*self.Transporter_type
                        count, ispossible = Count_retrieval(
                            self.stock_yard[b, :, :, 2:3 + self.TP_capacity_type_length], TP_capacity, [x, y])
                        if not ispossible:
                            count = self.yard_size[0] * self.yard_size[1] / 3
                        edge_fea[i, e, 3 + TP_capacity] = count * self.RT_weight / self.tardy_high

        else:  # 적치
            node_fea[int(transporter[agent][1])][-1] = node_fea[int(transporter[agent][1])][-1] + 1.0 / self.yard_size[
                0] / self.yard_size[1]

            for e, edge in enumerate(edge_fea[int(transporter[agent][1])]):
                if edge_fea_idx[int(transporter[agent][1])][e] != -1:
                    for TP_capacity in range(self.TP_capacity_type_length):
                        b, x, y = \
                        np.argwhere(self.stock_yard[:, :, :, 0] == edge_fea[int(transporter[agent][1]), e, -1].item())[
                            0]
                        # edge_fea      0                 1       2           3,4,                               5,6
                        #          processing_time, ready_time, tardy_time, RT(self.Transporter_type) , weight one hot encoding(self.Transporter_type) 3+2*self.Transporter_type
                        count, ispossible = Count_retrieval(
                            self.stock_yard[b, :, :, 2:3 + self.TP_capacity_type_length], TP_capacity, [x, y])
                        if not ispossible:
                            count = self.yard_size[0] * self.yard_size[1] / 3
                        edge_fea[
                            int(transporter[agent][1]), e, 3 + TP_capacity] = count * self.RT_weight / self.tardy_high

        return transporter, edge_fea_idx, node_fea, edge_fea, event_list, ett, td, rt

    def next_state(self, transporter, edge_fea_idx, node_fea, edge_fea, tardiness, min_time, next_agent):
        transporter[:, 2] -= min_time

        # node_fea
        node_fea[:, [2 * i + 1 for i in range(self.TP_capacity_type_length)]] = node_fea[:, [2 * i + 1 for i in range(
            self.TP_capacity_type_length)]] - min_time
        node_fea[node_fea < 0] = 0
        if node_fea[int(transporter[next_agent][1]), int(transporter[next_agent][0]) * 2] > 1:
            node_fea[int(transporter[next_agent][1]), int(transporter[next_agent][0]) * 2 + 1] = node_fea[int(
                transporter[next_agent][1]), int(transporter[next_agent][0]) * 2 + 1] * (node_fea[
                int(transporter[next_agent][1]), int(transporter[next_agent][0]) * 2]) / (node_fea[int(
                transporter[next_agent][1]), int(transporter[next_agent][0]) * 2] - 1)
        node_fea[int(transporter[next_agent][1]), int(transporter[next_agent][0]) * 2] -= 1
        # edge_fea
        mask = torch.where(edge_fea_idx >= 0, torch.tensor(1.0), torch.tensor(0.0))
        edge_fea[:, :, [1, 2]] = edge_fea[:, :, [1, 2]] - mask.unsqueeze(2).repeat(1, 1, 2) * min_time
        edge_fea[:, :, 1][edge_fea[:, :, 1] < 0] = 0

        tardiness_next = edge_fea[:, :, 2][edge_fea[:, :, 2] < 0].sum().item()
        tardy = tardiness_next - tardiness
        # tardiness 수정, weight constraint 고려 ready time
        return transporter, edge_fea_idx, node_fea, edge_fea, tardiness_next, tardy

    def select_agent(self, transporter):
        event = transporter[:, 2]
        min_time = event.min()
        argmin = np.where((min_time == transporter[:, 2]) & (transporter[:, 0] == 0))[0]
        i = 0
        while len(argmin) == 0:
            i += 1
            argmin = np.where((min_time == transporter[:, 2]) & (transporter[:, 0] == i))[0]
        agent = int(random.choice(argmin))
        return agent, min_time

    def plot_gantt_chart(self, events, B, T):
        """

        # event_list 현재 시간, ett,tardy,완료시간,tp,몇번,tardy,ett,reward


        """

        # version 1:
        colorset = plt.cm.rainbow(np.linspace(0, 1, B))

        # Set up figure and axis
        fig, ax = plt.subplots()

        # Plot Gantt chart bars
        for event in events:
            job_start = event[0]
            empty_travel_end = event[1]
            ax.barh(y=event[4], width=empty_travel_end - job_start, left=job_start, height=0.6,
                    label=f'transporter {event[4] + 1}', color='grey')
            job_end = event[3]
            ax.barh(y=event[4], width=job_end - empty_travel_end, left=empty_travel_end, height=0.6,
                    label=f'transporter {event[4] + 1}', color=colorset[int(event[5])])
            # ax.text((job_start+empty_travel_end)/2, event[3], 'empty travel time',ha='center',fontsize=7,va='center')
            ax.text((empty_travel_end + job_end) / 2, event[4], 'Block' + str(int(event[5])), ha='center', fontsize=6,
                    va='center')

        # Customize the plot
        ax.set_xlabel('Time')
        ax.set_yticks(range(T))
        ax.set_yticklabels([f'transporter {i + 1}' for i in range(T)])

        # Show the plot
        plt.show()


if __name__=="__main__":

    problem_dir='/output/problem_set/'
    if not os.path.exists(problem_dir):
        os.makedirs(problem_dir)
    model_dir='/output/model/ppo/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    history_dir='/output/history/'
    if not os.path.exists(history_dir):
        os.makedirs(history_dir)
    device = 'cuda'
    # small problem
    input_list = [6, 4, (5, 5), (10, 12), (20, 21), 250, 100, 100, 300, [300, 500], 8, (1, 500), 3500, 500, 120, 20, 10]
    #input_list = [9, 6, (5, 5), (10, 12), (30, 31), 250, 100, 100, 300, [300, 500], 12, (1, 500), 3500, 500, 120, 20, 10]
    
    # middle problem
    # problem, block: 60 pl:15 tp: 12
    # 0 factory number
    # 1 yard number
    # 2 yard size
    # 3 block number distribution per init yard
    # 4 source block per day
    # 5 storage_period_scale
    # 6 ready high
    # 7 gap
    # 8 tardy high
    # 9 TP capacity type
    # 10 TP number
    # 11 Weight distribution
    # 12 Dis high
    # 13 Dis low
    # 14 TP speed
    # 15 RT weight 15
    # 16 RT time 30
    learning_rate = 0.001
    lmbda = 0.95
    gamma = 1
    discount_factor = 1
    epsilon = 0.2
    alpha = 0.5
    beta = 0.01
    location_num = 10
    lookahead_block_num = 1
    grid_size = (5, 5)

    hidden_dim = 32
    transporter_type = 2
    feature_dim = 1 + transporter_type
    ppo = 0
    #mod = 'GCN2'
    placement = Placement(feature_dim+1, hidden_dim, lookahead_block_num, grid_size, learning_rate, lmbda, gamma, alpha,beta, epsilon, 'GAT').to('cuda')
    checkpoint = torch.load('/input/Final_GAT_RL.pth', map_location=torch.device('cuda'))  # 파일에서 로드할 경우
    full_state_dict = checkpoint['model_state_dict']
    filtered_state_dict = OrderedDict({k: v for k, v in full_state_dict.items() if 'Critic_net' not in k})
    placement.load_state_dict(filtered_state_dict, strict=False)

    #placement = Heuristic(grid_size=(5,5),TP_type_len=transporter_type,mod='ASR')
    Simulation = Simulate_yard(input_list, ppo, placement)
    Simulation.ppo = PPO(learning_rate, lmbda, gamma, alpha, beta, epsilon, discount_factor, location_num,
                         transporter_type, Simulation.Dis)

    # small problem
    # problem, block: 40 pl:10 tp: 8
    # 0 factory number
    # 1 yard number
    # 2 yard size
    # 3 block number distribution per init yard
    # 4 source block per day
    # 5 storage_period_scale
    # 6 ready high
    # 7 gap
    # 8 tardy high
    # 9 TP capacity type
    # 10 TP number
    # 11 Weight distribution
    # 12 Dis high
    # 13 Dis low
    # 14 TP speed
    # 15 RT weight 15
    # 15 RT time 15
    scheduling_mode = 'RL_full'
    train_step=1000
    eval_num = 10
    eval_step = 40
    eval_set = []
    for _ in range(eval_num):
        eval_yard, eval_block = Simulation.Create_problem(10)
        eval_set.append([eval_yard.copy(), eval_block.copy()])

    K = 2
    for step in range(train_step):
        data, reward_list, done_list, prob_list, action_list, ave_reward = Simulation.Run_simulation(simulation_day=10,
                                                                                                     scheduling_mode=scheduling_mode,
                                                                                                     init_yard=None,
                                                                                                     init_block=None,
                                                                                                     batch_step=20)
        vessl.log(step=step, payload={'train_reward': ave_reward})
        print(step)
        for _ in range(K):
            ave_loss, v_loss, p_loss = Simulation.ppo.update(data, prob_list, reward_list, action_list, done_list, step,
                                                             model_dir)
        if step % eval_step == 0:
            eval_reward = 0
            for j in range(eval_num):
                data, reward_list, done_list, prob_list, action_list, ave_reward = Simulation.Run_simulation(
                    simulation_day=10, scheduling_mode=scheduling_mode, init_yard=eval_set[j][0].copy(),
                    init_block=eval_set[j][1].copy(), batch_step=5)
                eval_reward += ave_reward
            vessl.log(step=step, payload={'eval_reward': eval_reward / eval_num})

