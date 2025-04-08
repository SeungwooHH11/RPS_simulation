import numpy as np
import torch
from Modified_Retrieval2 import *

class Heuristic():
    def __init__(self, grid_size,TP_type_len,mod):
        super(Heuristic, self).__init__()
        self.grid_size=grid_size
        self.TP_type_len=TP_type_len
        self.mod=mod

    
    def find_coordinates(self, m,arr):
        # 3번째 차원의 첫 번째 채널이 m보다 작은 값의 인덱스 찾기
        indices = np.argwhere((arr[:, :, 0] < m) & (arr[:,:,1:].sum()>0))
        # [x, y] 형태로 변환
        return indices.tolist()
    def find_space(self,arr):
        indices = np.argwhere((arr[:, :, 0]==0))
        # [x, y] 형태로 변환
        return indices.tolist()
    
    def Locate(self,grid_tensor,block_tensor,mask_tensor,ans=None):
        r=self.grid_size[0]
        c=self.grid_size[1]
        grid=np.array(grid_tensor.cpu()).reshape(r,c,-1).copy()
        grid[:,:,0]=grid[:,:,0]*500.0
        block=np.array(block_tensor[:,0,:].cpu()).flatten().copy()
        block[0]=block[0]*500.0
        mask=np.array(mask_tensor.cpu()).reshape(r,c,1).copy()
        
        if self.mod=='ASR':
            target_exit=block[0]
            cal_block=self.find_coordinates(target_exit,grid)
            candidate_space=self.find_space(mask)
            
            score=[]
            if len(cal_block)==0:
                candidate_space=self.find_space(mask)
                best_x, best_y = max(candidate_space, key=lambda p: (p[0], -p[1]))
                return torch.tensor(1), torch.tensor(best_x*c+best_y)

            TP_capacity_num = int(block[1:].sum())  # 3 0,1,2 1 1,2
            for space in candidate_space:
                temp_grid=grid.copy()
                temp_grid[space[0],space[1],:]=block
                ave_score=0

                step=0

                for bl in cal_block:
                    for TP_capacity in range(self.TP_type_len-TP_capacity_num,self.TP_type_len):
                        count,ispossible=Count_retrieval(temp_grid,TP_capacity,bl.copy())
                        if ispossible:
                            ave_score+=count
                            step+=1
                score.append(ave_score/step)
            min_index=np.argmin(np.array(score))
            
            coord=candidate_space[min_index][0]*c+candidate_space[min_index][1]
            return torch.tensor(1),torch.tensor(coord)
        
        elif self.mod=='Random':
            candidate_space=self.find_space(mask)
            
            
            random_index=np.random.randint(0,len(candidate_space))
            return torch.tensor(1),torch.tensor(candidate_space[random_index][0]*c+candidate_space[random_index][1])
        elif self.mod=='BLF':
            candidate_space=self.find_space(mask)
            best_x, best_y = max(candidate_space, key=lambda p: (p[0], -p[1]))
            return torch.tensor(1), torch.tensor(best_x*c+best_y)
