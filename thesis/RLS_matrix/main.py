import numpy as np

from Simulation_DAN import *
from Network_DAN import *
import torch
import vessl
np.random.seed(1)
random.seed(1)
torch.manual_seed(1)

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

    device='cuda'
    block_number=40
    location_number=20
    transporter_type=2
    transporter_number=8
    dis_high=3000
    dis_low=500
    ready_high=100 
    tardy_high=300
    gap=100
    K_epoch=2
    Pr_sampler=Problem_sampling(block_number,location_number,transporter_type,transporter_number,dis_high,dis_low,ready_high,tardy_high,gap)
    temp_dis=dis_low/Pr_sampler.Dis
    indices = np.diag_indices(min(temp_dis.shape))
    temp_dis[indices] = 0
    
    dis=torch.tensor(temp_dis,dtype=torch.float32).to(device)
    ppo=PPO( learning_rate=0.001, lmbda=0.95, gamma=1, alpha=0.5, beta=0.01, epsilon=0.2, discount_factor=1,location_num=location_number,transporter_type=transporter_type,dis=dis)
    number_of_validation=20
    number_of_validation_batch=50
    number_of_problem=10 # 한번에 몇개의 문제를
    number_of_batch=80 # 문제당 몇 episode씩 한번에 학습할껀지
    number_of_trial=1  #1, 10, 100, 1000 #이를 몇번 반복할껀지
    number_of_iteration=int(1001/number_of_trial)  # 전체 iteration #iteration 단위로 문제 변화
    validation=[]
    validation_step = 10
    Control_result=np.zeros((20,7,6))
    history = np.zeros((number_of_iteration * number_of_trial,2))
    validation_history=np.zeros((int(1001/validation_step)+10,12))
    step = 0
    mode_list = ['Random', 'SSPT', 'SET', 'SRT', 'ATCS', 'MDD', 'COVERT']
    temp_step = 0
    past_time_step=0

    for j in range(number_of_validation):
        B, T, b, tp, efi, nf, ef, dis, step_to_ij = Pr_sampler.sample()
        efi = efi.astype('int')
        validation.append([B, T, tp, b, efi, nf, ef, dis, step_to_ij, tardy_high])

        for nu, mod in enumerate(mode_list):
            rs = np.zeros(20)
            es = np.zeros(20)
            ts = np.zeros(20)
            for k in range(20):
                reward_sum, tardy_sum, ett_sum, event, episode, actions, probs, rewards, dones = simulation(
                    validation[j][0], validation[j][1], validation[j][2], validation[j][3], validation[j][4], validation[j][5],
                    validation[j][6], validation[j][7], validation[j][8], validation[j][9], mod, ppo)
                rs[k] = reward_sum
                es[k] = ett_sum
                ts[k] = tardy_sum
            Control_result[temp_step, nu, 0] = rs.mean()
            Control_result[temp_step, nu, 1] = rs.var()
            Control_result[temp_step, nu, 2] = es.mean()
            Control_result[temp_step, nu, 3] = es.var()
            Control_result[temp_step, nu, 4] = ts.mean()
            Control_result[temp_step, nu, 5] = ts.var()
        temp_step += 1
    for nu, mod in enumerate(mode_list):
        print(mod, Control_result[past_time_step:temp_step, nu, 0].mean(),
              Control_result[past_time_step:temp_step, nu, 2].mean(),
              Control_result[past_time_step:temp_step, nu, 4].mean())

    for i in range(number_of_iteration):
        problem=[]
        temp_step=0
        for j in range(number_of_problem):
            B, T, b, tp, efi, nf, ef, dis, step_to_ij = Pr_sampler.sample()
            efi = efi.astype('int')
            problem.append([B, T, tp, b, efi, nf, ef, dis, step_to_ij, tardy_high])

            

        for k in range(number_of_trial):
            ave_reward = 0
            ave_tardy = 0
            ave_ett = 0
            loss_temp = 0
            data = [] #batch
            action_list = np.array([])
            prob_list = np.array([])
            reward_list = np.array([])
            done_list = np.array([])
            for j in range(number_of_problem):
                
                for l in range(number_of_batch):
                    reward_sum, tardy_sum, ett_sum, event, episode, actions, probs, rewards, dones = simulation(
                        problem[j][0], problem[j][1], problem[j][2], problem[j][3], problem[j][4], problem[j][5],
                        problem[j][6], problem[j][7], problem[j][8], problem[j][9], 'RL_full', ppo)
                    ave_reward += reward_sum.item()
                    ave_ett += ett_sum
                    ave_tardy += tardy_sum
                    data.append(episode)
                    action_list = np.concatenate((action_list, actions))
                    prob_list = np.concatenate((prob_list, probs))
                    reward_list = np.concatenate((reward_list, rewards))
                    done_list = np.concatenate((done_list, dones))
            for m in range(K_epoch):
                ave_loss, v_loss, p_loss = ppo.update(data, prob_list, reward_list, action_list, done_list,step,model_dir)
                loss_temp += ave_loss
            ave_reward = float(ave_reward) / number_of_problem / number_of_batch
            ave_ett = float(ave_ett) / number_of_problem /number_of_batch
            ave_tardy = float(ave_tardy) / number_of_problem / number_of_batch
            
            history[step,0]=ave_reward
            vessl.log(step=step, payload={'train_average_reward': ave_reward})
            
            step += 1
            if step%validation_step==1 and step>500:
                valid_reward_HR=0
                valid_ett_HR=0
                valid_tardy_HR=0
                valid_reward_RHR=0
                valid_ett_RHR=0
                valid_tardy_RHR=0
                valid_reward_full=0
                valid_ett_full=0
                valid_tardy_full=0
                
                best_reward=0
                best_ett=0
                best_tardy=0

                for j in range(number_of_validation):
                    temp_best_reward = -100
                    temp_ett_reward = -100
                    temp_tardy_reward = -100
                    for l in range(number_of_validation_batch):
                        
                        
                        reward_sum, tardy_sum, ett_sum, event, episode, actions, probs, rewards, dones = simulation(
                            validation[j][0], validation[j][1], validation[j][2], validation[j][3], validation[j][4],
                            validation[j][5], validation[j][6], validation[j][7], validation[j][8], validation[j][9], 'RL_full', ppo)
                        valid_reward_full += reward_sum.item()
                        valid_ett_full += ett_sum
                        valid_tardy_full += tardy_sum
                        
                        temp_best_reward=max(reward_sum.item(),temp_best_reward)
                        temp_ett_reward = max(ett_sum, temp_ett_reward)
                        temp_tardy_reward = max(tardy_sum, temp_tardy_reward)
                    best_reward+=temp_best_reward
                    best_ett+=temp_ett_reward
                    best_tardy+=temp_tardy_reward
                valid_reward_HR=valid_reward_HR/(number_of_validation*number_of_validation_batch)
                valid_reward_RHR=valid_reward_RHR/(number_of_validation*number_of_validation_batch)
                valid_reward_full=valid_reward_full/(number_of_validation*number_of_validation_batch)
                
                valid_ett_HR = valid_ett_HR / (number_of_validation * number_of_validation_batch)
                valid_ett_RHR = valid_ett_RHR / (number_of_validation * number_of_validation_batch)
                valid_ett_full = valid_ett_full / (number_of_validation * number_of_validation_batch)
                
                valid_tardy_HR = valid_tardy_HR / (number_of_validation * number_of_validation_batch)
                valid_tardy_RHR = valid_tardy_RHR / (number_of_validation * number_of_validation_batch)
                valid_tardy_full = valid_tardy_full / (number_of_validation * number_of_validation_batch)
                
                best_reward=best_reward/(number_of_validation)
                best_ett = best_ett / (number_of_validation)
                best_tardy = best_tardy / (number_of_validation)

                valid_step=int(step/validation_step)
                validation_history[valid_step, 0] = valid_reward_HR
                validation_history[valid_step, 1] = valid_ett_HR
                validation_history[valid_step, 2] = valid_tardy_HR
                validation_history[valid_step, 3] = valid_reward_RHR
                validation_history[valid_step, 4] = valid_ett_RHR
                validation_history[valid_step, 5] = valid_tardy_RHR
                validation_history[valid_step, 6] = valid_reward_full
                validation_history[valid_step, 7] = valid_ett_full
                validation_history[valid_step, 8] = valid_tardy_full
                validation_history[valid_step, 9] = best_reward
                validation_history[valid_step, 10] = best_ett
                validation_history[valid_step, 11] = best_tardy
                
                vessl.log(step=step, payload={'valid_average_reward_full':valid_reward_full})

                
    history=pd.DataFrame(history)
    validation_history=pd.DataFrame(validation_history)
    history.to_excel(history_dir+'history.xlsx', sheet_name='Sheet', index=False)
    validation_history.to_excel(history_dir + 'valid_history.xlsx', sheet_name='Sheet', index=False)

