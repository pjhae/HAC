import gym
import torch
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os
import os.path as osp
import datetime
from utils import VideoRecorder

# writer = SummaryWriter(osp.join(log_path, 'tb')) # /home/ss/bb/tb
# writer.add_scalar('training/Q_loss', loss, training_steps)
# writer.add_scalar('evalation/evaluate_reward', reward, training_steps)
# tensorboard --logdir=./ --port 7777 --host 147.47.91.??
# 47.47.91.??:7777
import mujoco_maze
import asset

from HAC import HAC

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Current device is : ", device)

# log_path = '/home/jonghae/Hierarchical-Actor-Critic-HAC-PyTorch'
# writer = SummaryWriter(osp.join(log_path, 'tb')) # /home/ss/bb/tb

def train():
    #################### Hyperparameters ####################
    env_name = "AntSquareRoom-v0"
    save_episode = 10               # keep saving every n episodes
    max_episodes = 10000             # max num of training episodes
    random_seed = 0
    render = False
    
    env = gym.make(env_name).unwrapped


    state_dim = env._get_obs().shape[0]        ## 2
    action_dim = env.action_space.shape[0]     ## 8

    """
     Actions (both primitive and subgoal) are implemented as follows:
       action = ( network output (Tanh) * bounds ) + offset
       clip_high and clip_low bound the exploration noise
    """
    
    # primitive action bounds and offset
    action_bounds = (env.action_space.high - env.action_space.low)/2
    action_offset = (env.action_space.high + env.action_space.low)/2
    
    action_clip_low  = np.array(action_offset - action_bounds)
    action_clip_high = np.array(action_offset + action_bounds)
    
    action_bounds= torch.FloatTensor(action_bounds.reshape(1, -1)).to(device)
    action_offset = torch.FloatTensor(action_offset.reshape(1, -1)).to(device)


    # state bounds and offset
    state_bounds = np.array((env._get_obs_space().high - env._get_obs_space().low)/2)
    state_offset =  np.array((env._get_obs_space().high + env._get_obs_space().low)/2)
    
    for i in range(len(state_bounds)):
        if env._get_obs_space().high[i] == float('inf'):
            state_bounds[i] = 1000
            state_offset[i] = 0

    state_clip_low  = np.array(state_offset - state_bounds)
    state_clip_high = np.array(state_offset + state_bounds)
    
    state_bounds = torch.FloatTensor(state_bounds.reshape(1, -1)).to(device)
    state_offset = torch.FloatTensor(state_offset.reshape(1, -1)).to(device)

    print(state_clip_low, state_clip_high)

    # exploration noise std for primitive action and subgoals
    exploration_action_noise = np.array([0.25] * env.action_space.high.shape[0])        
    exploration_state_noise = np.array([0.2] * state_dim ) 
    
    goal_state = np.array([2.5, 2.5] + [0]*8 )        # final goal state to be achived
    threshold = np.array([0.5, 0.5] + [0]*8 )         # threshold value to check if goal state is achieved
    
    # HAC parameters:
    k_level = 2                 # num of levels in hierarchy
    H = 175                     # time horizon to achieve subgoal
    lamda = 0.2                 # subgoal testing parameter
    
    # DDPG parameters:
    gamma = 0.95                # discount factor for future rewards
    n_iter = 100                # update policy n_iter times in one DDPG update
    batch_size = 100           # num of transitions sampled from replay buffer
    lr = 0.001
    
    # save trained models
    directory = "./preTrained/{}/{}level/".format(env_name, k_level) 
    filename = "HAC_{}".format(env_name)
    #########################################################
    
    
    if random_seed:
        print("Random Seed: {}".format(random_seed))
        env.seed(random_seed)
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
    
    video = VideoRecorder(dir_name = '/home/jonghae/Hierarchical-Actor-Critic-HAC-PyTorch/video/{}'.format(datetime.datetime.now().strftime("%H:%M:%S %p")))
    # creating HAC agent and setting parameters
    agent = HAC(k_level, H, state_dim, action_dim, render, threshold, 
                action_bounds, action_offset, state_bounds, state_offset, lr)
    
    agent.set_parameters(lamda, gamma, action_clip_low, action_clip_high, 
                       state_clip_low, state_clip_high, exploration_action_noise, exploration_state_noise)
    
    # logging file:
    log_f = open("log.txt","w+")
    
    # training procedure 
    for i_episode in range(1, max_episodes+1):
        agent.reward = 0
        agent.timestep = 0
        
        state = env.reset()
        video.init(enabled=(i_episode%1 == 0))
        # collecting experience in environment
        last_state, done = agent.run_HAC(env, k_level-1, state, goal_state, False, video)
        
        # writer.add_scalar('training/Reward', agent.reward , i_episode)

        ### 완전 성공한 경우의 SAVE , 파일명 뒤에 solved 붙음
        if agent.check_goal(last_state, goal_state, threshold):
            print("################ Solved! ################ ")
            name = filename + '_solved'
            agent.save(directory, name)
        
        # update all levels
        agent.update(n_iter, batch_size)
        
        # logging updates:
        log_f.write('{},{}\n'.format(i_episode, agent.reward))
        log_f.flush()
        

        ### 그냥 SAVE , 파일명 뒤에 solved 안 붙음
        if i_episode % save_episode == 0:
            agent.save(directory, filename)
        
        print("Episode: {}\t Reward: {}".format(i_episode, agent.reward))

        if i_episode % 10 == 0: # 10번에 1번 test
            # start test episodes
            for test_num in range(1):
                if test_num == 0:
                    video.init(enabled=True)
                else:
                    video.init(enabled=False)
                state = env.reset()
                last_state, done = agent.run_HAC(env, k_level-1, state, goal_state, True, video, is_test = True)
                
                

        
    
if __name__ == '__main__':
    train()
 
