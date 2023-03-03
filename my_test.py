import gym
import torch
import random
import numpy as np

import mujoco_maze
import asset

from HAC import HAC

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Current device is : ", device)


def test():
    
    #################### Hyperparameters ####################
    env_name = "AntSquareRoom-v0"
    save_episode = 10               # keep saving every n episodes
    max_episodes = 10000             # max num of training episodes
    random_seed = 0
    render = False
    
    env = gym.make(env_name).unwrapped
    # env.action_space.high[0] = 0.5
    # env.action_space.low[0]  = -0.5


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

    # exploration noise std for primitive action and subgoals
    exploration_action_noise = np.array([0.5] * env.action_space.high.shape[0])        
    exploration_state_noise = np.array([1] * state_dim ) 
    
    goal_state = np.array([2.5, 2.5] + [0]*8 )        # final goal state to be achived
    threshold = np.array([0.5, 0.5] + [0]*8 )         # threshold value to check if goal state is achieved
    

    # HAC parameters:
    k_level = 2                 # num of levels in hierarchy
    H = 50                     # time horizon to achieve subgoal
    lamda = 0.2                 # subgoal testing parameter
    
    # DDPG parameters:
    gamma = 0.95                # discount factor for future rewards
    n_iter = 100                # update policy n_iter times in one DDPG update
    batch_size = 100            # num of transitions sampled from replay buffer
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
    
    # creating HAC agent and setting parameters
    agent = HAC(k_level, H, state_dim, action_dim, render, threshold, 
                action_bounds, action_offset, state_bounds, state_offset, lr)
    
    agent.set_parameters(lamda, gamma, action_clip_low, action_clip_high, 
                       state_clip_low, state_clip_high, exploration_action_noise, exploration_state_noise)
    
    # load agent
    agent.load(directory, filename)

    # Evaluation
    for i_episode in range(1, max_episodes+1):
        
        agent.reward = 0
        agent.timestep = 0
        
        state = env.reset()
        last_state, done = agent.run_HAC(env, k_level-1, state, goal_state, False)
        
        print("Episode: {}\t Reward: {}\t len: {}".format(i_episode, agent.reward, agent.timestep))
    
    env.close()


if __name__ == '__main__':
    test()
 
  

