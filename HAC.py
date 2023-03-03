import torch
import numpy as np
from DDPG import DDPG
from utils import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
import os
import os.path as osp
import matplotlib.pyplot as plt


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class HAC:
    def __init__(self, k_level, H, state_dim, action_dim, render, threshold, 
                 action_bounds, action_offset, state_bounds, state_offset, lr):
        
        # adding lowest level
        self.HAC = [DDPG(state_dim, action_dim, action_bounds, action_offset, lr, H)]
        self.replay_buffer = [ReplayBuffer()]
        
        # adding remaining levels
        for _ in range(k_level-1):
            self.HAC.append(DDPG(state_dim, state_dim, state_bounds, state_offset, lr, H))
            self.replay_buffer.append(ReplayBuffer())
        
        # set some parameters
        self.k_level = k_level
        self.H = H
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.threshold = threshold
        self.render = render
        
        # logging parameters
        self.goals = [None]*self.k_level
        self.reward = 0
        self.timestep = 0
        
        # tensorboard parameters
        self.log_path = '/home/jonghae/Hierarchical-Actor-Critic-HAC-PyTorch'
        self.writer = SummaryWriter(osp.join(self.log_path, 'tb')) # /home/ss/bb/tb
        self.tb_0level_timestep = 0
        self.tb_1level_timestep = 0
        self.tb_2level_timestep = 0

    def set_parameters(self, lamda, gamma, action_clip_low, action_clip_high, 
                       state_clip_low, state_clip_high, exploration_action_noise, exploration_state_noise):
        
        self.lamda = lamda
        self.gamma = gamma
        self.action_clip_low = action_clip_low
        self.action_clip_high = action_clip_high
        self.state_clip_low = state_clip_low
        self.state_clip_high = state_clip_high
        self.exploration_action_noise = exploration_action_noise
        self.exploration_state_noise = exploration_state_noise
    
    # check only (x,y) states , not all states
    def check_goal(self, state, goal, threshold):
        for i in range(2):
            if abs(state[i]-goal[i]) > threshold[i]:
                return False
        # print("Subgoal : Clear")
        return True
    
    
    def run_HAC(self, env, i_level, state, goal, is_subgoal_test, video, is_test=False):
        next_state = None
        done = None
        goal_transitions = []
        
        # logging updates
        self.goals[i_level] = goal
        self.cum_reward = np.zeros(self.k_level)
        
        # H attempts
        for _ in range(self.H):
            # if this is a subgoal test, then next/lower level goal has to be a subgoal test
            is_next_subgoal_test = is_subgoal_test
            
            action = self.HAC[i_level].select_action(state, goal)
            
            #   <================ high level policy ================>
            if i_level > 0:
                # add noise or take random action if not subgoal testing
                if not is_subgoal_test:
                    if np.random.random_sample() > 0.2:
                      action = action + np.random.normal(0, self.exploration_state_noise)
                      action = action.clip(self.state_clip_low, self.state_clip_high)
                    else:
                      action = np.random.uniform(self.state_clip_low, self.state_clip_high)
                
                # Determine whether to test subgoal (action)
                if np.random.random_sample() < self.lamda:
                    is_next_subgoal_test = True

                if self.tb_1level_timestep % 225 == 0:

                    plt.scatter(np.array(action[0]), np.array(action[1]),color = 'blue', label = 'subgoal')
                    plt.scatter(np.array(env._get_obs()[0]) , np.array(env._get_obs()[1]),color =  'black', label = 'current state')
                    plt.scatter(2.5, 2.5, color =  'red', label = 'global goal')
                    plt.legend()
                    plt.xlim([-4, 4])  
                    plt.ylim([-4, 4])  
                    plt.show(block=False)
                    plt.pause(0.1)
                    plt.close()
 

                # print("GOAL :",action[0:2])
                # print("OBSV :",env._get_obs()[0:2])

                # Pass subgoal to lower level 
                next_state, done = self.run_HAC(env, i_level-1, state, action, is_next_subgoal_test, video)

                # if subgoal was tested but not achieved, add subgoal testing transition
                if is_next_subgoal_test and not self.check_goal(action, next_state, self.threshold):
                    self.replay_buffer[i_level].add((state, action, -self.H/2, next_state, goal, 0.0, float(done)))
                
                # for hindsight action transition
                action = next_state
                
                # tb logging for level 1, 2
                if i_level == 1:
                    state_tb = torch.FloatTensor(state.reshape(1, -1)).to(device)
                    action_tb = torch.FloatTensor(action.reshape(1, -1)).to(device)
                    goals_1_tb = torch.FloatTensor(self.goals[1].reshape(1, -1)).to(device)
                    if is_test:
                        self.writer.add_scalar('testingng/Q[1-level]',self.HAC[1].critic.forward(state_tb, action_tb, goals_1_tb), self.tb_1level_timestep)
                    else:
                        self.writer.add_scalar('training/Q[1-level]',self.HAC[1].critic.forward(state_tb, action_tb, goals_1_tb), self.tb_1level_timestep)
                    if not is_test:
                        self.tb_1level_timestep += 1

                if i_level == 2:
                    state_tb = torch.FloatTensor(state.reshape(1, -1)).to(device)
                    action_tb = torch.FloatTensor(action.reshape(1, -1)).to(device)
                    goals_2_tb = torch.FloatTensor(self.goals[2].reshape(1, -1)).to(device)
                    self.writer.add_scalar('training/Q[2-level]',self.HAC[2].critic.forward(state_tb, action_tb, goals_2_tb), self.tb_2level_timestep)
                    if not is_test:
                        self.tb_2level_timestep += 1

                goal_achieved = self.check_goal(next_state, goal, self.threshold)
                if goal_achieved == True:
                    self.cum_reward[i_level] += 1
                    print("sub goal : clear")

            #   <================ low level policy ================>
            else:
                # add noise or take random action if not subgoal testing
                if not is_subgoal_test:
                    if np.random.random_sample() > 0.2:
                      action = action + np.random.normal(0, self.exploration_action_noise)
                      action = action.clip(self.action_clip_low, self.action_clip_high)
                    else:
                      action = np.random.uniform(self.action_clip_low, self.action_clip_high)
                    
                # take primitive action
                next_state, rew, done, _ = env.step(action)
                video.record(env.render(mode='rgb_array', camera_name='top_view'))

                # env.render(mode='rgb_array', camera_name='top_view')
                if self.render:
                    
                    # env.render() ##########
                    
                    if self.k_level == 2:
                        env.unwrapped.render_goal(self.goals[0], self.goals[1])
                    elif self.k_level == 3:
                        env.unwrapped.render_goal_2(self.goals[0], self.goals[1], self.goals[2])
                    
                    
                # this is for logging
                self.reward += rew
                self.timestep +=1
                if not is_test:
                    self.tb_0level_timestep +=1
        
                # tb logging for level 0
                state_tb = torch.FloatTensor(state.reshape(1, -1)).to(device)
                action_tb = torch.FloatTensor(action.reshape(1, -1)).to(device)
                goals_0_tb = torch.FloatTensor(self.goals[0].reshape(1, -1)).to(device)
                self.writer.add_scalar('training/Q[0-level]',self.HAC[0].critic.forward(state_tb, action_tb, goals_0_tb), self.tb_0level_timestep)

                goal_achieved = self.check_goal(next_state, goal, self.threshold)
                if goal_achieved == True:
                    self.cum_reward[i_level] += 1
                    print("primitive goal : clear")
            #   <================ finish one step/transition ================>
            
            # check if goal is achieved
        

            # hindsight action transition
            if goal_achieved:
                self.replay_buffer[i_level].add((state, action, 0.0, next_state, goal, 0.0, float(done)))
            else:
                self.replay_buffer[i_level].add((state, action, -1.0, next_state, goal, self.gamma, float(done)))
                
            # copy for goal transition
            goal_transitions.append([state, action, -1.0, next_state, None, self.gamma, float(done)])
            
            state = next_state
            
            if done or goal_achieved:
                break
        
        
        #   <================ finish H attempts ================>
        
        # hindsight goal transition
        # last transition reward and discount is 0
        goal_transitions[-1][2] = 0.0
        goal_transitions[-1][5] = 0.0
        for transition in goal_transitions:
            # last state is goal for all transitions
            transition[4] = next_state
            self.replay_buffer[i_level].add(tuple(transition))
        if is_test:
            self.writer.add_scalar('testing/reward{}'.format(i_level),self.cum_reward[i_level], self.tb_1level_timestep)
        else:
            self.writer.add_scalar('training/reward{}'.format(i_level),self.cum_reward[i_level], self.tb_1level_timestep)
        
        if is_test:
            video.save('test_{}.mp4'.format(self.tb_1level_timestep))
        else:
            video.save('train_{}.mp4'.format(self.tb_1level_timestep))
            
        return next_state, done
    
    
    def update(self, n_iter, batch_size):
        for i in range(self.k_level):
            self.HAC[i].update(self.replay_buffer[i], n_iter, batch_size)
    
    
    def save(self, directory, name):
        for i in range(self.k_level):
            self.HAC[i].save(directory, name+'_level_{}'.format(i))
    
    
    def load(self, directory, name):
        for i in range(self.k_level):
            self.HAC[i].load(directory, name+'_level_{}'.format(i))
    
        
        
        
        
        
