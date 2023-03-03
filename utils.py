import numpy as np
import os
import imageio
class ReplayBuffer:
    def __init__(self, max_size=5e5):
        self.buffer = []
        self.max_size = int(max_size)
        self.size = 0
    
    def add(self, transition):
        assert len(transition) == 7, "transition must have length = 7"
        
        # transiton is tuple of (state, action, reward, next_state, goal, gamma, done)
        self.buffer.append(transition)
        self.size +=1
    
    def sample(self, batch_size):
        # delete 1/5th of the buffer when full
        if self.size > self.max_size:
            del self.buffer[0:int(self.size/5)]
            self.size = len(self.buffer)
        
        indexes = np.random.randint(0, len(self.buffer), size=batch_size)
        states, actions, rewards, next_states, goals, gamma, dones = [], [], [], [], [], [], []
        
        for i in indexes:
            states.append(np.array(self.buffer[i][0], copy=False))
            actions.append(np.array(self.buffer[i][1], copy=False))
            rewards.append(np.array(self.buffer[i][2], copy=False))
            next_states.append(np.array(self.buffer[i][3], copy=False))
            goals.append(np.array(self.buffer[i][4], copy=False))
            gamma.append(np.array(self.buffer[i][5], copy=False))
            dones.append(np.array(self.buffer[i][6], copy=False))
        
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(goals),  np.array(gamma), np.array(dones)
    


class VideoRecorder(object):
    def __init__(self, dir_name, height=256, width=256, fps=25):
        self.dir_name = dir_name
        try:
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
        except OSError:
            print ('Error: Creating directory. ' +  dir_name)
    
        self.height = height
        self.width = width
        self.fps = fps
        self.frames = []
        
    def init(self, enabled=True):
        self.frames = []
        self.enabled = self.dir_name is not None and enabled

    def record(self, obs):
        if self.enabled:
            self.frames.append(obs)
        
    def save(self, file_name):
        if self.enabled:
            path = os.path.join(self.dir_name, file_name)
            imageio.mimsave(path, self.frames, fps=self.fps)