import numpy as np
import gym
from tqdm import tqdm

if __name__=="__main__":
    size=100
    dims=2
    nS=size**dims
    nA=3**dims
    save1=np.random.randint(nS,size=1000)
    np.savetxt("mdp/S1",save1)


class CustomEnv(gym.Env):
    def __init__(self):
        save1=np.loadtxt("mdp/S1").astype(int)
        self.size=100
        self.dims=2
        self.nS=self.size**self.dims
        self.nA=3**self.dims
        self.R=np.zeros(self.nS)
        
        k=np.unravel_index(save1,(self.size,)*self.dims)
        self.R[save1]=self.size-abs(k[1]-k[0])
        self.P=np.ones((self.nA,self.nA))
        for i in range(self.nA):
            self.P[i,i]=9*(self.nA-1)
            self.P[i]/=np.sum(self.P[i])
        

    def reset(self):
        start_state=0
        self.state=np.ones(self.dims)*start_state
        return start_state

    def step(self,action):
        direction=np.random.choice(self.nA,p=self.P[action])
        for i in reversed(range(self.dims)):
            self.state[i]=(self.state[i]+direction%3-1)%self.size
            direction=direction//3
        state_index=0
        for i in range(self.dims):
            state_index=state_index*self.size+self.state[i]
        reward=self.R[int(state_index)]
        done=False
        return int(state_index),reward,done,None

