import numpy as np
import gym

if __name__=="__main__":
    T=2
    size=100
    dims=2
    nS=size**dims
    nA=3**dims
    save1=np.random.randint(nS,size=(T,1000))
    save2=np.random.randint(nS,size=(T,1000))
    np.savetxt("mdp/S1",save1)
    np.savetxt("mdp/S2",save2)


class CustomEnv(gym.Env):
    def __init__(self):
        save1=np.loadtxt("mdp/S1").astype(int)
        save2=np.loadtxt("mdp/S2").astype(int)
        self.T=2
        self.size=100
        self.dims=2
        self.nS=self.size**self.dims
        self.nA=3**self.dims
        self.R=np.zeros((self.T,self.nS))
        self.G=np.zeros((self.T,self.nS))
        for i in range(self.T):
            k=np.unravel_index(save1[i],(self.size,)*self.dims)
            self.R[i,save1[i]]=self.size-abs(k[1]-k[0])
            self.G[i,save2[i]]=1
        self.P=np.ones((self.nA,self.nA))
        for i in range(self.nA):
            self.P[i,i]=9*(self.nA-1)
            self.P[i]/=np.sum(self.P[i])
        #self.P=self.P/np.sum(self.P[0])
    def reset(self):
        self.t=0
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
        reward=self.R[0,int(state_index)]
        constraint=self.G[0,int(state_index)]
        self.t+=1
        done=self.t==self.T
        return int(state_index),reward,done,constraint

