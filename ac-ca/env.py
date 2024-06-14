import numpy as np
import gym
from tqdm import tqdm

if __name__=="__main__":
    size,dims=20,2
    #size,dims=10,3
    #size,dims=10,4
    nS=size**dims
    nA=2*dims
    save1=np.random.randint(nS,size=40)
    #save1=np.random.randint(nS,size=100)
    #save1=np.random.randint(nS,size=1000)
    np.savetxt("mdp/S",save1)

class CustomEnv(gym.Env):
    def __init__(self):
        save1=np.loadtxt("mdp/S").astype(int)
        self.size,self.dims=20,2
        #self.size,self.dims=10,3
        #self.size,self.dims=10,4
        self.nS=self.size**self.dims
        self.nA=2*self.dims

        self.R=np.zeros(self.nS)
        self.R[save1]=100
        
        self.P=np.zeros((self.nA,self.nS,self.nS))
        for j in range(self.nS):
            idx=list(np.unravel_index(j,(self.size,)*self.dims))
            for k in range(self.dims):
                idx[k]=(idx[k]+1)%self.size
                state=np.ravel_multi_index(idx,(self.size,)*self.dims)
                self.P[:,j,state]=1
                self.P[2*k,j,state]=9*(self.nA-1)
                idx[k]=(idx[k]-2)%self.size
                state=np.ravel_multi_index(idx,(self.size,)*self.dims)
                self.P[:,j,state]=1
                self.P[2*k+1,j,state]=9*(self.nA-1)
                idx[k]=(idx[k]+1)%self.size
            
            for k in range(self.nA):
                self.P[k,j]=self.P[k,j]/np.sum(self.P[k,j])

        self.V=np.zeros(self.nS)
        self.policy=np.zeros(self.nS).astype(int)
        iters=200
        gamma=0.9
        
        for it in tqdm(range(iters)):
            val=np.copy(self.V)
            for i in range(self.nS):
                self.policy[i]=np.argmax([np.dot(self.P[u,i],self.R+gamma*val) for u in range(self.nA)])
                self.V[i]=np.max([np.dot(self.P[u,i],self.R+gamma*val) for u in range(self.nA)])
            print(np.linalg.norm(self.V))

    
    def reset(self):
        self.state=0
        return self.state

    def step(self,action):
        next_state=np.random.choice(self.nS,p=self.P[action,self.state])
        reward=self.R[next_state]
        self.state=next_state
        done=False
        return next_state,reward,done,None

    def sample(self,state,action):
        next_state=np.random.choice(self.nS,p=self.P[action,state])
        reward=self.R[next_state]
        return next_state,reward

