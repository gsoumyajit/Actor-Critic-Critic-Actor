import numpy as np
from scipy.special import softmax
from env import CustomEnv
from numpy.random import choice,randint
from collections import deque
import sys,os
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

class value_function(nn.Module):
    def __init__(self, nS, nH1, nH2, nA):
        super(value_function, self).__init__()
        self.h1 = nn.Linear(nS, nH1)
        self.h2 = nn.Linear(nH1, nH2)
        self.out = nn.Linear(nH2, nA)

    def forward(self, x):
        x = torch.tanh(self.h2(torch.tanh(self.h1(x))))
        x = self.out(x)
        return x

size,dims=10,4

nS=size**dims
nA=2*dims
gamma=0.9

alpha=float(sys.argv[1])
beta=float(sys.argv[2])

logrd="data/acnn_"+sys.argv[1]+"_"+sys.argv[2]+"/"
if not os.path.exists(logrd):
          os.makedirs(logrd)
run_num = len(next(os.walk(logrd))[2])
fr=open(logrd+str(run_num)+".csv","w")

K=1
value = value_function(1, 10, 10, 1)
voptim = torch.optim.SGD(value.parameters(),lr=1)
vscheduler=LambdaLR(voptim,lambda n:1/(n//K+1)**beta)
theta=np.zeros((nS,nA))

def feat(state):
    return torch.FloatTensor([state])

N=100000000
env=CustomEnv()
returns=deque(maxlen=100000)
vstep=np.ones(nS)
pstep=np.ones((nS,nA))
fr.write("timestep\treturn\tverror\n")
state=env.reset()
for n in range(1,N+1):
    probs=softmax(theta[state])
    action=choice(nA,p=probs/np.sum(probs))
    next_state,reward,_,_=env.step(action)
    
    state1=randint(nS)
    probs1=softmax(theta[state1])
    action1=choice(nA,p=probs1/np.sum(probs1))
    next_state1,reward1=env.sample(state1,action1)

    state2,action2=randint(nS),randint(nA)
    next_state2,reward2=env.sample(state2,action2)
    
    delta1=reward1+gamma*value(feat(next_state1)).detach()-value(feat(state1))
    vloss=0.5*delta1**2
    voptim.zero_grad()
    vloss.backward()
    voptim.step()
    vscheduler.step()

    theta[state2,action2]+=(1/(pstep[state2,action2]//K+1)**alpha)*(reward2+gamma*value(feat(next_state2)).detach()-value(feat(state2))).item()
    pstep[state2,action2]+=1
    returns.append(reward)
    if n%10000==0:
        mean=np.mean(returns)
        values=np.array([value(feat(k)).item() for k in range(nS)])
        error=np.linalg.norm(env.V-values)
        fr.write(str(n)+"\t"+str(mean)+"\t"+str(error)+"\n")
        fr.flush()
        print(n//10000,":",mean,error)
    state=next_state
fr.close()




