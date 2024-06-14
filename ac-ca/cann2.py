import numpy as np
from scipy.special import softmax
from env1 import CustomEnv
from numpy.random import choice,randint
from collections import deque
import sys
from tqdm import tqdm
import torch
from torch.distributions import Categorical
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
import os

class discrete_policy(nn.Module):
    def __init__(self, nS, nH1, nH2, nA):
        super(discrete_policy, self).__init__()
        self.h1 = nn.Linear(nS, nH1)
        self.h2 = nn.Linear(nH1, nH2)
        self.out = nn.Linear(nH2, nA)

    def forward(self, x):
        x = torch.tanh(self.h2(torch.tanh(self.h1(x))))
        x = torch.softmax(self.out(x), dim=0)
        return x

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

size=100
dims=2
nS=size**dims
nA=3**dims
gamma=0.9
epsilon=0
K=10000
an=int(sys.argv[1])

policy = discrete_policy(dims, 10, 10, nA)
value = value_function(dims, 10, 10, 1)
voptim = torch.optim.SGD(value.parameters(),lr=0.1)
poptim = torch.optim.SGD(policy.parameters(),lr=0.1)

if an%2==0:
    print("Executing 1.critic actor")
    vscheduler=LambdaLR(poptim,lambda n:1/(n//1000+1))
    pscheduler=LambdaLR(voptim,lambda n:np.log(n//1000+2)/(n//1000+2))
else:
    print("Executing 2.critic actor")
    pscheduler=LambdaLR(poptim,lambda n:1/(n//1000+1))
    vscheduler=LambdaLR(voptim,lambda n:1/(np.log(n//1000+2)*(n//1000+2)))

def feat(state):
    res=torch.zeros(dims)
    for i in range(dims):
        res[i]=state%size
        state=state//size
    return res

logrd="data/cann2_"+sys.argv[1]+"/"
if not os.path.exists(logrd):
          os.makedirs(logrd)
run_num = len(next(os.walk(logrd))[2])
fr=open(logrd+str(run_num)+".csv","w")

N=1000000
env=CustomEnv()
state=env.reset()
returns=deque(maxlen=100000)
fr.write("timestep\treturn\n")
for n in range(N):
    probs=policy(feat(state))
    random=choice(2,p=[1-epsilon,epsilon])
    action=torch.tensor(randint(nA)) if random else Categorical(probs).sample()
    next_state,reward,_,_=env.step(action)
    delta=reward+gamma*value(feat(next_state)).detach()-value(feat(state))
    vloss=0.5*delta**2
    ploss=-Categorical(probs).log_prob(action)*delta.detach()
    voptim.zero_grad()
    vloss.backward()
    voptim.step()
    if not random:
        poptim.zero_grad()
        ploss.backward()
        poptim.step()
    vscheduler.step()
    pscheduler.step()
    returns.append(reward)
    if n%K==0:
        epsilon=max(0,epsilon-0.02)
        mean=np.mean(returns)
        fr.write(str(n)+"\t"+str(mean)+"\n")
        fr.flush()
        print(n//K,":",mean)
    state=next_state
fr.close()




