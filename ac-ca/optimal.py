import numpy as np
from scipy.special import softmax
from env import CustomEnv
from numpy.random import choice,randint
from collections import deque
import sys,os

size,dims=20,2
#size,dims=10,3
#size,dims=10,4

nS=size**dims
nA=2*dims
gamma=0.9

logrd="data/optimal/"
if not os.path.exists(logrd):
          os.makedirs(logrd)
run_num = len(next(os.walk(logrd))[2])
fr=open(logrd+str(run_num)+".csv","w")

K=100
value=np.zeros(nS)
theta=np.zeros((nS,nA))

N=100000000
env=CustomEnv()
returns=deque(maxlen=100000)
fr.write("timestep\treturn\n")
state=env.reset()
for n in range(1,N+1):
    action=env.policy[state]
    next_state,reward,_,_=env.step(action)
    
    returns.append(reward)
    if n%10000==0:
        mean=np.mean(returns)
        fr.write(str(n)+"\t"+str(mean)+"\n")
        fr.flush()
        print(n//10000,":",mean)
    state=next_state
fr.close()




