import numpy as np
from scipy.special import softmax
from env import CustomEnv
from numpy.random import choice,randint
from collections import deque

size=100
dims=2
nS=size**dims
nA=3**dims
gamma=0.9
epsilon=0

value=np.zeros(nS)
theta=np.zeros((nS,nA))
fr=open("data/ac.csv","w")
N=100000000
env=CustomEnv()
env1=CustomEnv()
state=env.reset()
returns=deque(maxlen=100000)
vstep=np.ones(nS)
pstep=np.ones((nS,nA))
fr.write("timestep\treturn\n")
for n in range(N):
    probs=softmax(theta[state])
    random=choice(2,p=[1-epsilon,epsilon])
    action=randint(nA) if random else choice(nA,p=probs/np.sum(probs))
    next_state,reward,_,_=env.step(action)
    value[state]+=(1/vstep[state]**0.55)*(reward+gamma*value[next_state]-value[state])
    theta[state,action]+=(1/pstep[state,action]**0.55)*(reward+gamma*value[next_state]-value[state])
    vstep[state]+=1
    pstep[state,action]+=1
    returns.append(reward)
    if n%100000==0:
        mean=np.mean(returns)
        fr.write(str(n)+"\t"+str(mean)+"\n")
        fr.flush()
        print(n//100000,":",mean)
    state=next_state
fr.close()




