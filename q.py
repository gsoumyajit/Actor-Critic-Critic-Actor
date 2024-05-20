import numpy as np
from scipy.special import softmax
from env import CustomEnv
from numpy.random import choice,randint
from collections import deque
import sys

size=100
dims=2
nS=size**dims
nA=3**dims
gamma=0.9
epsilon=0.1

value=np.zeros(nS)
theta=np.zeros((nS,nA))
Q=np.zeros((nS,nA))

fr=open("data/q.csv","w")
an=float(sys.argv[1])
N=100000000
env1=CustomEnv()
state1=env1.reset()
returns1=deque(maxlen=100000)
qstep=np.ones((nS,nA))
fr.write("timestep\treturn\n")
for n in range(N):
    random1=choice(2,p=[1-epsilon,epsilon])
    action1=randint(nA) if random1 else np.argmax(Q[state1])
    next_state1,reward1,_,_=env1.step(action1)
    Q[state1,action1]+=(1/qstep[state1,action1]**an)*(reward1+gamma*np.max(Q[next_state1])-Q[state1,action1])
    qstep[state1,action1]+=1
    returns1.append(reward1)
    if n%100000==0:
        epsilon-=0.001
        mean=np.mean(returns1)
        fr.write(str(n)+"\t"+str(mean)+"\n")
        fr.flush()
        print(n//100000,":",mean)
    state1=next_state1
fr.close()



