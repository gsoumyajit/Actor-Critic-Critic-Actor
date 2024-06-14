import numpy as np
from scipy.special import softmax
from env1 import CustomEnv
from numpy.random import choice,randint
from collections import deque
import sys
from tqdm import tqdm
import os

size=100
dims=2
nS=size**dims
nA=3**dims
gamma=0.9
epsilon=1

ff=100
def feat(state):
    res=np.zeros(nS//ff)
    res[state//ff]=1
    return res
def phi(state,action):
    res=np.zeros(nA*nS//ff)
    res[action*nS//ff:(action+1)*nS//ff]=feat(state)
    return res

Q=np.zeros(nA*nS//ff)
logrd="data/qf_"+sys.argv[1]+"/"
if not os.path.exists(logrd):
          os.makedirs(logrd)
run_num = len(next(os.walk(logrd))[2])
fr=open(logrd+str(run_num)+".csv","w")
an=float(sys.argv[1])
N=10000000
K=100000
env1=CustomEnv()
state1=env1.reset()
returns1=deque(maxlen=K)
fr.write("timestep\treturn\n")
for n in range(N):
    a=1/(n//K+2)**an
    random1=choice(2,p=[1-epsilon,epsilon])
    action1=randint(nA) if random1 else np.argmax([np.dot(Q,phi(state1,k)) for k in range(nA)])
    next_state1,reward1,_,_=env1.step(action1)
    Q+=a*(reward1+gamma*np.max([np.dot(Q,phi(next_state1,k)) for k in range(nA)])-np.dot(Q,phi(state1,action1)))*phi(state1,action1)
    returns1.append(reward1)
    if n%K==0:
        epsilon=max(0,epsilon-0.02)
        mean=np.mean(returns1)
        fr.write(str(n)+"\t"+str(mean)+"\n")
        fr.flush()
        print(n//K,":",mean)
    state1=next_state1
fr.close()



