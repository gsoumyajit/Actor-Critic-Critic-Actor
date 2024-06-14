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
epsilon=0

ff=100
def feat(state):
    res=np.zeros(nS//ff)
    res[state//ff]=1
    return res
def phi(state,action):
    res=np.zeros(nA*nS//ff)
    res[action*nS//ff:(action+1)*nS//ff]=feat(state)
    return res

value=np.zeros(nS//ff)
theta=np.zeros(nA*nS//ff)
an=int(sys.argv[1])

if an%2==0: print("Executing 1.critic actor")
else: print("Executing 2.critic actor")

logrd="data/caf2_"+sys.argv[1]+"/"
if not os.path.exists(logrd):
          os.makedirs(logrd)
run_num = len(next(os.walk(logrd))[2])
fr=open(logrd+str(run_num)+".csv","w")
N=10000000
env=CustomEnv()
state=env.reset()
K=100000
returns=deque(maxlen=K)
fr.write("timestep\treturn\n")

for n in range(N):
    if an%2==0:
        a=1/(n//K+1)
        b=np.log(n//K+2)/(n//K+2)
    else:
        b=1/(n//K+1)
        a=1/(np.log(n//K+2)*(n//K+2))


    probs=softmax([np.dot(theta,phi(state,k)) for k in range(nA)])
    random=choice(2,p=[1-epsilon,epsilon])
    action=randint(nA) if random else choice(nA,p=probs/np.sum(probs))
    next_state,reward,_,_=env.step(action)
    delta=reward+gamma*np.dot(value,feat(next_state))-np.dot(value,feat(state))
    value+=a*delta*feat(state)
    psi=phi(state,action)-np.sum([probs[k]*phi(state,k) for k in range(nA)])
    if not random:
        theta+=b*psi*delta
    returns.append(reward)
    if n%K==0:
        #epsilon=max(0.1,epsilon-0.01)
        mean=np.mean(returns)
        fr.write(str(n)+"\t"+str(mean)+"\n")
        fr.flush()
        print(n//K,":",mean,probs,epsilon)
        
    state=next_state
fr.close()




