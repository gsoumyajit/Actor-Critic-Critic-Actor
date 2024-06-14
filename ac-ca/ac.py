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

alpha=float(sys.argv[1])
beta=float(sys.argv[2])

logrd="data/ac_"+sys.argv[1]+"_"+sys.argv[2]+"/"
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
    
    value[state1]+=(1/(vstep[state1]//K+1)**beta)*(reward1+gamma*value[next_state1]-value[state1])
    theta[state2,action2]+=(1/(pstep[state2,action2]//K+1)**alpha)*(reward2+gamma*value[next_state2]-value[state2])
    vstep[state1]+=1
    pstep[state2,action2]+=1
    returns.append(reward)
    if n%10000==0:
        mean=np.mean(returns)
        error=np.linalg.norm(env.V-value)
        fr.write(str(n)+"\t"+str(mean)+"\t"+str(error)+"\n")
        fr.flush()
        print(n//10000,":",mean,error)
    state=next_state
fr.close()




