import numpy as np
from scipy.special import softmax
from env import CustomEnv
from numpy.random import choice,randint
from collections import deque
import sys,os

size,dims=10,3

nS=size**dims
nA=2*dims
gamma=0.9

alpha=float(sys.argv[1])
beta=float(sys.argv[2])

logrd="data/acf_"+sys.argv[1]+"_"+sys.argv[2]+"/"
if not os.path.exists(logrd):
          os.makedirs(logrd)
run_num = len(next(os.walk(logrd))[2])
fr=open(logrd+str(run_num)+".csv","w")

K=100
ff=10
def feat(state):
    res=np.zeros(nS//ff)
    res[state//ff]=1
    return res

value=np.zeros(nS//ff)
theta=np.zeros((nS,nA))

N=100000000
env=CustomEnv()
returns=deque(maxlen=100000)
vstep=np.ones(nS)
pstep=np.ones((nS,nA))
fr.write("timestep\treturn\tverror\n")
state=env.reset()
for n in range(1,N+1):
    b=1/(n//K+1)**beta
    probs=softmax(theta[state])
    action=choice(nA,p=probs/np.sum(probs))
    next_state,reward,_,_=env.step(action)
    
    state1=randint(nS)
    probs1=softmax(theta[state1])
    action1=choice(nA,p=probs1/np.sum(probs1))
    next_state1,reward1=env.sample(state1,action1)

    state2,action2=randint(nS),randint(nA)
    next_state2,reward2=env.sample(state2,action2)
    
    delta1=reward1+gamma*np.dot(value,feat(next_state1))-np.dot(value,feat(state1))
    value+=b*delta1*feat(state1)

    theta[state2,action2]+=(1/(pstep[state2,action2]//K+1)**alpha)*(reward2+gamma*np.dot(value,feat(next_state2))-np.dot(value,feat(state2)))
    pstep[state2,action2]+=1
    returns.append(reward)
    if n%10000==0:
        mean=np.mean(returns)
        values=np.array([np.dot(value,feat(k)) for k in range(nS)])
        error=np.linalg.norm(env.V-values)
        fr.write(str(n)+"\t"+str(mean)+"\t"+str(error)+"\n")
        fr.flush()
        print(n//10000,":",mean,error)
    state=next_state
fr.close()




