import numpy as np
p = np.zeros(4)
p[0]=0.1
p[1]=0.4
p[2]=0.3
p[3]=0.4

'''
index= np.where(p==np.max(p))
choice=np.zeros(4)
for i in range(len(choice)):
    if(i==index[0][0]):
        choice[i]=0.8 #greedy
    else:
        choice[i]=0.2/3
print(choice)
for i in range(30):
    action = np.random.choice(4, p=choice)
    print(action)
'''
def makeChoice(p):
    index= np.where(p==np.max(p))
    choice=np.zeros(4)
    for i in range(len(choice)):
        if(i==index[0][0]):
            choice[i]=0.8
        else:
            choice[i]=0.2/3
    action=np.random.choice(4,p=choice)
    #return np.random.choice(4,choice)
    return action
#print(makeChoice(p))
for i in range(30):
    print(makeChoice(p))