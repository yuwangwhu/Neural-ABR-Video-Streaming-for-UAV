import random
import torch
import numpy as np
from torch.autograd import Variable

from pylab import *

#Hyper Parameters
ThroughputSize = 8
ActionSize = 4
Hidden_size = 128
BitMin=300/1000
#miu =2
miu=2.6
lamada =0.9

N=4000
M=95 #47代表47组数据，95代表总的数据

#throughput_data=np.loadtxt('train/throuput.txt')
throughput_data=np.loadtxt('allData/throughput.txt')
prethroughput_data =np.zeros([M,50])


train_data =np.zeros([N,50])

for i in range(len(throughput_data)):
    max_data=np.max(throughput_data[i])
    min_data=np.min(throughput_data[i])
    for j in range(len(throughput_data[i])):
        prethroughput_data[i][j] = 2*throughput_data[i][j]/(max_data-min_data)+(0.5*max_data-2.5*min_data)/(max_data-min_data)

for i in range(N):
    train_data[i]=prethroughput_data[i%95]
#print(len(train_data))

def makeChoice(p):
    index= np.where(p==np.max(p))
    choice=np.zeros(4)
    for i in range(len(choice)):
        if(i==index[0][0]):
            choice[i]=0.9
        else:
            choice[i]=0.1/3
    action=np.random.choice(4,p=choice)
    #return np.random.choice(4,choice)
    return action

def getRandomData( ):
    data = []
    for i in range(130):
        data.append(random.uniform(0.5, 2.5))
    return np.array(data)

def getThroughputData(epoch):
    global train_data
    return train_data[epoch]

def num_flat_features(x):
        # 由于默认批量输入，第零维度的batch剔除
    size = x.size()[1:]
    num_features = 1
    for s in size:
        num_features *= s
    return num_features

#ThroughPut为1x8的numpy数组，buffersize为一个实数，bitrate为一个实数,返回值为一个1x1x10的variable类型的state
def getState(ThroughPut,bufferSize,bitRate):
    state1=Variable(torch.from_numpy(ThroughPut))
    state2=Variable(torch.from_numpy(np.array([bufferSize],dtype=np.float32 )))
    state2=state2.view(1,-1)
    state3=Variable(torch.from_numpy(np.array([bitRate],dtype=np.float32)))
    state3=state3.view(1,-1)
    state=torch.cat((state1,state2),1)
    state=torch.cat((state,state3),1)
    return state.view(1,1,-1)

def ActionData(action):
    data=torch.exp(action)
    bitrate = np.random.choice(ActionSize,p=data.cpu().data.numpy()[0])
    return bitrate

def BitrateTransform(action):
    if action==3:#表示视频为1080p，BitRate为2850kbps
        BitRate=2850/1000
        #BitRate = 3550 / 1000
    elif action ==2:#表示视频为720p，BitRate为1850kbps
        BitRate=1850/1000
        #BitRate = 2550 / 1000
    elif action==1:#表示视频为360p，BitRate为750kbps
        BitRate=750/1000
    else:#表示视频为240p，BitRate为300kbps
        BitRate=300/1000
    return BitRate

#对于每一个state而言，此函数实现的是在环境S下，选择bitrate为action情况下，更新buffer的过程
#输入为action表示比特率，throughput表示发送此chunk期间的throughput，TimeToPlay表示在Sn时刻距离播放的时间
#输出为下一state Sn的buffer，在Sn环境下选择action 导致rebuffer的时间
''' 
def updateBuffer(buffer,action,throughput,timetoplay=0):
    rebuffer =0
    if 2*BitrateTransform(action)/throughput > (buffer+timetoplay) : #在Sn环境下选择action会导致rebuffer
        while (buffer+timetoplay+rebuffer) <(2*BitrateTransform(action)/throughput):
            rebuffer = rebuffer + 0.5
        Newbuffer=2
        NewTimeToPlay =buffer + timetoplay + rebuffer - 2*BitrateTransform(action)/throughput
    elif 2*BitrateTransform(action)/throughput <(buffer+timetoplay) and 2*BitrateTransform(action)/throughput <timetoplay: #在Sn环境下选择action会导致rebuffer
        Newbuffer =buffer+2
        NewTimeToPlay =timetoplay - 2*BitrateTransform(action)/throughput
    else:#在Sn环境下选择action不会导致rebuffer
        Newbuffer =buffer+2-(2*BitrateTransform(action)/throughput-timetoplay)
        NewTimeToPlay = 0
    return Newbuffer,rebuffer,NewTimeToPlay
'''
def updateBuffer(buffer,action,throughput):

    if 2*BitrateTransform(action)/throughput < buffer : #不卡
        newbuffer =buffer +2- 2*BitrateTransform(action)/throughput
        rebuffering = 0
    elif 2*BitrateTransform(action)/throughput >buffer  and  buffer+2 > 2*BitrateTransform(action)/throughput: #不卡
        newbuffer = buffer +2- 2*BitrateTransform(action)/throughput
        rebuffering = 0
    else:
        newbuffer = 0
        rebuffering = (math.ceil((2*BitrateTransform(action)/throughput -2-buffer)/0.5))*0.5
    return newbuffer, rebuffering

''' 
th=random.uniform(0.3,1.5)
test1,test2= updateBuffer(2.6,3,th)
print("newbuffer  is {0},rebuffer is {1},throughput is {2}".format(test1,test2,th))

th=random.uniform(0.3,1.5)
test1,test2= updateBuffer(2.6,1,th)
print("newbuffer  is {0},rebuffer is {1}throughput is {2}".format(test1,test2,th))

#th=random.uniform(1,9)
th=0.5
test1,test2= updateBuffer(2.6,1,th)
print("newbuffer  is {0},rebuffer is {1}throughput is {2}".format(test1,test2,th))
'''

def Reward(BitRate,LastBitRate,Rebuffering):
    #return BitrateTransform(BitRate)-miu*Rebuffering-lamada*abs(BitrateTransform(BitRate)-BitrateTransform(LastBitRate))
    return math.log(BitrateTransform(BitRate)/BitMin) - miu*Rebuffering - lamada*abs(math.log(BitrateTransform(BitRate)/BitMin)-math.log(BitrateTransform(LastBitRate)/BitMin))

def Input(SyntheticData,BufferSize,BitRate,TrainTime):
    #SyntheticData=getThroughputData()
    #ThroughPut=SyntheticData[2*TrainTime-2:2*TrainTime+6]
    ThroughPut = SyntheticData[TrainTime - 1:TrainTime + 7]
    ThroughPut =np.array(ThroughPut,dtype=np.float32)
    buffer=np.array([BufferSize],dtype=np.float32)
    action=np.array([BitRate],dtype=np.float32)
    networkIn= np.append(ThroughPut,buffer)
    networkIn = np.append(networkIn, action)
    return networkIn

def smooth(loss,stride=10):
    smooth_loss=[]
    for i in range(int(len(loss)/stride)):
        sum=0
        for j in range(stride):
            sum=sum+loss[i*stride+j]
        #sum=loss[10*i]+loss[10*i+1]+loss[10*i+2]+loss[10*i+3]+loss[10*i+4]+loss[10*i+5]+loss[10*i+6]+loss[10*i+7]+loss[10*i+8]+loss[10*i+9]
        smooth_loss.append(sum/stride)
    return smooth_loss







