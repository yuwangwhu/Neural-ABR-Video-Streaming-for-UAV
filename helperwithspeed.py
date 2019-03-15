import random
import torch
from LoadDatabefore import *
from pylab import *

#Hyper Parameters
ThroughputSize = 8
ActionSize = 4
Hidden_size = 128
BitMin=300/1000
#miu =2
miu=2.6
lamada =0.45

def getThroughputData(epoch):
    global train_throughput
    global train_speed
    global train_distance
    global train_acce
    return train_throughput[epoch],train_speed[epoch],train_distance[epoch],train_acce[epoch]

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

def num_flat_features(x):
        # 由于默认批量输入，第零维度的batch剔除
    size = x.size()[1:]
    num_features = 1
    for s in size:
        num_features *= s
    return num_features



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



def Reward(BitRate,LastBitRate,Rebuffering):
    #return BitrateTransform(BitRate)-miu*Rebuffering-lamada*abs(BitrateTransform(BitRate)-BitrateTransform(LastBitRate))
    return math.log(BitrateTransform(BitRate)/BitMin) - miu*Rebuffering - lamada*abs(math.log(BitrateTransform(BitRate)/BitMin)-math.log(BitrateTransform(LastBitRate)/BitMin))

def Input(SyntheticData,TestSpeed,TestDistance,TestAcce,BufferSize,BitRate,TrainTime):

    #SyntheticData=getThroughputData()
    #ThroughPut=SyntheticData[2*TrainTime-2:2*TrainTime+6]
    ThroughPut = SyntheticData[TrainTime - 1:TrainTime + 7]
    ThroughPut =np.array(ThroughPut,dtype=np.float32)
    speed=np.array(TestSpeed[TrainTime+7:TrainTime+8],dtype=np.float32)
    distance = np.array(TestDistance[TrainTime + 7: TrainTime + 8], dtype=np.float32)
    acce = np.array(TestAcce[TrainTime + 7 :TrainTime + 8], dtype=np.float32)
    buffer=np.array([BufferSize],dtype=np.float32)
    action=np.array([BitRate],dtype=np.float32)
    networkIn= np.append(ThroughPut,buffer)
    networkIn = np.append(networkIn, action)
    networkIn =np.append(networkIn,speed)
    networkIn =np.append(networkIn,distance)
    networkIn =np.append(networkIn,acce)
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

#speed=np.array(speed_data[1][7:8],dtype=np.float32)
#print(speed)
#state=Input(train_data[1],speed_data[1],distance_data[1],acce_data[1],2.1,3,5)
#print(len(state))







