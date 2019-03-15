import numpy as np
from pylab import *
import random

N=7000
M=95

pre_throughput=np.loadtxt('allData/throughput.txt')
#print(len(pre_throughput))
speed =np.loadtxt('allData/speed.txt')
distance =np.loadtxt('allData/distance1.txt')
acce=np.loadtxt('allData/acce.txt')

#缩小
throughput=np.zeros([M,50])
for i in range(len(throughput)):
    min_data=np.min(pre_throughput[i])
    max_data=np.max(pre_throughput[i])
    for j in range(len(throughput[0])):
        throughput[i][j]=2*pre_throughput[i][j]/(max_data-min_data)+(0.5*max_data-2.5*min_data)/(max_data-min_data)


train_throughput=np.zeros([N,50])
train_speed=np.zeros([N,50])
train_distance=np.zeros([N,50])
train_acce=np.zeros([N,50])

for  i in range(len(train_throughput)):
    train_throughput[i]=throughput[i%M]
    train_speed[i]=speed[i%M]
    train_distance[i]=distance[i%M]
    train_acce[i] =acce[i%M]