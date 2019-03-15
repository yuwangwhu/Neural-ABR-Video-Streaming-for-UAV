import numpy as np
import matplotlib.pyplot as plt
from pylab import *

#load throughputData
throughput=np.loadtxt('train/throuput.txt')
print("the size of throughput",len(throughput))
print("the size of each throughput",len(throughput[0]))

all=np.loadtxt('allData/throughput.txt')
print(len(all))

'''
#load speedData
speed=np.loadtxt('speed_2.txt')
print("the size of throughput",len(speed))
print("the size of each throughput",len(speed[0]))

#load distanceData
distance =np.loadtxt("distance_2.txt")
print("the size of distance",len(distance))
print("the size of each distance",len(distance[0]))

#load acceData
acce =np.loadtxt("acce_2.txt")
print("the size of acce",len(acce))
print("the size of each acce",len(acce[0]))

#polt the data
plotId =2
figure(1)
title("the throughput data")
plot(throughput[plotId])

figure(2)
title("the speed data")
plot(speed[plotId])

figure(3)
title("the distance data")
plot(distance[plotId])

figure(4)
title("the acce data")
plot(acce[0])
show()
'''