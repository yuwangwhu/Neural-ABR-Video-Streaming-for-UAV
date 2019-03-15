import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import math
import random
from helper import *
gama=0.9
total_times =40

#不加任何传感器信息

#定义网络
class ActorNetwork(nn.Module):
    def __init__(self,input_channels=1,output_channels=128,output_size=4):
        super(ActorNetwork ,self ).__init__()
        self.cov1 =nn.Conv1d(input_channels,64,kernel_size=3)
        self.cov2 =nn.Conv1d(64,128,kernel_size=3)
        self.fc1= nn.Linear(514,320)
        self.fc2= nn.Linear(320,64)
        self.fc3= nn.Linear(64,8)
        self.fc4= nn.Linear(8,output_size)
    def forward(self,x):#输入1x1x10
        #in_size=x.size(0)
        x1 = np.zeros([8], dtype=np.float32)
        for i in range(8):
            x1[i] = x[i]
        x2 = np.array([x[8]], dtype=np.float32)
        x3 = np.array([x[9]], dtype=np.float32)
        x1 = Variable(torch.from_numpy(x1))
        x2 = Variable(torch.from_numpy(x2))
        x3 = Variable(torch.from_numpy(x3))
        x1 = x1.view(1, 1, -1)
        x2 = x2.view(1, -1)
        x3 = x3.view(1, -1)

        x1 = self.cov1(x1)
        x1 = self.cov2(x1)
        x1 = x1.view(-1, num_flat_features(x1))
        datain = torch.cat((x1, x2), 1)
        datain = torch.cat((datain, x3), 1)
        out = F.relu(self.fc1(datain))
        out =F.relu(self.fc2(out))
        out =F.relu(self.fc3(out))
        out = self.fc4(out)
        return F.log_softmax(out)

class ValueNetwork(nn.Module):
    def __init__(self,input_channels=1,output_channels=128,output_size=1):
        super(ValueNetwork  ,self ).__init__()
        self.cov =nn.Conv1d(input_channels,64,kernel_size=3)
        self.cov2=nn.Conv1d(64,128,kernel_size=3)
        self.fc1=nn.Linear(514,320)
        self.fc2=nn.Linear(320,64)
        self.fc3=nn.Linear(64,8)
        self.fc4=nn.Linear(8,output_size)
    def forward(self,x):#输入1x1x10
        #in_size=x.size(0)
        x1 = np.zeros([8], dtype=np.float32)
        for i in range(8):
            x1[i] = x[i]
        x2 = np.array([x[8]], dtype=np.float32)
        x3 = np.array([x[9]], dtype=np.float32)
        x1 = Variable(torch.from_numpy(x1))
        x2 = Variable(torch.from_numpy(x2))
        x3 = Variable(torch.from_numpy(x3))
        x1 = x1.view(1, 1, -1)
        x2 = x2.view(1, -1)
        x3 = x3.view(1, -1)

        x1 = self.cov(x1)
        x1 = self.cov2(x1)
        x1 = x1.view(-1, num_flat_features(x1))
        datain = torch.cat((x1, x2), 1)
        datain = torch.cat((datain, x3), 1)
        out=F.relu(self.fc1(datain))
        out=F.relu(self.fc2(out))
        out=F.relu(self.fc3(out))
        out=self.fc4(out)
        return out

def roll_out(actor_network,value_network,TestThroughput):
    #initial
    CurrentBufferSize =0
    LastBitRate = 0
    train_time =1
    initial_state=Input(TestThroughput , CurrentBufferSize, LastBitRate, train_time)
    state=initial_state

    #return data
    states=[]
    actions =[]
    rewards =[]
    buffers=[]
    action_all=[]
    buffers.append(CurrentBufferSize)
    rebuffer_all=[]
    #is_done =False
    final_r =0

    for j in range(total_times):
        states.append(state)
        log_softmax_action =actor_network(state)
        softmax_action =torch.exp(log_softmax_action)
        action=np.random.choice(4,p=softmax_action.cpu().data.numpy()[0])
        #action = makeChoice(softmax_action.cpu().data.numpy()[0])
        print(action)
        action_all.append(action)
        one_hot_action=[int (k==action) for k in range(4)]
        throughput=TestThroughput[train_time+8]
        CurrentBufferSize,rebuffer =updateBuffer(CurrentBufferSize,action,throughput)
        buffers.append(CurrentBufferSize)
        rebuffer_all.append(rebuffer)

        reward=Reward(action,LastBitRate,rebuffer)
        train_time =train_time+1
        LastBitRate = action
        next_state =Input(TestThroughput , CurrentBufferSize, LastBitRate, train_time)
        final_state=next_state
        state=next_state

        actions.append(one_hot_action)
        rewards.append(reward)
        if (j==total_times-1):
            last_softmax_action=actor_network(final_state)
            last_action=torch.exp(last_softmax_action)
            last_choose_action=np.random.choice(4,p=last_action.cpu().data.numpy()[0])
            last_throughput = TestThroughput[train_time + 8]
            last_buffer,last_rebuffer=updateBuffer(CurrentBufferSize,last_choose_action,last_throughput)
            final_r=Reward(last_action,LastBitRate,last_rebuffer)
            '''
            if(last_rebuffer==0):
                final_r=15
            else:
                final_r=0
            '''
    #final_r=value_network(final_state)
    #final_r=final_r.data.numpy()[0][0]
    return states,actions,rewards,buffers,final_r,action_all,rebuffer_all

def discount_reward(r,gama,final_r):
    discounted_r =np.zeros_like(r)
    running_add =final_r
    for t in reversed(range(0,len(r))):
        running_add=running_add*gama+r[t]
        discounted_r[t]=running_add
    return discounted_r

def main():

    #initial value network
    #decayed_learning_rate = 0.001
    value_network=ValueNetwork()
    print(value_network )
    #value_network_optim=torch.optim.Adam(value_network.parameters(),lr=decayed_learning_rate)

    #initial actor network
    actor_network=ActorNetwork()
    print(actor_network)
    #actor_network_optim =torch.optim.Adam(actor_network.parameters(),lr =0.1)
    #actor_network_optim = torch.optim.RMSprop(actor_network.parameters(), lr=0.00005)

    #initial throughput
    #TestThroughput = getThroughputData()

    steps =[]
    task_episode =[]
    test_result = []
    total_valueloss=[]
    total_actorloss=[]
    x=[]
    total_reward = []
    #initial the learning-rate
    #decayed_learning_rate_value = 0.01 # 0.001 is the best
    decayed_learning_rate_value = 0.01# 0.01 is the best
    #decayed_learning_rate_actor = 0.004
    decayed_learning_rate_actor = 0.00001#0.00001  is the best

    test_data=getThroughputData(2)
    test_reward=[]
    test_buffer1=[]
    test_buffer2=[]
    train_before=[]
    train_after=[]

    print("start")
    maxReward=-100
    maxepoch=0
    before_action=[]
    after_action=[]
    count=0
    for step in range (N):

        print("epoch",step)

        if (step == 0):
            for i in range(M):
                test1 = getThroughputData(i)
                _, _, rewards, this_buffers, final_r, action_all,_ = roll_out(actor_network, value_network, test1)
                train_before.append(sum(rewards) + final_r)
        '''
        if (step == N - 2):
            for i in range(M):
                test1= getThroughputData(i)
                _, _, rewards, this_buffers, final_r, action_all,_ = roll_out(actor_network, value_network, test1)
                train_after.append(sum(rewards) + final_r)
        '''
        #if (step)%300==0:
            #decayed_learning_rate_value = 0.005 * (0.8 **(step / 100))
            #decayed_learning_rate_value =decayed_learning_rate_value* 0.88

        #if (step)%150==0:
            #decayed_learning_rate_actor =decayed_learning_rate_actor * 0.9

        #TestThroughput = getThroughputData()
        TestThroughput = getThroughputData(step)
        #TestThroughput = getThroughputData(0)
        #TestThroughput =getRandomData()
        value_network_optim = torch.optim.Adam(value_network.parameters(), lr=decayed_learning_rate_value)
        #actor_network_optim = torch.optim.Adam(actor_network.parameters(), lr=decayed_learning_rate_actor)
        actor_network_optim = torch.optim.Adam(actor_network.parameters(), lr=decayed_learning_rate_actor)
        states,actions,rewards,buffers,final_r,_,_=roll_out(actor_network,value_network,TestThroughput)
        #print(rewards)
        #new_states=[]
        data=sum(rewards)+final_r
        total_reward.append(data)
        #actor_network_optim = torch.optim.RMSprop(actor_network.parameters(), lr=decayed_learning_rate_ac

        #将states转换成numpy数组，一边转换成variable
        new_states=np.zeros([len(states),len(states[0])],dtype=np.float32)
        #print(new_states.shape)
        for i in range(len(states)):
            for j in range(len(states[i])):
                new_states[i][j]=states[i][j]

            #print(actions)
            #print(states)
            #print(new_states)

            #print(new_states)
            #print(actions)
        actions_var =Variable(torch.Tensor(actions).view(-1,4))
        states_var  =Variable(torch.from_numpy(new_states))
        #print(actions_var)
        #print(states_var)

        #train actor network
        actor_network_optim.zero_grad()
        log_softmax_actions=actor_network(states[0])

        for i in range(1,len(states)):
            log_softmax_actions =torch.cat((log_softmax_actions,actor_network(states[i])),0)
        #print(log_softmax_actions )
        vs=value_network(states[0])
        for i in range(1,len(states)):
                vs=torch.cat((vs,value_network(states[i])),0)
        #print(vs)
        vs=vs.detach()
        #caculate qs
        qs= Variable(torch.Tensor(discount_reward(rewards,0.99,final_r)))

        advantages=qs-vs
        # print(advantages)
        actor_network_loss =-torch.mean(torch.sum(log_softmax_actions*actions_var,1)* advantages)
        #print(actor_network_loss)
        #print(log_softmax_actions)
        total_actorloss.append(actor_network_loss.cpu().data.numpy()[0])
        actor_network_loss.backward()
        torch.nn.utils.clip_grad_norm(actor_network.parameters(), 0.5)
        actor_network_optim.step()

        #train value network
        value_network_optim.zero_grad()
        target_value =qs
        values=value_network(states[0])
        for i in range(1,len(states)):
            values=torch.cat((values,value_network(states[i])),0)
        #print(values)
        criterion =nn.MSELoss()
        value_network_loss =criterion(values,target_value)
        value_network_loss.backward()
        torch.nn.utils.clip_grad_norm(value_network.parameters(), 0.5)
        value_network_optim.step()
        #print(value_network_loss)
        total_valueloss.append(value_network_loss.cpu().data.numpy()[0])
        x.append(step)

        #test and caculate the reward everytime
        _, _, rewards, this_buffers,final_r,action_all,test_rebuffer= roll_out(actor_network, value_network, test_data)

        test_reward.append(sum(rewards)+final_r)
        print(sum(rewards) + final_r)
        print(value_network_loss.cpu().data.numpy()[0])
        print(actor_network_loss.cpu().data.numpy()[0])
        if(sum(rewards)>25):
            count=count+1

        if count==20:
            for i in range(M):
                test1 = getThroughputData(i)
                _, _, rewards, this_buffers, final_r, action_all, _ = roll_out(actor_network, value_network, test1)
                train_after.append(sum(rewards) + final_r)
            break

        if (sum(rewards) > maxReward):

            maxReward = sum(rewards)
            maxepoch = step
            test_buffer2 = this_buffers
            choose_action = action_all
            print(maxReward)
            print(maxepoch)
            print("action")
            print(choose_action)
            print("every rebuffer")
            print(test_rebuffer)
            print("every reward")
            print(rewards)
            print("the buffer")
            print(this_buffers)
        if (step == 0):
            test_buffer1=this_buffers
            print("action ")
            print(action_all)
            print("every rebuffer")
            print(test_rebuffer)
            print("every reward")
            print(rewards)
            print(" the buffer")
            print(this_buffers)

        #最后一次的buffer

        if ((step>N-100 and sum(rewards)>30) or step==N-10 ):
            #test_buffer2=this_buffers
            #choose_action=action_all
            print("end training")


    #save the model
    torch.save(actor_network,'actor.pkl')
    torch.save(value_network,'value.pkl')
    print(len(total_valueloss))
    print(len(total_valueloss))

    print(len(test_reward))
    print(maxReward)
    print(maxepoch)
    #print(test_reward)

    #plt.plot(x,total_reward )
    #plt.plot(x, total_actorloss)
    #plt.show()

    figure(1)
    title("total_valueloss")
    plot(x,total_valueloss)

    figure(2)
    title("total_actorloss")
    plot(x,total_actorloss)

    figure(3)
    title("smooth_valueloss")
    plot(smooth(total_valueloss))
    figure(4)
    title("smoth_actorloss")
    plot(smooth(total_actorloss))


    figure(5)
    plot(test_reward)
    figure(6)
    plot(smooth(test_reward,stride=20))
    figure(7)
    plot(test_buffer1)
    figure(8)

    plot(test_buffer2)
    figure(9)
    plot(test_data)

    figure(10)
    plot(choose_action)
    plot(before_action)
    figure(11)
    plot(train_before)
    plot(train_after)

    show()



if __name__ =='__main__':
    main()


