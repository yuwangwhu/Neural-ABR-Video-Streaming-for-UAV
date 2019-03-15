这部分详细讲解我们实现的自适应算法的代码实现部分，主要包括系统信息获取部分、网络部分以及网络更新部分。其中系统信息获取部分包括导入训练数据及数据预处理、得到网络输入、更新视频缓存、获取奖励等；网络部分主要是上一节提到的actor-critic网络的搭建；网络更新部分主要是根据强化学习算法部分多网络进行优化更新，这也是我们的重点。
我们在实际实现时，采取了CNN、CNN+传感器、RNN+传感器三种方式的网络，三者在大体上差不多，因此在这我们仅介绍前面谈到过的CNN+传感器的网络实现。如果想看完整的代码和采集的无人机飞行中吞吐量与速度、加速度、距离之间的数据信息，请去我的github.
下面是我们系统部分部分代码的结构： 

其中文件夹train_data里面是采集的数据，readData、py导入训练数据及数据预处理，helperwithspeed、py包含系统网络输入获取、缓存更新、获取奖励等功能、train_with_speed、py主要是网络的搭建以及网络的更新。下面将详细讲解每个部分的具体功能以及实现。

## 1、数据部分：
包括测量的吞吐量throughput.txt以及对应的速度信息speed.txt、距离信息distance.txt、加速度信息acce.txt.后续数据还在继续扩充。 

吞吐量中一共95组数据，每一组数据包括50s的吞吐量大小，大小在0到16Mps之间。 

加速度一共95组，每一组是与吞吐量相对应的50s的加速度大小，通过预处理，我们将加速度分为0和1两个等级，当实际加速度超过18.5m/s^2的时候将加速度置为1，反之则为0. 

距离一共95组，每一组是与吞吐量对应的50m的距离大小，通过预处理，将加速度分为0和1两个等级，当与接收端距离超过50m时将距离置为1，反之则为0 

速度也是95组，每一组是与吞吐量对应的50s的速度大小，通过预处理，将速度划分为三个等级，速度不超过8m/s的置为0，8-12m/s的置为1，超过12m/s的置为2. 

## 2、数据导入及预处理
这部分主要是从txt中导入数据以及进行数据归一化预处理。

数据的读取比较简单，用np.loadtxt()就可以直接将txt读取为numpy数组。 

接下来是数据的归一化，在这里我们将吞吐量归一化到0.5-2.5之间。 

最后，我们将数据进行重复，得到N组数据，即train_throughput、train_speed、train_distance、train_acce,他们都是维数为N*50的二维数组，其中，N为我们训练的次数，在实验中我们可以根据需要进行自行调整，在这里，我们暂取N=10000. 

## 3.网络搭建
网络的搭建和深度学习中的网络类似，唯一需要注意的是在这里两个有两个独立的网络：actor和critic网络。前面已经谈到过，这两个网络的输入部分相同，都是输入的相应的状态，即一个1x13的numpy数组。具体网络搭建的代码如下：
``` python 	
	class ActorNetwork(nn.Module):
	    def __init__(self):
	        super(ActorNetwork, self).__init__()
	
	        self.rnn = nn.LSTM(
	            input_size=2,
	            hidden_size=64,
	            num_layers=2,
	            batch_first=True,
	        )
	
	        self.fc1 = nn.Linear(69, 30)
	        self.fc2 = nn.Linear(30, 10)
	        self.fc3 = nn.Linear(10, 4)
	        self.relu = nn.ReLU()
	
	    def forward(self, x):  # x shape(1x1x13)
	        x1 = np.zeros([8], dtype=np.float32)  # 代表throughput
	        for i in range(8):
	         x1[i] = x[i]
	
	        x2 = np.array([x[8]], dtype=np.float32)  # 代表buffersize
	        x3 = np.array([x[9]], dtype=np.float32)  # 代表lastAction
	        x4 = np.array([x[10]], dtype=np.float32)  # 代表speed
	        x5 = np.array([x[11]], dtype=np.float32)  # 代表distance
	        x6 = np.array([x[12]], dtype=np.float32)  # 代表acce
	
	        x1 = Variable(torch.from_numpy(x1))
	        x2 = Variable(torch.from_numpy(x2))
	        x3 = Variable(torch.from_numpy(x3))
	        x4 = Variable(torch.from_numpy(x4))
	        x5 = Variable(torch.from_numpy(x5))
	        x6 = Variable(torch.from_numpy(x6))
	
	        x1 = x1.view(-1, 4, 2)
	        x2 = x2.view(1, -1)
	        x3 = x3.view(1, -1)
	        x4 = x4.view(1, -1)
	        x5 = x5.view(1, -1)
	        x6 = x6.view(1, -1)
	
	        r_out, (h_n, h_c) = self.rnn(x1, None)
	
	        #r_out = r_out.view(-1, num_flat_features(r_out))
	        datain = torch.cat((r_out[:,-1,:], x2), 1)
	        datain = torch.cat((datain, x3), 1)
	        datain = torch.cat((datain, x4), 1)
	        datain = torch.cat((datain, x5), 1)
	        datain = torch.cat((datain, x6), 1)
	
	        out = self.relu(self.fc1(datain))
	        out = self.relu(self.fc2(out))
	        out = self.fc3(out)
	        return F.log_softmax(out)

``` 

上面是actor网络部分的实现，网络的输入是每时刻的状态，是一个1x13维的numpy数组，输出为1x4维的tesnor，标识的是状态动作概率。需要注意的是，这里输出的是log_softmax处理后的概率，在具体使用时需要进行指数处理。

``` python 
	class ValueNetwork(nn.Module):
	    def __init__(self,input_channels=1,output_channels=128,output_size=1):
	        super(ValueNetwork  ,self ).__init__()
	        self.cov1 = nn.Conv1d(input_channels, 64, kernel_size=3)
	        self.cov2 = nn.Conv1d(64, output_channels, kernel_size=3)
	        self.fc1 = nn.Linear(517, 128)
	        self.fc2 = nn.Linear(128, 30)
	        self.fc3 = nn.Linear(30, 8)
	        self.fc4 = nn.Linear(8, output_size)
	        self.drop = nn.Dropout(p=0.5)
	    def forward(self,x):#输入1x1x13
	        #in_size=x.size(0)
	        x1 = np.zeros([8], dtype=np.float32) #代表throughput
	        for i in range(8):
	            x1[i] = x[i]
	        x2 = np.array([x[8]], dtype=np.float32) #代表buffersize
	        x3 = np.array([x[9]], dtype=np.float32) #代表lastAction
	        x4 = np.array([x[10]],dtype=np.float32) #代表speed
	        x5 =np.array([[x[11]]],dtype=np.float32) #代表distance
	        x6 =np.array([x[12]],dtype=np.float32) #代表acce
	
	        x1 = Variable(torch.from_numpy(x1))
	        x2 = Variable(torch.from_numpy(x2))
	        x3 = Variable(torch.from_numpy(x3))
	        x4= Variable(torch.from_numpy(x4))
	        x5 = Variable(torch.from_numpy(x5))
	        x6 = Variable(torch.from_numpy(x6))
	
	        x1 = x1.view(1, 1, -1)
	        x2 = x2.view(1, -1)
	        x3 = x3.view(1, -1)
	        x4 = x4.view(1, -1)
	        x5 = x5.view(1, -1)
	        x6 = x6.view(1, -1)
	
	
	        x1 = self.cov1(x1)
	        x1= self.cov2(x1)
	        x1 = x1.view(-1, num_flat_features(x1))
	        datain = torch.cat((x1, x2), 1)
	        datain = torch.cat((datain, x3), 1)
	        datain = torch.cat((datain, x4), 1)
	        datain = torch.cat((datain, x5), 1)
	        datain = torch.cat((datain, x6), 1)
	
	        out = F.relu(self.fc1(datain))
	        out = self.drop(F.relu(self.fc2(out)))
	        out = F.relu(self.fc3(out))
	        out = self.fc4(out)
	        return out

```
 
上面是critic网路部分的实现，输入的也是每时刻的1x13维的状态矩阵，输出的是1x1维的tesnor，表示的是每个状态下的状态价值V(s).



从结构上来说，actor网络和critic网络的输入以及网络结构都是相同的，均是2层1维CNN后在连接三层全连层网络。需要注意并不是将输入直接进行卷积处理，因为我们卷积的只是输入的一部分，因此需要进行数据的拆分和合并处理。在这里，还需要注意的是tensor,variable以及numpy数组的转换。

## 4、系统状态获取及更新
系统状态获取以及更新也是整个系统比较重要的部分，其中主要包括获训练数据，获取送入网络的数据，状态更新以及计算奖励。下面我主要介绍这部分的相关函数。

函数getThroughput比较简单，输入参数是训练的epoch，返回参数是这次训练整个过程中的吞吐量以及对应的速度、加速度以及距离信息。代码如下：
``` python 
	def getThroughputData(epoch):
	    global train_throughput
	    global train_speed
	    global train_distance
	    global train_acce
	    return train_throughput[epoch],train_speed[epoch],train_distance[epoch],train_acce[epoch]

```  



函数Input实现的是根据上面得到的每个epoch的数据，在每个视频块需要播放的时候送往actor-critic网络的表示状态的1x13维的numpy数组。状态是由过去八个视频块的吞吐量、此刻的视频缓存、上一视频块的比特率、此刻的速度、距离以及加速度组成。具体实现的代码如下：

``` python 
	def Input(SyntheticData,TestSpeed,TestDistance,TestAcce,BufferSize,BitRate,TrainTime):
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

``` 



函数UpdateBuffer实现的是在状态s下，选择特定比特率action之后到达下一个状态，此过程中buffer的更新以及此action导致的卡顿时间rebuffering.需要注意的是我们在这里提到的action都是用{0，1，2，3}来表示{300kbps,750kpbs,1850kbps,2850kbps}的视频比特率的，在具体计算的时候需要通过函数BiterateTransform来进行转换一下，这一函数很简单，我们就不做描述。具体代码如下：

``` python 
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

``` 


函数Reward是根据选择视频块的BitRate以及这一BitRate造成的卡顿时间rebuffering和上一视频块的比特率来计算的，计算的是根据下面公式算出的视频的QoE,也就是我们系统中的奖励值。



具体实现的代码如下，实现思路就是比较视频缓存大小和下载该视频块的时间进行比较来分类讨论进行的。

``` python 
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

``` 

函数discount_reward是根据某次播放一段视频每个状态选择动作后得到的奖励值r、最后一步的奖励值final_r以及折扣因子gama来计算这段视频中每个状态的动作价值函数的，计算的依据是下面动作价值函数的定义：


具体实现的代码如下：
``` python 
	def discount_reward(r,gama,final_r):
	    discounted_r =np.zeros_like(r)
	    running_add =final_r
	    for t in reversed(range(0,len(r))):
	    running_add=running_add*gama+r[t]
	    discounted_r[t]=running_add
	    return discounted_r

``` 


## 5.网络优化更新
前面我们已经做好了所有的准备工作，我们大致实现了如下功能：

每一次你训练时得到一段50s包含吞吐量以及对应速度、加速度、距离的训练数据。这一组数据对应一段完整的视频播放过程，需要进行42次视频比特率的选择。

对于每一组训练数据，我们在每个视频块需要选择比特率的时候得到此刻的状态信息，将它合并在一个1x13维的numpy数组中。
搭建好actor-critic网络，对于每一个视频块，输入此刻的状态，actor网络可以输出动作概率用来选择概率，value网络输出此状态下的状态价值V(st)。
选择比特率后，我们能达到下一状态并进行状态的更新以及获得即时奖励r。
计算每一个状态的动作价值函数Q(st,at)
接下来我们需要做的是根据这一过程进行网络的优化更新，这也是我们的重中之重。更新过程主要分为两部分，第一部分是根据每个epoch得到的数据进行一次完整的视频播放。具体代码如下：
``` python 
	def roll_out(actor_network,value_network,TestThroughput,TestSpeed,TestDistance,TestAcce):
	#initial
	    CurrentBufferSize =0
	    LastBitRate = 0
	    train_time =1
	    initial_state=Input(TestThroughput,TestSpeed,TestDistance,TestAcce, CurrentBufferSize, LastBitRate, train_time)
	    state=initial_state
	
	    #return data
	    states=[]
	    actions =[]
	    rewards =[]
	    buffers=[]
	    rebuffer_all=[]
	    action_all=[]
	    buffers.append(CurrentBufferSize)
	    #is_done =False
	    final_r =0
	
	    for j in range(total_times-1):
	        states.append(state)
	        log_softmax_action =actor_network(state)
	        softmax_action =torch.exp(log_softmax_action)
	        action=np.random.choice(4,p=softmax_action.cpu().data.numpy()[0])
	        #action=makeChoice(softmax_action.cpu().data.numpy()[0])
	        print(action)
	        action_all.append(action)
	        one_hot_action=[int (k==action) for k in range(4)]
	        throughput=TestThroughput[train_time+8]
	        CurrentBufferSize,rebuffer =updateBuffer(CurrentBufferSize,action,throughput)
	        rebuffer_all.append(rebuffer)
	        buffers.append(CurrentBufferSize)
	
	        reward=Reward(action,LastBitRate,rebuffer)
	        LastBitRate = action
	        train_time =train_time+1
	        next_state =Input(TestThroughput,TestSpeed,TestDistance,TestAcce, CurrentBufferSize, LastBitRate, train_time)
	        final_state=next_state
	        state=next_state
	
	        actions.append(one_hot_action)
	        rewards.append(reward)
	
	        if (j == total_times - 1):
	            last_softmax_action = actor_network(final_state)
	            last_action = torch.exp(last_softmax_action)
	            last_choose_action = np.random.choice(4, p=last_action.cpu().data.numpy()[0])
	            last_throughput = TestThroughput[train_time + 8]
	            last_buffer, last_rebuffer = updateBuffer(CurrentBufferSize, last_choose_action, last_throughput)
	            final_r = Reward(last_action, LastBitRate, last_rebuffer)
	    return states,actions,rewards,buffers,final_r,action_all,rebuffer_all

``` 

这一部分的思路是在每个视频快获取到状态信息，然后送入actor网络根据网络输出选择特定的比特率,然后更新到下一个状态并计算出此选择的奖励值，这样循环下去直到这个视频播放完毕。在这个过程中我们只进行选择但不进行网络更新，返回值主要包括以下参数： 这一段视频中的所有状态信息（以矩阵的形式存储在states中）、每个状态下选择的动作的one_hot编码表示（以矩阵的形式存储在actions中，每个元素都是用[1，0，0，0]、[0，1，0，0]、[0，0，1，0]、[0，0，0，1]表示）、每个状态下的动作（以矩阵的形式存储在action_all中，每个元素都是0，1，2，3来表示不同的比特率)、每个状态下做出动作后的奖励（以矩阵形式存储在rewards中）、最后一个状态的奖励（final_r）以及每个状态下的卡段时间（以矩阵形式存储在rebuffer_all中。

上面得到了一个完整视频播放的所有信息，接下来就可以进行一次更新了。更新分为actor网络的更新和value网络的更新。更新的核心代码如下：
``` python 
	train_throughput,train_speed,train_distance,train_acce= getThroughputData(step)
	value_network_optim = torch.optim.Adam(value_network.parameters(), lr=decayed_learning_rate_value)
	actor_network_optim = torch.optim.Adam(actor_network.parameters(), lr=decayed_learning_rate_actor)
	states, actions, rewards, buffers, final_r, _,_ = roll_out(actor_network, value_network, train_throughput,train_speed,train_distance,train_acce)
	data=sum(rewards)
	total_reward.append(data)
	
	new_states=np.zeros([len(states),len(states[0])],dtype=np.float32)
	
	for i in range(len(states)):
	    for j in range(len(states[i])):
	        new_states[i][j]=states[i][j]
	
	actions_var=Variable(torch.Tensor(actions).view(-1,4))
	states_var= Variable(torch.from_numpy(new_states))
	#train actor_network
	actor_network_optim.zero_grad()
	log_softmax_actions=actor_network(states[0])
	for i in range(1,len(states)):
	    log_softmax_actions=torch.cat((log_softmax_actions,actor_network(states[i])),0)
	
	vs=value_network(states[0]
	for i in range(1,len(states)):
	    vs=torch.cat((vs,value_network(states[i])),0)
	vs=vs.detach()
	qs=Variable(torch.Tensor(discount_reward(rewards,0.99,final_r)))
	advantages= qs-vs
	actor_network_loss=-torch.mean(torch.sum(log_softmax_actions*actions_var,1)*advantages)
	total_actorloss.append(actor_network_loss.cpu().data.numpy())
	actor_network_loss.backward()
	torch.nn.utils.clip_grad_norm(actor_network.parameters(),0.5)
	actor_network_optim.step()
	
	#train value_network
	value_network_optim.zero_grad()
	target_value =qs
	values=value_network(states[0])
	for i in range(1,len(states)):
	    values=torch.cat((values,value_network(states[i])),0)
	
	criterion =nn.MSELoss()       
	target_value  = target_value.view(-1,1)
	#print(target_value.size())
	value_network_loss=criterion(values,target_value)
	value_network_loss.backward()
	torch.nn.utils.clip_grad_norm(value_network.parameters(),0.5)
	value_network_optim.step()
	
	total_valueloss.append(value_network_loss.cpu().data.numpy())

``` 
actor网络是根据梯度上升来进行更新的。



actor网络更新需要log_sofmax动作概率以及优势函数A(st,at)=Q(st,at)-V(st,w).其中log_softmax动作概率由action网络得到，Q(st,at)由实际选择获得的即时奖励与折扣因子计算得到，V（st,w）由critic网络得到。

critic网络根据梯度下降来进行更新的。



critic网络更新需要即时奖励rt，下一个状态的价值函数V(st+1)以及这个状态的状态价值V(st).其中rt是根据实际值得到的QoE,V(st+1)和V(st)是根据critic网络得到的。

上面完成的就是一次完整的视频播放及更新过程。在设置好训练次数进行多次的更新即可达到稳定值。在这个心目中，我们大概训练15000次左右就能达到稳定的reward值。我们在训练过程中主要是根据actor-loss和value-loss来进行网络参数的调整，经过尝试，对网络效果影响最大的参数是actor和value网络的学习率，最终，我们发现两者在0.00003和0.01左右事能达到较好的效果。

在下一节，我们主要展示训练得到的结果以及将要介绍一下其他自适应比特率算法。

本节是项目的重点，中间网络的训练我们采取了较为简单的actor-critic网络来训练，后续我们将在网络结构以及考虑用异步actor-critic（A3C）算法来提升训练速度和效果。
