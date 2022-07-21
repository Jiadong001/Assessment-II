'''模型搭建

    input: 氨基酸序列的BLOSUM50编码
    output: 一个标量, 数值接近1表示binding, 数值接近0表示non-binding

'''

import torch
import torch.nn as nn
import torch.nn.functional as F                 #可用于调激活函数


'''model 1 --- NetMHCpan1.0'''
class NetMHCpan_10(nn.Module):                  #class 子类（父类）-- "subclass"

    def __init__(self, input_size, hidden_size, output_size):
        
        super(NetMHCpan_10, self).__init__()            #继承构造

        #hidden layer
        self.f1 = nn.Linear(input_size, hidden_size)    #全连接
        self.relu1 = nn.ReLU()                          #ReLU函数

        #a single neroun output layer
        self.f2 = nn.Linear(hidden_size, output_size)


    def forward(self, input):
        '''
            input: [1, pHLA_code_length]  (1 = batch_size)
            output: [1, 1]
        '''

        output = self.f1(input)
        output = self.relu1(output)

        output = self.f2(output)                #可以在后面加上 sigmod 将结果限制在（0，1）
#        output = F.softmax(self.f2(output)) for 2 neroun

        return output


'''model 2 -- NetMHCpan1.0_LSTM'''
class NetMHCpan_10_LSTM(nn.Module):                 

    def __init__(self, input_size_pep, input_size_HLA, lstm_hidden_size, hidden_size, output_size):
        
        super(NetMHCpan_10_LSTM, self).__init__()              

        #2 LSTM layers
        self.lstm_pep = nn.LSTM(input_size = input_size_pep, hidden_size = lstm_hidden_size, num_layers = 1)    #1层
        self.lstm_HLA = nn.LSTM(input_size = input_size_HLA, hidden_size = lstm_hidden_size, num_layers = 1)

        #hidden layer
        self.f1 = nn.Linear(lstm_hidden_size*2, hidden_size)    #全连接(dense)，输入是LSTM输出的结合
        self.relu1 = nn.ReLU()                                  #ReLU函数

        #a single neroun output layer
        self.f2 = nn.Linear(hidden_size, output_size)


    def forward(self, input_pep, input_HLA):
        '''
            input: [seq_len, 1, embedding dim]  (batch_size=1)
            output: [1, 1]
        '''

        #lstm output
        output_pep, (ht1, ct1) = self.lstm_pep(input_pep)       #LSTM的输出是一个tuple，output_pep是三维Tensor
        output_HLA, (ht2, ct2) = self.lstm_HLA(input_HLA)
        
        #concatenate ht as the input of the next layer
        seq_len, batch_size, hidden_size = ht1.shape
        output_pep = ht1.view(-1, hidden_size)                  #改为2维，列数为16，用最后一个状态（t）的隐含层的状态值作为下一层的输入
        seq_len, batch_size, hidden_size = ht2.shape
        output_HLA = ht2.view(-1, hidden_size)                  #用output（lstm输出的第一个量）的话，seq_len 不一致，无法cat，但之后可以尝试填充方法

        output = torch.cat((output_pep, output_HLA), dim = 1)   #列拼接，不改变行数，变为32列

        #feed forward
        output = self.f1(output)
        output = self.relu1(output)
        output = self.f2(output)

        return output


'''model 3 -- NetMHCpan1.0_BiLSTM'''
class NetMHCpan_10_BiLSTM(nn.Module):                 

    def __init__(self, input_size_pep, input_size_HLA, lstm_hidden_size, hidden_size, output_size):
        
        super(NetMHCpan_10_BiLSTM, self).__init__()         

        #2 LSTM layers
        self.lstm_pep = nn.LSTM(input_size = input_size_pep, hidden_size = lstm_hidden_size, num_layers = 1, bidirectional = True)    #1层
        self.lstm_HLA = nn.LSTM(input_size = input_size_HLA, hidden_size = lstm_hidden_size, num_layers = 1, bidirectional = True)

        #hidden layer
        self.f1 = nn.Linear(lstm_hidden_size*2*2, hidden_size)  #全连接(dense)，输入是LSTM输出的结合，size相比model2，多了1倍，因为LSTM输出的是双向的
        self.relu1 = nn.ReLU()                                  #ReLU函数

        #a single neroun output layer
        self.f2 = nn.Linear(hidden_size, output_size)


    def forward(self, input_pep, input_HLA):
        '''
            input: [seq_len, 1, embedding dim]  (batch_size=1)
            output: [1, 1]
        '''
        
        #lstm output
        output_pep, (ht1, ct1) = self.lstm_pep(input_pep)       #LSTM的输出是一个tuple，output_pep是三维Tensor，ht也是
        output_HLA, (ht2, ct2) = self.lstm_HLA(input_HLA)       #BiLSTM出来的结果是“一对”ht
        
        #concatenate ht as the input of the next layer
        seq_len, batch_size, hidden_size = ht1.shape
        output_pep = ht1.view(-1, hidden_size*2)                #改为2维，列数为16*2，用最后一个状态（t）的隐含层的状态值作为下一层的输入
        seq_len, batch_size, hidden_size = ht2.shape
        output_HLA = ht2.view(-1, hidden_size*2)                #用output（lstm输出的第一个量）的话，seq_len 不一致，无法cat，但之后可以尝试填充方法

        output = torch.cat((output_pep, output_HLA), dim = 1)   #列拼接，不改变行数，变为32列

        #feed forward
        output = self.f1(output)
        output = self.relu1(output)
        output = self.f2(output)

        return output


'''model 4 -- NetMHCpan1.0_AttBiLSTM'''
class NetMHCpan_10_AttBiLSTM(nn.Module):                  

    def __init__(self, input_size_pep, input_size_HLA, batch_size, lstm_hidden_size, hidden_size, output_size):
        
        super(NetMHCpan_10_AttBiLSTM, self).__init__()          
        
        #2 LSTM layers
        self.bilstm_pep = nn.LSTM(input_size = input_size_pep, hidden_size = lstm_hidden_size, num_layers = 1, bidirectional = True)    #1层
        self.bilstm_HLA = nn.LSTM(input_size = input_size_HLA, hidden_size = lstm_hidden_size, num_layers = 1, bidirectional = True)
        self.lstm_hidden_size = lstm_hidden_size

        #attention layer's parameters
        self.att_wight = nn.Parameter(torch.randn((batch_size, 1, lstm_hidden_size)))

        #dense layer
        self.f1 = nn.Linear(lstm_hidden_size*2, hidden_size)    #全连接(dense)，输入是AttBiLSTM输出的结合
        self.relu1 = nn.ReLU()                                  #ReLU函数
        self.f2 = nn.Linear(hidden_size, output_size)           #a single neroun output layer

    def attention(self, H):

        M = torch.tanh(H)
        a = torch.bmm(self.att_wight, M)    #注意是M，不是H
        a = F.softmax(a, dim=2)             #在dim2上softmax
        a = a.transpose(1,2)
        r = torch.bmm(H, a)

        return r

    def forward(self, input_pep, input_HLA):
        '''
            input: [seq_len, 1, embedding dim]  (batch_size=1)
            output: [1, 1]
        '''
        
        #lstm output
        output_pep, (ht1, ct1) = self.bilstm_pep(input_pep)     #output[seq_len, batch_size, hidden_size * 2]
        output_HLA, (ht2, ct2) = self.bilstm_HLA(input_HLA)       
        
        output_pep = output_pep[:, :, :self.lstm_hidden_size] + output_pep[:, :, self.lstm_hidden_size:]
        output_HLA = output_HLA[:, :, :self.lstm_hidden_size] + output_HLA[:, :, self.lstm_hidden_size:]
        
        output_pep = output_pep.permute(1,2,0)                  #转为正确的attention输入格式
        output_HLA = output_HLA.permute(1,2,0)

        #attention output
        output_pep = self.attention(output_pep)                 #[b, lstm_h, 1]
        output_HLA = self.attention(output_HLA)

        output_pep = torch.tanh(output_pep)
        output_HLA = torch.tanh(output_HLA)

        output_pep = output_pep.view(-1, self.lstm_hidden_size)
        output_HLA = output_HLA.view(-1, self.lstm_hidden_size) #转为正确的dense层输入格式

        #concatenate output above as the input of the next layer
        output = torch.cat((output_pep, output_HLA), dim = 1)   #列拼接，不改变行数，变为32列

        #feed forward
        output = self.f1(output)
        output = self.relu1(output)
        output = self.f2(output)

        return output
