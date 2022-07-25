'''完整预测模型

    input: 含氨基酸序列的文件
    output: 1. 数据集中的每个数据是否预测正确
            2. accuracy

    包含以下模型:
                NetMHCpan_10
                NetMHCpan_10_LSTM
                NetMHCpan_10_BiLSTM
                NetMHCpan_10_AttBiLSTM

                TransPHLA from https://github.com/a96123155/TransPHLA-AOMP

'''

import pandas as pd
import numpy as np
import torch
import task_models
import model as Tmodel          #TransPHLA model
import torch.utils.data as Data

'''BLOSUM50'''
BLOSUM50_Matrix = pd.read_table('BLOSUM50_Matrix.txt',sep = '\s+')          #用\s+,如果用\t会让txt中第一列会被当作数据
amino_acid_sort = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V','B','J','Z','X']    #X可以用于padding
amino_acid_num = len(amino_acid_sort)
BLOSUM50_Matrix = BLOSUM50_Matrix.loc[amino_acid_sort, amino_acid_sort]     #去掉BLOSUM50中的*

#参数
pep_len_max = 14
HLA_len = 34

#设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def use_NetMHCpan_10(file_name):
    
    #设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input_size = (pep_len_max + HLA_len) * amino_acid_num   #pHLA编码长度
    hidden_size = 60                                        #22-86 from paper
    output_size = 1                                         #a single neroun

    '''提取数据'''
    pep_data = pd.read_csv(file_name, usecols=["peptide","length"]) 
    HLA_data = pd.read_csv(file_name, usecols=["HLA_sequence"]) 
    bind_data = pd.read_csv(file_name, usecols=["label"]) 

    pep_seq, pep_len= np.array(pep_data["peptide"]), np.array(pep_data["length"])
    HLA_seq = np.array(HLA_data["HLA_sequence"])
    bind_label = np.array(bind_data["label"])

    data_num = bind_label.size          #数据数量

    '''加载模型'''
    use_model = torch.load("model/NetMHCpan_10_fold2.pkl")

    '''预测'''
    pre_result = np.zeros(data_num)

    for i in range(data_num):

        '''编码'''
        pHLA_code = np.zeros((HLA_len+pep_len_max, amino_acid_num))

        for j in range(pep_len[i]):
            pHLA_code[j,:] = BLOSUM50_Matrix[pep_seq[i][j]]             #先对peptide编码

        for j in range(pep_len[i], pep_len_max):
            pHLA_code[j,:] = BLOSUM50_Matrix['X']                       #peptide长度不足14，padding，用X编码 

        for j in range(HLA_len):
            pHLA_code[j+pep_len_max,:] = BLOSUM50_Matrix[HLA_seq[i][j]] #最后对HLA序列编码，放在后面

        pHLA_code = pHLA_code.reshape(-1, input_size)                   #转为1*1152数组

        '''前向传播，得到预测'''
        pHLA_code = torch.Tensor(pHLA_code).to(device)                  #输入GPU
        target = bind_label[i]
        target = target.reshape(-1,1)
        target = torch.Tensor(target).to(device, dtype = torch.float)   #MSE backward()中要求是float，不是long

        prediction = use_model(pHLA_code)
        pre_result[i] = prediction.item()

    '''probability--->binding(1) or non-binding(0)'''
    pre_result[pre_result >= 0.5] = 1
    pre_result[pre_result < 0.5] = 0

    '''prediction与label比较'''
    pre_result[pre_result != bind_label] = 0    #用pre_result保存比较结果
    pre_result[pre_result == bind_label] = 1

    correct_num = np.sum(pre_result)
    accuracy = correct_num/data_num

    return pre_result, accuracy

def use_LSTM(file_name):

    #设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    embedding_dim = amino_acid_num
    lstm_hidden_size = 16                                   #16 from paper
    hidden_size = 60                                        #60 from paper
    output_size = 1                                         #a single neroun

    '''提取数据'''
    pep_data = pd.read_csv(file_name, usecols=["peptide","length"]) 
    HLA_data = pd.read_csv(file_name, usecols=["HLA_sequence"]) 
    bind_data = pd.read_csv(file_name, usecols=["label"]) 

    pep_seq, pep_len = np.array(pep_data["peptide"]), np.array(pep_data["length"])
    HLA_seq = np.array(HLA_data["HLA_sequence"])
    bind_label = np.array(bind_data["label"])

    data_num = bind_label.size          #数据数量

    '''加载模型'''
    use_model = torch.load("model/NetMHCpan_10_LSTM2_fold2.pkl")   #使用非填充编码训练的模型

    '''预测'''
    pre_result = np.zeros(data_num)

    for i in range(data_num):

        '''先编码'''
        pep_code = np.zeros((pep_len_max, amino_acid_num))
        HLA_code = np.zeros((HLA_len, amino_acid_num))

        for j in range(pep_len[i]):
            pep_code[j,:] = BLOSUM50_Matrix[pep_seq[i][j]]              #先对peptide编码

        for j in range(HLA_len):
            HLA_code[j,:] = BLOSUM50_Matrix[HLA_seq[i][j]]              #最后对HLA序列编码，放在后面
                
        pep_code = pep_code.reshape(-1, 1, embedding_dim)               #lstm input: shape = [seq_length, batch_size, input_size]的张量
        HLA_code = HLA_code.reshape(-1, 1, embedding_dim)               #batch_size = 1           

        '''前向传播，得到预测'''
        pep_code = torch.Tensor(pep_code).to(device)                    #输入GPU
        HLA_code = torch.Tensor(HLA_code).to(device)

        target = bind_label[i]
        target = target.reshape(-1,1)
        target = torch.Tensor(target).to(device, dtype = torch.float)   #MSE backward()中要求是float，不是long

        prediction = use_model(pep_code, HLA_code)
        pre_result[i] = prediction.item()

    '''probability--->binding(1) or non-binding(0)'''
    pre_result[pre_result >= 0.5] = 1
    pre_result[pre_result < 0.5] = 0

    '''prediction与label比较'''
    pre_result[pre_result != bind_label] = 0    #用pre_result保存比较结果
    pre_result[pre_result == bind_label] = 1

    correct_num = np.sum(pre_result)
    accuracy = correct_num/data_num

    return pre_result, accuracy

def use_BiLSTM(file_name):
    
    embedding_dim = amino_acid_num
    lstm_hidden_size = 16                                   #16 from paper
    hidden_size = 60                                        #60 from paper
    output_size = 1                                         #a single neroun

    '''提取数据'''
    pep_data = pd.read_csv(file_name, usecols=["peptide","length"]) 
    HLA_data = pd.read_csv(file_name, usecols=["HLA_sequence"]) 
    bind_data = pd.read_csv(file_name, usecols=["label"]) 

    pep_seq, pep_len = np.array(pep_data["peptide"]), np.array(pep_data["length"])
    HLA_seq = np.array(HLA_data["HLA_sequence"])
    bind_label = np.array(bind_data["label"])

    data_num = bind_label.size          #数据数量

    '''加载模型'''
    use_model = torch.load("model/NetMHCpan_10_BiLSTM_fold2.pkl")

    '''预测'''
    pre_result = np.zeros(data_num)

    for i in range(data_num):

        '''先编码'''
        pep_code = np.zeros((pep_len_max, amino_acid_num))
        HLA_code = np.zeros((HLA_len, amino_acid_num))

        for j in range(pep_len[i]):
            pep_code[j,:] = BLOSUM50_Matrix[pep_seq[i][j]]              #先对peptide编码

        for j in range(pep_len[i], pep_len_max):
            pep_code[j,:] = BLOSUM50_Matrix['X']                        #peptide长度不足14，padding，用X编码 

        for j in range(HLA_len):
            HLA_code[j,:] = BLOSUM50_Matrix[HLA_seq[i][j]]              #最后对HLA序列编码，放在后面
                
        pep_code = pep_code.reshape(-1, 1, embedding_dim)               #lstm input: shape = [seq_length, batch_size, input_size]的张量
        HLA_code = HLA_code.reshape(-1, 1, embedding_dim)               #batch_size = 1           

        '''前向传播，得到预测'''
        pep_code = torch.Tensor(pep_code).to(device)                    #输入GPU
        HLA_code = torch.Tensor(HLA_code).to(device)

        target = bind_label[i]
        target = target.reshape(-1,1)
        target = torch.Tensor(target).to(device, dtype = torch.float)   #MSE backward()中要求是float，不是long

        prediction = use_model(pep_code, HLA_code)
        pre_result[i] = prediction.item()

    '''probability--->binding(1) or non-binding(0)'''
    pre_result[pre_result >= 0.5] = 1
    pre_result[pre_result < 0.5] = 0

    '''prediction与label比较'''
    pre_result[pre_result != bind_label] = 0    #用pre_result保存比较结果
    pre_result[pre_result == bind_label] = 1

    correct_num = np.sum(pre_result)
    accuracy = correct_num/data_num

    return pre_result, accuracy

def use_AttBiLSTM(file_name):
    
    embedding_dim = amino_acid_num
    lstm_hidden_size = 16                                   #16 from paper
    hidden_size = 60                                        #60 from paper
    output_size = 1                                         #a single neroun

    '''提取数据'''
    pep_data = pd.read_csv(file_name, usecols=["peptide","length"]) 
    HLA_data = pd.read_csv(file_name, usecols=["HLA_sequence"]) 
    bind_data = pd.read_csv(file_name, usecols=["label"]) 

    pep_seq, pep_len, pep_len_max= np.array(pep_data["peptide"]), np.array(pep_data["length"]), 14
    HLA_seq, HLA_len = np.array(HLA_data["HLA_sequence"]), 34
    bind_label = np.array(bind_data["label"])

    data_num = bind_label.size          #数据数量

    '''加载模型'''
    use_model = torch.load("model/NetMHCpan_10_AttBiLSTM_fold2.pkl")

    '''预测'''
    pre_result = np.zeros(data_num)

    for i in range(data_num):

        '''先编码'''
        pep_code = np.zeros((pep_len_max, amino_acid_num))
        HLA_code = np.zeros((HLA_len, amino_acid_num))

        for j in range(pep_len[i]):
            pep_code[j,:] = BLOSUM50_Matrix[pep_seq[i][j]]              #先对peptide编码 

        for j in range(HLA_len):
            HLA_code[j,:] = BLOSUM50_Matrix[HLA_seq[i][j]]              #最后对HLA序列编码，放在后面
                
        pep_code = pep_code.reshape(-1, 1, embedding_dim)               #lstm input: shape = [seq_length, batch_size, input_size]的张量
        HLA_code = HLA_code.reshape(-1, 1, embedding_dim)               #batch_size = 1           

        '''前向传播，得到预测'''
        pep_code = torch.Tensor(pep_code).to(device)                    #输入GPU
        HLA_code = torch.Tensor(HLA_code).to(device)

        target = bind_label[i]
        target = target.reshape(-1,1)
        target = torch.Tensor(target).to(device, dtype = torch.float)   #MSE backward()中要求是float，不是long

        prediction = use_model(pep_code, HLA_code)
        pre_result[i] = prediction.item()

    '''probability--->binding(1) or non-binding(0)'''
    pre_result[pre_result >= 0.5] = 1
    pre_result[pre_result < 0.5] = 0

    '''prediction与label比较'''
    pre_result[pre_result != bind_label] = 0    #用pre_result保存比较结果
    pre_result[pre_result == bind_label] = 1

    correct_num = np.sum(pre_result)
    accuracy = correct_num/data_num

    return pre_result, accuracy

def use_TransPHLA(file_name):
    #设备
    use_cuda = False        #cpu
    device = torch.device("cuda" if use_cuda else "cpu")
    
    '''提取数据并处理'''
    data = pd.read_csv(file_name) 
    
    bind_labels = np.array(data['label'])
    data_num = len(bind_labels)

    #变成可以用TransPHLA处理的数据
    pep_inputs, HLA_inputs = Tmodel.make_data(data)
    val_loader = Data.DataLoader(Tmodel.MyDataSet(pep_inputs, HLA_inputs), batch_size = 1, shuffle = False, num_workers = 0)    
    #受C/GPU容量限制，设batch_size = 1，方便处理大量数据

    '''加载模型'''
    model_file = 'model/model_layer1_multihead9_fold4.pkl'

    model_eval = Tmodel.Transformer().to(device)
    model_eval.load_state_dict(torch.load(model_file), strict = True)

    '''预测'''
    model_eval.eval()
    y_pred, y_prob, attns = Tmodel.eval_step(model_eval, val_loader, 0.5, use_cuda)     #output: 预测值(0/1)，概率，注意力

    y_pred[y_pred != bind_labels] = 0
    y_pred[y_pred == bind_labels] = 1

    correct_num = np.sum(y_pred)
    accuracy = correct_num/data_num

    return y_pred, accuracy
