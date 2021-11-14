import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import give_valid_test
import _pickle as cpickle

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

def make_batch(train_path, word2number_dict, batch_size, n_step):
    all_input_batch = []
    all_target_batch = []

    text = open(train_path, 'r', encoding='utf-8') #open the file

    input_batch = []
    target_batch = []
    for sen in text:
        word = sen.strip().split(" ")  # space tokenizer
        word = ["<sos>"] + word
        word = word + ["<eos>"]

        if len(word) <= n_step:   #pad the sentence
            word = ["<pad>"]*(n_step+1-len(word)) + word

        for word_index in range(len(word)-n_step):
            input = [word2number_dict[n] for n in word[word_index:word_index+n_step]]  # create (1~n-1) as input
            target = word2number_dict[word[word_index+n_step]]  # create (n) as target, We usually call this 'casual language model'
            input_batch.append(input)
            target_batch.append(target)

            if len(input_batch) == batch_size:
                all_input_batch.append(input_batch)
                all_target_batch.append(target_batch)
                input_batch = []
                target_batch = []

    return all_input_batch, all_target_batch # (batch num, batch size, n_step) (batch num, batch size)

def make_dict(train_path):
    text = open(train_path, 'r', encoding='utf-8')  #open the train file
    word_list = set()  # a set for making dict

    for line in text:
        line = line.strip().split(" ")
        word_list = word_list.union(set(line))

    word_list = list(sorted(word_list))   #set to list

    word2number_dict = {w: i+2 for i, w in enumerate(word_list)}
    number2word_dict = {i+2: w for i, w in enumerate(word_list)}

    #add the <pad> and <unk_word>
    word2number_dict["<pad>"] = 0
    number2word_dict[0] = "<pad>"
    word2number_dict["<unk_word>"] = 1
    number2word_dict[1] = "<unk_word>"
    word2number_dict["<sos>"] = 2
    number2word_dict[2] = "<sos>"
    word2number_dict["<eos>"] = 3
    number2word_dict[3] = "<eos>"

    return word2number_dict, number2word_dict




class TextLSTM(nn.Module):#**********************************
    def __init__(self, input_size: int, hidden_size: int):
        super(TextLSTM, self).__init__()
        self.embedding = nn.Embedding(n_class, embedding_dim=emb_size)
        # self.LSTM = nn.LSTM(input_size=emb_size, hidden_size=n_hidden)#定义参数
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.input_size1 = input_size
        self.hidden_size1 = hidden_size
        #第一层LSTM
        # f_t
        self.U_f = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.V_f = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_f = nn.Parameter(torch.Tensor(hidden_size))

        # i_t
        self.U_i = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.V_i = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_i = nn.Parameter(torch.Tensor(hidden_size))

        # g_t
        self.U_g = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.V_g = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_g = nn.Parameter(torch.Tensor(hidden_size))

        # c_t
        self.U_c = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.V_c = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_c = nn.Parameter(torch.Tensor(hidden_size))

        # o_t
        self.U_o = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.V_o = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_o = nn.Parameter(torch.Tensor(hidden_size))


        #第二层LSTM
        # f1_t
        self.U_f1 = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.V_f1= nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_f1 = nn.Parameter(torch.Tensor(hidden_size))

        # i1_t
        self.U_i1 = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.V_i1 = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_i1 = nn.Parameter(torch.Tensor(hidden_size))

        # g1_t
        self.U_g1 = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.V_g1 = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_g1 = nn.Parameter(torch.Tensor(hidden_size))

        # c1_t
        self.U_c1 = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.V_c1 = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_c1 = nn.Parameter(torch.Tensor(hidden_size))

        # o1_t
        self.U_o1 = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.V_o1 = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_o1 = nn.Parameter(torch.Tensor(hidden_size))

        self.U = nn.Linear(hidden_size, n_class, bias=False)
        self.V = nn.Linear(hidden_size, n_class, bias=False)
        self.b = nn.Parameter(torch.ones([n_class]))
        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, X):

        outputs=[]
        X = self.embedding(X)
        batch_size, n_step, embed = X.size()
        #第一层LSTM
        #短期记忆状态
        hidden_state = torch.ones(batch_size, self.hidden_size)  # [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
        #长期记忆状态
        cell_state = torch.ones(batch_size, self.hidden_size)     # [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
        #第二层LSTM
        #短期记忆状态
        hidden_state1 = torch.ones(batch_size, self.hidden_size)  # [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
        #长期记忆状态
        cell_state1 = torch.ones(batch_size, self.hidden_size)     # [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
        # X : [batch_size,n_step,embeding size]

        for t in range(n_step):
            ###########第一层
            x_t = X[:, t, :]
            #遗忘门
            f_t = torch.sigmoid(x_t @ self.U_f + hidden_state @ self.V_f+self.b_f)
            #输入门
            i_t = torch.sigmoid(x_t @ self.U_i + hidden_state @ self.V_i + self.b_i)
            g_t = torch.tanh(x_t @ self.U_g +hidden_state @ self.V_g + self.b_g)
            #记忆状态更新
            cell_state = f_t * cell_state + i_t * g_t
            #输出门
            o_t = torch.sigmoid(x_t @ self.U_o + hidden_state @ self.V_o1 + self.b_o1)
            hidden_state = o_t * torch.tanh(cell_state)
            ############第二层
            f1_t = torch.sigmoid(hidden_state @ self.U_f1 + hidden_state1 @ self.V_f1 + self.b_f1)
            # 输入门
            i1_t = torch.sigmoid(hidden_state @ self.U_i1 + hidden_state1 @ self.V_i1 + self.b_i1)
            g1_t = torch.tanh(hidden_state @ self.U_g1 + hidden_state1 @ self.V_g1 + self.b_g1)
            # 记忆状态更新
            cell_state1 = f1_t * cell_state1 + i1_t * g1_t
            # 输出门
            o1_t = torch.sigmoid(hidden_state @ self.U_o1 + hidden_state1 @ self.V_o1 + self.b_o1)
            hidden_state1 = o1_t * torch.tanh(cell_state1)
            # save the state after each cycle
            outputs.append(hidden_state1.unsqueeze(0))

        # outputs(after the cat) [n_step , batch_size , n_hidden]
        outputs = torch.cat(outputs, dim=0)
        outputs = outputs[-1] # [batch_size, num_directions(=1) * n_hidden]
        model = self.U(outputs) +self.U(outputs)+ self.b # model : [batch_size, n_class]
        return model

def train_LSTMlm():
    model = TextLSTM(input_size=emb_size,hidden_size=n_hidden)#采用LSTM模型进行训练从这里进行修改
    model.to(device)
    print(model)

    criterion = nn.CrossEntropyLoss()#使用交叉熵损失函数
    optimizer = optim.Adam(model.parameters(), lr=learn_rate)#自适应函数下降法
    
    # Training
    batch_number = len(all_input_batch)
    for epoch in range(all_epoch):
        count_batch = 0
        for input_batch, target_batch in zip(all_input_batch, all_target_batch):
            optimizer.zero_grad()

            # input_batch : [batch_size, n_step, n_class]
            output = model(input_batch)

            # output : [batch_size, n_class], target_batch : [batch_size] (LongTensor, not one-hot)
            loss = criterion(output, target_batch)
            ppl = math.exp(loss.item())
            if (count_batch + 1) % 100 == 0:
                print('Epoch:', '%04d' % (epoch + 1), 'Batch:', '%02d' % (count_batch + 1), f'/{batch_number}',
                      'loss =', '{:.6f}'.format(loss), 'ppl =', '{:.6}'.format(ppl))

            loss.backward()
            optimizer.step()

            count_batch += 1
        print('Epoch:', '%04d' % (epoch + 1), 'Batch:', '%02d' % (count_batch + 1), f'/{batch_number}',
                'loss =', '{:.6f}'.format(loss), 'ppl =', '{:.6}'.format(ppl))

        # valid after training one epoch
        all_valid_batch, all_valid_target = give_valid_test.give_valid(data_root, word2number_dict, n_step)
        all_valid_batch = torch.LongTensor(all_valid_batch).to(device)  # list to tensor
        all_valid_target = torch.LongTensor(all_valid_target).to(device)

        total_valid = len(all_valid_target)*128  # valid and test batch size is 128
        with torch.no_grad():
            total_loss = 0
            count_loss = 0
            for valid_batch, valid_target in zip(all_valid_batch, all_valid_target):
                valid_output = model(valid_batch)
                valid_loss = criterion(valid_output, valid_target)
                total_loss += valid_loss.item()
                count_loss += 1
          
            print(f'Valid {total_valid} samples after epoch:', '%04d' % (epoch + 1), 'loss =',
                  '{:.6f}'.format(total_loss / count_loss),
                  'ppl =', '{:.6}'.format(math.exp(total_loss / count_loss)))

        if (epoch+1) % save_checkpoint_epoch == 0:
            torch.save(model, f'models/LSTMlm_model_epoch{epoch+1}.ckpt')

def test_LSTMlm(select_model_path):
    model = torch.load(select_model_path, map_location="cpu")  #load the selected model
    model.to(device)

    #load the test data
    all_test_batch, all_test_target = give_valid_test.give_test(data_root, word2number_dict, n_step)
    all_test_batch = torch.LongTensor(all_test_batch).to(device)  # list to tensor
    all_test_target = torch.LongTensor(all_test_target).to(device)
    total_test = len(all_test_target)*128  # valid and test batch size is 128
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    count_loss = 0
    for test_batch, test_target in zip(all_test_batch, all_test_target):
        test_output = model(test_batch)
        test_loss = criterion(test_output, test_target)
        total_loss += test_loss.item()
        count_loss += 1

    print(f"Test {total_test} samples with {select_model_path}……………………")
    print('loss =','{:.6f}'.format(total_loss / count_loss),
                  'ppl =', '{:.6}'.format(math.exp(total_loss / count_loss)))

if __name__ == '__main__':
    n_step = 5 # number of cells(= number of Step)
    n_hidden = 128 # number of hidden units in one cell
    batch_size = 128 # batch size
    learn_rate = 0.0005
    all_epoch = 5 #the all epoch for training
    emb_size = 256 #embeding size
    save_checkpoint_epoch = 5 # save a checkpoint per save_checkpoint_epoch epochs !!! Note the save path !!!
    data_root = 'penn_small'
    train_path = os.path.join(data_root, 'train.txt') # the path of train dataset

    print("print parameter ......")
    print("n_step:", n_step)
    print("n_hidden:", n_hidden)
    print("batch_size:", batch_size)
    print("learn_rate:", learn_rate)
    print("all_epoch:", all_epoch)
    print("emb_size:", emb_size)
    print("save_checkpoint_epoch:", save_checkpoint_epoch)
    print("train_data:", data_root)

    word2number_dict, number2word_dict = make_dict(train_path)#生成一一对应词表
    #print(word2number_dict)

    print("The size of the dictionary is:", len(word2number_dict))
    n_class = len(word2number_dict)  #n_class (= dict size)
    #print(n_class)

    print("generating train_batch ......")
    all_input_batch, all_target_batch = make_batch(train_path, word2number_dict, batch_size, n_step)  # make the batch
    #print(all_input_batch)
    #print(all_target_batch)
    train_batch_list = [all_input_batch, all_target_batch]
    #print(train_batch_list)

    print("The number of the train batch is:", len(all_input_batch))
    all_input_batch = torch.LongTensor(all_input_batch).to(device)   #list to tensor
    all_target_batch = torch.LongTensor(all_target_batch).to(device)
    # print(all_input_batch.shape)
    # print(all_target_batch.shape)
    all_input_batch = all_input_batch.reshape(-1, batch_size, n_step)
    all_target_batch = all_target_batch.reshape(-1, batch_size)
    # print(all_input_batch.shape)
    # print(all_target_batch.shape)
    print("\nTrain the LSTMLM……………………")
    train_LSTMlm()

    print("\nTest the LSTMLM……………………")
    select_model_path = "models/LSTMlm_model_epoch5.ckpt"
    test_LSTMlm(select_model_path)
