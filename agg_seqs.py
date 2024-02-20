import torch 
from torch import nn 
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from tqdm.auto import tqdm 

class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, nclasses, device) -> None:
        super().__init__()

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim).to(device)
        self.lstm_model = nn.LSTM(embedding_dim, hidden_size//2, bidirectional=True).to(device)
        self.ffwd_lay = nn.Linear(hidden_size, nclasses).to(device)
        self.softmax = nn.Softmax(dim=1).to(device)

        self.optim = torch.optim.Adam(self.parameters(), lr=1e-2)
        self.criterion = nn.CrossEntropyLoss()


    def forward(self, batch_X, batch_Y, seq_lens, device):
        out = self.embedding(batch_X.to(device)) # L x vocab_size -> L x embedding_dim
        out = pack_padded_sequence(out, seq_lens, batch_first=True, enforce_sorted=False)
        out, _ = self.lstm_model(out) # L x hidden_size
        out, seq_lens = pad_packed_sequence(out, batch_first=True)
        
        batch_predict = []
        batch_real = []
        for it_X, it_Y, seq_len in zip(out, batch_Y, seq_lens):
            batch_predict.append(self.softmax(self.ffwd_lay(it_X[:seq_len])))
            batch_real.append(it_Y[:seq_len])

        return torch.cat(batch_predict, dim=0), torch.cat(batch_real, dim=0).to(device)


    def fit(self, train_X, train_Y, seq_lens, nepochs, lr, device):
        self.train()
        self.to(device)

        for g in self.optim.param_groups:
            g['lr'] = lr    
        
        for ep in tqdm(range(nepochs)):
            eploss = 0
            
            for batch_X, batch_Y, batch_seq_len in tqdm(zip(train_X, train_Y, seq_lens)):
                predict, real = self.forward(batch_X, batch_Y, batch_seq_len, device)
                
                self.optim.zero_grad()
                loss = self.criterion(predict, real)
                loss.backward()
                self.optim.step()

                eploss += loss.item()
            
            printbool = ep % (nepochs//10) == 0 if nepochs > 10 else True
            if printbool:
                print(f'Train loss: {eploss/len(train_X):.3f}')


def data_label_split(data, label, train_size=0.8):
    randidx = np.arange(len(data))
    data_train, data_test = train_test_split(data, randidx, train_size)
    label_train, label_test = train_test_split(label, randidx, train_size)

    return data_train, data_test, label_train, label_test

def train_test_split(data, randidx, train_size):
    N = len(data)
    return [data[i] for i in randidx[:int(train_size*N)]], [data[i] for i in randidx[int(train_size*N):]]

def shuffle_data_label_lists(data, label):
    randidx = np.arange(len(data))
    np.random.shuffle(randidx)
    return [data[i] for i in randidx], [label[i] for i in randidx]

def batch_split(X, Y, seq_len, batch_size=1000):
    x_batched = []
    y_batched = []
    seq_len_batched = []

    n = len(X)
    pointer = 0
    while pointer + batch_size < n:
        x_batched.append(X[pointer:pointer+batch_size])
        y_batched.append(Y[pointer:pointer+batch_size])
        seq_len_batched.append(seq_len[pointer:pointer+batch_size])
        pointer += batch_size 
    
    x_batched.append(X[pointer:])
    y_batched.append(Y[pointer:])
    seq_len_batched.append(seq_len[pointer:])

    return x_batched, y_batched, seq_len_batched

# encoding tokens and labels
with open('data/train.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

unique_tokens, unique_labels = set(), set()
for doc_i, doc in enumerate(tqdm(data)):
    unique_tokens |= set(np.unique(doc['tokens']))
    unique_labels |= set(np.unique(doc['labels']))

token2num = dict(zip(unique_tokens, range(1, len(unique_tokens)+1)))
label2num = {
    'O': 0,
    'B-URL_PERSONAL': 0, 
    'I-URL_PERSONAL': 0, 
    'B-ID_NUM': 0, 
    'I-ID_NUM': 0, 
    'B-EMAIL': 0, 
    'I-EMAIL': 0,
    'B-NAME_STUDENT': 1, 
    'I-NAME_STUDENT': 1, 
    'B-PHONE_NUM': 0, 
    'I-PHONE_NUM': 0, 
    'B-USERNAME': 0,
    'I-USERNAME': 0, 
    'B-STREET_ADDRESS': 0, 
    'I-STREET_ADDRESS': 0, 
}
num2token = {}
for it in token2num:
    num2token[token2num[it]] = it

# load data and split by sentences
sentences = []
cur_sentence = []
sentences_labels = []
cur_sentences_labels = []

for doc_i, doc in enumerate(tqdm(data)):
    for token, label in zip(data[doc_i]['tokens'], data[doc_i]['labels']):
        cur_sentence.append(token2num[token])
        cur_sentences_labels.append(label2num[label])

        if (token == '.') | (token.endswith('\n')) | (token == '?') | (token == '!'):   
            # if sum(cur_sentences_labels) > 0:
            sentences.append(torch.LongTensor(cur_sentence))
            sentences_labels.append(torch.LongTensor(cur_sentences_labels))

            cur_sentences_labels = []
            cur_sentence = []
    
    if sum(cur_sentences_labels) > 0:
        sentences.append(cur_sentence)
        sentences_labels.append(cur_sentences_labels)

    cur_sentences_labels = []
    cur_sentence = []

# create train and test df 
name_sentences_labels = []
name_sentences = []

username_sentences_labels = []
username_sentences = []

o_sentences_labels = []
o_sentences = []

for i, it in enumerate(sentences):
    if 1 in sentences_labels[i]:
        name_sentences_labels.append(sentences_labels[i])
        name_sentences.append(sentences[i])
    # if 2 in sentences_labels[i]:
        # username_sentences_labels.append(sentences_labels[i])
        # username_sentences.append(sentences[i])
    else:
        o_sentences_labels.append(sentences_labels[i])
        o_sentences.append(sentences[i])


name_sentences_train, name_sentences_test, name_sentences_labels_train, name_sentences_labels_test = data_label_split(name_sentences, name_sentences_labels)
# user_sentences_train, user_sentences_test, user_sentences_labels_train, user_sentences_labels_test = data_label_split(username_sentences, username_sentences_labels)
o_sentences_train, o_sentences_test, o_sentences_labels_train, o_sentences_labels_test = data_label_split(o_sentences, o_sentences_labels)

sentences_train = o_sentences_train + name_sentences_train*280
sentences_labels_train = o_sentences_labels_train + name_sentences_labels_train*280

sentences_test = o_sentences_test + name_sentences_test*280
sentences_labels_test = o_sentences_labels_test + name_sentences_labels_test*280

sentences_train, sentences_labels_train = shuffle_data_label_lists(sentences_train, sentences_labels_train)
sentences_test, sentences_labels_test = shuffle_data_label_lists(sentences_test, sentences_labels_test)

# from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
# import torch.nn.functional as F

# seq_len_train = list(map(len, sentences_train))
# max_len = max(seq_len_train)
# sentences_train_ten = torch.cat([F.pad(sentences_train[i], (0, max_len-seq_len_train[i])).reshape(1,-1) for i in range(len(sentences_train))])
# sentences_labels_train_ten = torch.cat([F.pad(sentences_labels_train[i], (0, max_len-seq_len_train[i])).reshape(1,-1) for i in range(len(sentences_labels_train))])

# seq_len_test = list(map(len, sentences_test))
# max_len = max(seq_len_test)
# sentences_test_ten = torch.cat([F.pad(sentences_test[i], (0, max_len-seq_len_test[i])).reshape(1,-1) for i in range(len(sentences_test))])
# sentences_labels_test_ten = torch.cat([F.pad(sentences_labels_test[i], (0, max_len-seq_len_test[i])).reshape(1,-1) for i in range(len(sentences_labels_test))])

# sentences_train, sentences_labels_train, seq_len_train = batch_split(sentences_train_ten, sentences_labels_train_ten, seq_len_train)
# sentences_test, sentences_labels_test, seq_len_test = batch_split(sentences_test_ten, sentences_labels_test_ten, seq_len_test)