# model class 
import time
from importlib import reload

import torch 
from torch import nn 
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from tqdm.auto import tqdm 
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix, roc_auc_score

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()

def log_sum_exp(vec, is_matrix=True):
    ncl = vec.shape[1]
    
    if is_matrix:
        max_scores = torch.max(vec, dim=1).values
        return max_scores + torch.log(torch.sum(torch.exp(vec - max_scores.expand(ncl,ncl).T), dim=1))
    else:
        max_scores = torch.max(vec)
        return max_scores + torch.log(torch.sum(torch.exp(vec - max_scores.expand(1, ncl))))


class BiLSTM_CRF(nn.Module):
    def __init__(self, embedding_dim, hidden_size, nclasses, label2num, device='cpu') -> None:
        super().__init__()

        # self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim).to(device)
        self.lstm_model = nn.LSTM(embedding_dim, hidden_size//2, bidirectional=True).to(device)
        self.ffwd_lay = nn.Linear(hidden_size, nclasses+2).to(device)
        self.softmax = nn.Softmax(dim=1).to(device)

        self.optim = torch.optim.Adam(self.parameters(), lr=1e-2)
        
        self.loss_history = []
        self.hidden_size = hidden_size

        self.tag_to_ix = label2num.copy()

        self.START_TAG = '<START>'
        self.STOP_TAG = '<STOP>'
        self.tag_to_ix[self.START_TAG] = max(label2num.values()) + 1
        self.tag_to_ix[self.STOP_TAG] = max(label2num.values()) + 2

        self.tagset_size = nclasses+2

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))
        
        self.transitions.data[self.tag_to_ix[self.START_TAG], :] = -10000
        self.transitions.data[:, self.tag_to_ix[self.STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_size // 2),
                torch.randn(2, 1, self.hidden_size // 2))
    

    def _get_lstm_features(self, batch):
        out = self.lstm_model(batch)[0] # L x hidden_size
        out = self.ffwd_lay(out) # L x nclasses
        return out


    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[self.START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            # next_tag_var = self.transitions + forward_var
            # best_tag_id = torch.argmax(next_tag_var, dim=1)
            # viterbivars_t = torch.max(next_tag_var, dim=1).values  

            next_tag_var = self.transitions + forward_var
            bptrs_t = torch.argmax(next_tag_var, dim=1)
            viterbivars_t = torch.max(next_tag_var, dim=1).values

            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (viterbivars_t + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[self.STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[self.START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, torch.LongTensor(best_path)
    

    def neg_log_likelihood(self, sentence, tags, device):
        feats = self._get_lstm_features(sentence.to(device))
        forward_score = self._forward_alg(feats, device)
        gold_score = self._score_sentence(feats, tags, device)

        return forward_score - gold_score
    

    def _score_sentence(self, feats, tags, device):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1, device=device)
        tags = torch.cat([torch.tensor([self.tag_to_ix[self.START_TAG]], dtype=torch.long), tags]).to(device)
        
        # for i, feat in enumerate(feats):
            # score += self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        # score += self.transitions[self.tag_to_ix[self.STOP_TAG], tags[-1]]

        score = torch.sum(self.transitions[tags[1:], tags[:-1]] + feats[range(len(feats)), tags[1:]]) + self.transitions[self.tag_to_ix[self.STOP_TAG], tags[-1]]
        return score


    def _forward_alg(self, feats, device):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.).to(device)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[self.START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        ncl = self.transitions.shape[0]
        for feat in feats:
            forward_var = log_sum_exp(forward_var + self.transitions + feat.expand((ncl,ncl)).T).view(1, -1)

        terminal_var = forward_var + self.transitions[self.tag_to_ix[self.STOP_TAG]]
        alpha = log_sum_exp(terminal_var, is_matrix=False)
        return alpha
    

    def forward(self, batch, device):   
        lstm_feats = self._get_lstm_features(batch.to(device)).cpu()
        score, tag_seq = self._viterbi_decode(lstm_feats)

        return score, tag_seq
    

    def fit(self, train_X, train_Y, valid_X, valid_Y, nepochs, lr, device):
        self.train()
        self.to(device)

        for g in self.optim.param_groups:
            g['lr'] = lr
        
        for ep in tqdm(range(nepochs)):
            eploss = 0
        
            if ep % 20 == 0:
                for g in self.optim.param_groups:
                    g['lr'] = lr/2  

            for i, (batch_X, batch_Y) in tqdm(enumerate(zip(train_X, train_Y))):
                self.zero_grad()

                loss = self.neg_log_likelihood(batch_X, batch_Y, device)
            
                loss.backward()
                self.optim.step()

                eploss += loss.item()
                self.loss_history.append(loss.item())

            printbool = ep % (nepochs//10) == 0 if nepochs > 10 else True
            if printbool:
                with torch.no_grad():
                    train_predict = []
                    for batch_X in train_X:
                        score, predict = self.forward(batch_X.to(device), device)
                        train_predict += predict
                    
                    test_predict = torch.FloatTensor(test_predict)
                    train_real = torch.cat(train_Y)


                    with torch.no_grad():
                        test_predict = []
                        for batch_X in valid_X:
                            score, predict = self.forward(batch_X.to(device), device)
                            test_predict += predict
                        
                        test_predict = torch.FloatTensor(test_predict)
                        test_real = torch.cat(valid_Y)

                    # train_predict, train_real = train_predict[train_predict != 0], train_real[train_predict != 0]
                    # test_predict, test_real = test_predict[test_predict != 0], test_real[test_predict != 0]
                    TP = ((train_predict == train_real) & (train_predict != 0)).sum()
                    FP = ((train_predict != train_real) & (train_predict != 0)).sum()
                    FN = ((train_predict != train_real) & (train_predict == 0)).sum()
                    p_metric_train = TP / (TP + FP)
                    r_metric_train = TP / (TP + FN)

                    TP = ((test_predict == test_real) & (test_predict != 0)).sum()
                    FP = ((test_predict != test_real) & (test_predict != 0)).sum()
                    FN = ((test_predict != test_real) & (test_predict == 0)).sum()
                    p_metric_valid = TP / (TP + FP)
                    r_metric_valid = TP / (TP + FN)
                    
                    print(f'Iter: {ep}, Loss: {eploss/len(train_X):.3f} Train precision {p_metric_train:.3f}, Train recall: {r_metric_train:.3f}, Valid precision: {p_metric_valid:.3f}, Valid recall: {r_metric_valid:.3f}')

                    # print(f'Iter: {ep}, Loss: {eploss/len(train_X):.3f} Train BA: {balanced_accuracy_score(train_real, train_predict):.3f}, Train F1: {f1_score(train_real, train_predict, average="micro"):.3f}, Valid BA: {balanced_accuracy_score(test_real, test_predict):.3f}, Valid F1: {f1_score(test_real, test_predict, average="micro"):.3f}')
                        
                # printbool = ep % (nepochs//10) == 0 if nepochs > 10 else True
                # if printbool:
                #     print(f'Train loss: {eploss/len(train_X):.3f}')