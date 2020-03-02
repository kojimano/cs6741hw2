import os
import copy
import numpy as np
from collections import Counter
from collections import defaultdict

import torch
import torchtext
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from IPython import embed

def generate_ngrams(text, n_gram=1):
    tokens = [token for token in text]
    ngrams = zip(*[tokens[i:] for i in range(n_gram)])
    result = [ngram for ngram in ngrams]
    return result

class TrigramLM(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self._vocab_size = vocab_size
        self._n = 3
        #self._alphas = np.array([0.33, 0.33, 1-0.66])
        self._alphas = np.array([0.1, 0.8, 0.1])
        self._grams = [Counter()] + [{} for _ in range(1, self._n)]

    def set_alpha(self, alpha_1: float, alpha_2: float, alpha_3: float):
        norm = alpha_1 + alpha_2 + alpha_3
        alpha_1 = alpha_1 / float(norm)
        alpha_2 = alpha_2 / float(norm)
        alpha_3 = alpha_3 / float(norm)

        self._alpha_1 = nn.Parameter(torch.tensor(alpha_1))
        self._alpha_2 = nn.Parameter(torch.tensor(alpha_2))
        self._alpha_3 = nn.Parameter(torch.tensor(alpha_3))

    def set_ngrams(self, train_iter: torchtext.data.BucketIterator):
        totalNumWords = 0
        for b in iter(train_iter):
            for i in range(b.text.shape["batch"]):
                seq = b.text[{"batch": i}].tolist()
                for n in range(1, self._n):
                    for w in range(len(seq) - n):
                        next_tokens = tuple([seq[w + j] for j in range(n)])
                        current_token = seq[w+n]
                        current_dict = self._grams[n]
                        if next_tokens not in current_dict:
                            current_dict[next_tokens] = defaultdict(int)
                        current_dict[next_tokens][current_token] += 1
                        current_dict[next_tokens][-1] += 1

            values = b.text.values.contiguous().view(-1).tolist()
            self._grams[0].update(values)
            totalNumWords += len(values)
        self._grams[0][-1] = totalNumWords

    def forward(self, x: torch.FloatTensor,  smooth: int = 1):
        output = torch.zeros(self._vocab_size)
        for i in range(self._vocab_size):
            preds = []
            for n in range(self._n):

                # Select dict
                if n == 0:
                    tup_dict = self._grams[n]
                    key = i
                else:
                    dict_key = x[len(x)-n:]
                    if dict_key not in self._grams[n]:
                        tup_dict = {}
                    else:
                        tup_dict = self._grams[n][dict_key]
                    key = i

                # Calculate probability
                if key not in tup_dict:
                    pred = 0
                else:
                    pred = tup_dict[key] / float(tup_dict[-1])
                preds.append(pred)
            preds = np.array(preds)
            output[i] = np.dot(preds, self._alphas)

        return output
    """
    def em_forward(self, x: torch.FloatTensor, smooth: int = 1):
        batch_probs_alpha1 = []
        batch_probs_alpha2 = []
        batch_probs_alpha3 = []

        for batch_idx in range(x.shape[0]):
            sent = x[batch_idx, :].cpu()
            if torch.cuda.is_available():
                sent = sent.cpu()
            sent = sent.numpy()
            uni_grams = generate_ngrams(sent)
            bi_grams = [None] + generate_ngrams(sent, 2)
            tri_grams = [None, None] + generate_ngrams(sent, 3)

            sent_probs_alpha_1 = []
            sent_probs_alpha_2 = []
            sent_probs_alpha_3 = []

            for i in range(len(uni_grams)):
                norm = 0

                uni_term = self._unigrams[uni_grams[i]]
                prob1 = self._alpha_1 * (uni_term + smooth) / (np.sum(list(self._unigrams.values())) + smooth)

                # Unigram term
                numerator = self._unigrams[uni_grams[i]] + smooth
                denominator = np.sum(list(self._unigrams.values())) + smooth
                prob += self._alpha_1 * numerator / denominator
                alpha_norm += self._alpha_1

                # Bigram term
                if bi_grams[i] is not None:
                    numerator = 0 if bi_grams[i] not in self._bigrams.keys() else self._bigrams[bi_grams[i]]
                    numerator += smooth
                    denominator = self._unigrams[uni_grams[i-1]] + smooth
                    prob2 += self._alpha_2 * numerator / denominator
                    alpha_norm += self._alpha_2

                # Trigram term
                if tri_grams[i] is not None:
                    numerator = 0 if tri_grams[i] not in self._trigrams.keys() else self._trigrams[tri_grams[i]]
                    numerator += smooth
                    denominator = 0 if bi_grams[i-2] not in self._bigrams.keys() else self._bigrams[bi_grams[i-2]]
                    denominator += smooth
                    prob3 += self._alpha_3 * numerator / denominator
                    alpha_norm += self._alpha_3

                sent_probs_alpha_1.append(prob1)
                sent_probs_alpha_2.append(prob2)
                sent_probs_alpha_3.append(prob3)

            batch_probs_alpha1.append(torch.stack(sent_probs_alpha_1, 0))
            batch_probs_alpha2.append(torch.stack(sent_probs_alpha_2, 0))
            batch_probs_alpha3.append(torch.stack(sent_probs_alpha_3, 0))

        return torch.stack(batch_probs_alpha1, 0), torch.stack(batch_probs_alpha2, 0), torch.stack(batch_probs_alpha3, 0)
    """

class NeuralNetLM(nn.Module):
    """
    Implementation of "A Neural Probablistic Language Model" (Bengio et al., 2003)
    """
    def __init__(self, vocab_size: int):
        super().__init__()

        self._vocab_size = vocab_size
        self._h_dim = 100 # hidden simension
        self._n = 5 # context window size
        self._m = 60 # embeddings dimension

        self.U = nn.Linear(self._h_dim, self._vocab_size, bias=False)
        self.W = nn.Linear(self._m * (self._n-1), self._vocab_size)
        self.H = nn.Linear(self._m * (self._n-1), self._h_dim)

        self.C = nn.Embedding(self._vocab_size+1, self._m) # idx==self._vocab_size is for padding

        if torch.cuda.is_available():
            self.U.cuda()
            #self.W.cuda()
            self.H.cuda()
            self.C.cuda()

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=2)

    def _create_features(self, x):
        feature_indicies = []
        for i in range(self._n-1):
            index = torch.zeros(x.shape).long() # bs x seq_len
            index[:,:i+1] = self._vocab_size
            index[:,i+1:] = x[:, i+1:]
            feature_indicies.append(index)
        feature_indicies = torch.stack(feature_indicies,2) # bs x seq_len x n-1

        if torch.cuda.is_available():
            feature_indicies = feature_indicies.cuda()

        embedding = self.C(feature_indicies) # bs x seq_len x n-1 x m
        embedding = embedding.view(x.shape[0], x.shape[1], -1) # bs x seq_len x (n-1 * m)
        return embedding

    def forward(self, x: torch.FloatTensor, inference:bool = False):
        features = self._create_features(x) # bs x seq_len x (n-1 * embeddings)
        logits = self.W(features) + self.U(self.tanh(self.H(features))) # bs x seq_len x vocab_size
        return logits


class LstmLM(nn.Module):
    # https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
    # pre-trained embeddings
    def __init__(self, vocab_size, word_vec=None):
        super().__init__()
        self._vocab_size = vocab_size
        self._hidden_dim = 100
        self._embedding_dim = 300
        self._keep_p = 0.5
        self._num_layers = 2
        self._word_embeddings = nn.Embedding(self._vocab_size,  self._embedding_dim)
        if word_vec is not None:
            self._word_embeddings.weight.data = word_vec
        else:
            self._word_embeddings.weight.data = nn.init.xavier_uniform(self._word_embeddings.weight.data)

        self.lstm = nn.LSTM(self._embedding_dim, self._hidden_dim, self._num_layers, dropout=1-self._keep_p)  # Input dim is 3, output dim is 3
        self.linear = nn.Linear(self._hidden_dim, self._vocab_size)  # Input dim is 3, output dim is 3
        self.linear.weight.data = nn.init.xavier_uniform(self.linear.weight.data)

        if torch.cuda.is_available():
            self._word_embeddings.cuda()
            self.lstm.cuda()
            self.linear.cuda()

        self.dropout = nn.Dropout(p=1-self._keep_p)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x: torch.FloatTensor, inference:bool = False):
        embedding = self.dropout(self._word_embeddings(x))
        lstm_out, hidden = self.lstm(embedding, None)
        lstm_out = self.dropout(lstm_out)
        logits = self.linear(lstm_out)
        return logits
