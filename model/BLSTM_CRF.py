from typing import Tuple
import torch
import torch.nn as nn
from TorchCRF import CRF
from utils.config import *
from torch import optim
from tqdm import tqdm
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
import os

class BLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, input_size, hidden_size, num_labels, n_layers, lr, dropout):
        super(BLSTM_CRF, self).__init__()
        self.name = "BLSTM_CRF"
        self.blstm = BLSTM(vocab_size, input_size, hidden_size, num_labels, n_layers, dropout)
        self.crf = CRF(num_labels)

        self.blstm_optimizer = optim.Adam(self.blstm.parameters(), lr=lr, weight_decay=1e-4)
        self.crf_optimizer = optim.Adam(self.crf.parameters(), lr=lr, weight_decay=1e-4)

        self.loss = 0
        self.print_every = 1
        self.max_score = 0.


        if USE_CUDA:
            self.blstm = self.blstm.cuda()
            self.crf = self.crf.cuda()

    def print_loss(self):
        print_loss_avg = self.loss / self.print_every
        self.print_every += 1
        return 'L:{:.4f}'.format(print_loss_avg)

    def inference(self, data, labels_dict, path, dev_score):
        results = []
        _labels_dict = dict(zip(labels_dict.values(), labels_dict.keys()))
        for i, _data in enumerate(data):
            data_length = [len(_data)]
            _data = _data.unsqueeze(1)
            outputs, _ = self.blstm(_data, data_length, 1)
            mask = torch.zeros(_data.data.shape)
            for i, length in enumerate(data_length):
                mask[:length, i] = torch.ones(length)
            if USE_CUDA:
                mask = mask.cuda()
            _pred_labels = self.crf.viterbi_decode(outputs, mask)
            _pred_labels = [_labels_dict[l[0]] for l in _pred_labels]
            pre_label = _pred_labels[0]
            pre_index = 0
            _data = _data.squeeze(1).cpu().detach().numpy()
            temp = []
            for j, label in enumerate(_pred_labels[1:]):
                if label != pre_label:
                    temp.append('_'.join([str(e) for e in _data[pre_index:j+1]]) + '/' + pre_label)
                    pre_label = label
                    pre_index = j + 1
                if j == (len(_pred_labels) - 2) and (j + 1) >= pre_index:
                    temp.append('_'.join([str(e) for e in _data[pre_index:]]) + '/' + label)
            results.append('  '.join(temp))


        save_path = os.path.join(path, 'result_' + str(round(dev_score, 4)) + '.txt')
        with open(save_path, 'w') as f:
            for line in results:
                f.write(line + '\n')



    def evaluate(self, dev):
        avg_f1 = 0.
        pbar = tqdm(enumerate(dev), total=len(dev))
        for i, data in pbar:
            results = self.eval_batch(data[0], data[1])
            gold_labels = data[2].cpu().detach().numpy()
            y_true = []
            y_pred = []
            for j, length in enumerate(data[3]):
                y_true.append(list(gold_labels[:length, j]))
                y_pred.append([results[k][j] for k in range(length)])

            y_true = MultiLabelBinarizer().fit_transform(y_true)
            y_pred = MultiLabelBinarizer().fit_transform(y_pred)
            avg_f1 += f1_score(y_true, y_pred, average='micro')
            pbar.set_description('F1_SCORE:{:.4f}'.format(avg_f1 / float(len(dev))))
        return avg_f1 / float(len(dev))


    def eval_batch(self, data, data_lengths):
        outputs, _ = self.blstm(data, data_lengths)
        mask = torch.zeros(data.data.shape)
        for i, length in enumerate(data_lengths):
            mask[:length, i] = torch.ones(length)
        if USE_CUDA:
            mask = mask.cuda()
        results = self.crf.viterbi_decode(outputs, mask)
        return results


    def train_batch(self, data, data_lengths, labels, label_lengths, clip, reset):

        if reset:
            self.loss = 0
            self.print_every = 1

        outputs, _ = self.blstm(data, data_lengths)
        mask = torch.zeros(data.data.shape)
        for i, length in enumerate(data_lengths):
            mask[:length, i] = torch.ones(length)
        if USE_CUDA:
            mask = mask.cuda()
        score = self.crf.forward(outputs, labels, mask)

        loss = -torch.mean(score)
        loss.backward()

        # Clip gradient norm
        bc = torch.nn.utils.clip_grad_norm_(self.blstm.parameters(), clip)
        cc = torch.nn.utils.clip_grad_norm_(self.crf.parameters(), clip)

        # Update parameters with optimizers
        self.blstm_optimizer.step()
        self.crf_optimizer.step()
        self.loss += loss.item()

        return loss.item()


class BLSTM(nn.Module):
    def __init__(self, vocab_size, input_size, hidden_size, num_labels, n_layers, dropout):
        super(BLSTM, self).__init__()
        self.name = "BLSTM"
        self.vocab_size = vocab_size
        self.embedding_dim = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(vocab_size, input_size)
        self.embedding_dropout = nn.Dropout(dropout)
        self.blstm = nn.LSTM(input_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)
        self.out = nn.Linear(2 * hidden_size, num_labels)

    def init_hidden(self, batch_size) -> Tuple[torch.Tensor]:
        """
        initialize hidden state
        :return: (hidden state, cell of LSTM)
        """

        h = torch.zeros(self.n_layers * 2, batch_size, self.hidden_size)
        c = torch.zeros(self.n_layers * 2, batch_size, self.hidden_size)
        if USE_CUDA:
            h = h.cuda()
            c = c.cuda()
        return h, c

    def forward(self, input_seqs, input_lengths, batch_size=args['batch']):
        embedded = self.embedding(input_seqs)
        embedded = self.embedding_dropout(embedded)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)

        hidden = self.init_hidden(batch_size)
        outputs, hidden = self.blstm(packed, hidden)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        outputs = self.out(outputs)
        return outputs, hidden
