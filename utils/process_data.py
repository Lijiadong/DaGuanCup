import torch.utils.data as data
import torch
from utils.config import *
class Dataset(data.Dataset):
    def __init__(self, data, labels, vocab_size, labels_dict):
        self.data = data
        self.labels = labels
        self.vocab_size = vocab_size
        self.labels_dict = labels_dict
        self.label_unk_idx = labels_dict['o']

    def __getitem__(self, index):
        _data = self.data[index]
        _labels = self.labels[index]
        _data = torch.Tensor(_data)
        _labels = torch.Tensor(_labels)
        return _data, _labels, self.vocab_size, self.label_unk_idx

    def __len__(self):
        return len(self.data)

def collate_fn(data):
    data.sort(key=lambda x: len(x[0]), reverse=True)
    # seperate source and target sequences
    data, labels, vocab_sizes, label_unk_idx = zip(*data)
    vocab_size = vocab_sizes[0]
    label_unk_idx = label_unk_idx[0]
    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.ones(len(sequences), max(lengths)).long() * vocab_size
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths

    def merge_labels(sequences):
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.ones(len(sequences), max(lengths)).long() * label_unk_idx
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths
    # merge sequences (from tuple of 1D tensor to 2D tensor)
    data, data_lengths = merge(data)
    labels, label_lengths = merge_labels(labels)

    data = data.transpose(1, 0)
    labels = labels.transpose(1, 0)
    if USE_CUDA:
        data = data.cuda()
        labels = labels.cuda()


    return data, data_lengths, labels, label_lengths


def read_data(file_path):
    data = []
    labels = []
    vocab_size = 0
    labels_dict = {}
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                _data = []
                _labels = []
                temp = line.split(' ')
                for t in temp:
                    t = t.strip()
                    if t:
                        idx, label = t.split('/')
                        idx = [int(e) for e in idx.split('_')]
                        if max(idx) > vocab_size:
                            vocab_size = max(idx)
                        if label not in labels_dict:
                            labels_dict[label] = len(labels_dict)
                        label_idx = labels_dict[label]
                        _data.extend(idx)
                        _labels.extend([label_idx] * len(idx))
                data.append(_data)
                labels.append(_labels)
    return data, labels, vocab_size, labels_dict

def prepare_data(batch_size, shuffle=True):
    file_path = './data/train.txt'
    data, labels, vocab_size, labels_dict = read_data(file_path)
    train_length = int(len(data) * args['data_split'])
    train_data, train_labels = data[:train_length], labels[:train_length]
    dev_data, dev_labels = data[train_length:], labels[train_length:]

    def get_seq(data, labels):
        dataset = Dataset(data, labels, vocab_size, labels_dict)
        data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  collate_fn=collate_fn)
        return data_loader

    train = get_seq(train_data, train_labels)
    dev = get_seq(dev_data, dev_labels)
    vocab_size += 1

    return train, dev, vocab_size, labels_dict

def prepare_test_data():
    file_path = './data/test.txt'
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                temp = line.split('_')
                temp = torch.Tensor([int(t) for t in temp]).long()
                if USE_CUDA:
                    temp = temp.cuda()
                data.append(temp)
    return data
