import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Baseline(nn.Module):

    def __init__(self, embedding_dim, vocab):
        super(Baseline, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(vocab.vectors)
        self.fc = nn.Linear(embedding_dim, 1)

    def forward(self, x, lengths=None):
        embedded = self.embedding(x)

        average = embedded.mean(0)
        output = self.fc(average)
        output = nn.functional.sigmoid(output)

        return output.squeeze()

class CNN_LSTM_2_conv(nn.Module):
    def __init__(self, embedding_dim, vocab, hidden_dim=100):
        super(CNN_LSTM_2_conv, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(vocab.vectors)
        self.hidden_dim = hidden_dim

        self.conv1 = nn.Conv2d(1, 50,kernel_size=(2,embedding_dim)) #in_channels, out_chanels, kernel_size
        self.conv2 = nn.Conv2d(1, 50,kernel_size=(2,embedding_dim)) #in_channels, out_chanels, kernel_size
        
        
        self.lstm1 = nn.LSTM(embedding_dim,hidden_dim)
        target_size = 1
        self.fc1 = nn.Linear(hidden_dim, target_size)

    def forward(self, x, lengths):
    #   embeds = self.word_embeddings(sentence)
    #     lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
    #     tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
    #     tag_scores = F.log_softmax(tag_space, dim=1)
    #     return tag_scores
        
        x = self.embedding(x)
        # print('x.shape',x.shape)

        x = x.unsqueeze(0)
        # print('x = x.unsqueeze(0)',x.shape)
        x = x.transpose(1,2) # swaps 2nd and 3rd dimension
        # print('x = x.transpose(1,2)',x.shape)
        x = x.transpose(0,1) # swaps 1st and 2nd dimension, now it has the correct input (n_samples,channels, height, width)
        # print('x = x.transpose(0,1)',x.shape)
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x)) 
        # print('x1 = F.relu(self.conv1(x))',x1.shape)
        
        x1 = x1.squeeze(3)
        x2 = x2.squeeze(3)
        # print('x1 = x1.squeeze(3) hahaha',x1.shape)
        pool1 = nn.MaxPool1d(x1.size(2), 1)
        pool2 = nn.MaxPool1d(x2.size(2), 1) #maxpool
        
        x1 = pool1(x1)
        x2 = pool2(x2)
        # print('x1 = x1.squeeze(3) here',x1.shape)

        x = torch.cat((x1,x2),1)
        # print('pool1 = nn.MaxPool1d(x1.size(2), 1)',x.shape)
        # x = concat.view(-1,100)
        # print('x = x.',x.shape)
        
        x = x.transpose(1,2)
        # print('x === x.squeeze(1)',x.shape)
        x = x.transpose(0,1)
        # print('x === x.squeeze(0)',x.shape)


        #  pack the word embeddings in the batch together and 
        # run the RNN on this object
        # x = pack_padded_sequence(x,lengths) 
        # print('x.shape',x.shape)
        x,(h,c) = self.lstm1(x)
        # print('x.shape',x.shape)
        # print('h.shape',h.shape)
        # print('len(h)',len(h))
        # print('(x.view(len(lengths),-1)).shape',(x.view(len(lengths),-1)).shape)
        h_result = self.fc1(h.squeeze(0))
        # h_result = F.log_softmax(tag_space, dim=1) --> change to this with multiple cat
        h_result = torch.sigmoid(h_result)
        

        return h_result.squeeze(1)
