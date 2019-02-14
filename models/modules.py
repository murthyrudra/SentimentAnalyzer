__author__ = 'rudramurthy'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import re

class SubwordModule(nn.Module):

    def __init__(self, ngrams, inputDim, outDim):
        super(SubwordModule, self).__init__()

        self.ngrams = ngrams
        self.inputDim = inputDim

        self.outDim = outDim

# in_channels is the same as rgb equivalent in vision
# for nlp in_channels is 1 unless we are using multiple feature representations
# out_channels is the number of features extracted by cnn layer
# kernel_size is dimension of the cnn filter
# as we are flattening the input representation we need to specify stride as number of input feature dimension

        self.conv = nn.Conv1d(in_channels=1, out_channels=self.outDim, kernel_size= self.ngrams * self.inputDim, stride = self.inputDim, bias=False)
        nn.init.xavier_normal_(self.conv.weight)


    def forward(self, x):
# get the convolved output which is of size miniBatch , out_channels , (reduced_input_character_length)

        x_conv = self.conv(x)
# reshape the output into miniBatch , out_channels * (reduced_input_character_length)
        x_convOut = x_conv.view(x_conv.size()[0], x_conv.size()[1] * x_conv.size()[2])

# apply max_pool1d with stride as reduced_input_character_length
        x = F.max_pool1d(x_convOut.unsqueeze(1), x_conv.size()[2])
        return x

class SubwordModuleSigmoid(nn.Module):

    def __init__(self, ngrams, inputDim, outDim):
        super(SubwordModuleSigmoid, self).__init__()

        self.ngrams = ngrams
        self.inputDim = inputDim

        self.outDim = outDim

# in_channels is the same as rgb equivalent in vision
# for nlp in_channels is 1 unless we are using multiple feature representations
# out_channels is the number of features extracted by cnn layer
# kernel_size is dimension of the cnn filter
# as we are flattening the input representation we need to specify stride as number of input feature dimension

        self.conv = nn.Conv1d(in_channels=1, out_channels=self.outDim, kernel_size= self.ngrams * self.inputDim, stride = self.inputDim, bias=False)
        self.relu = nn.ReLU()
        nn.init.xavier_normal_(self.conv.weight)


    def forward(self, x):
# get the convolved output which is of size miniBatch , out_channels , (reduced_input_character_length)

        x_conv = self.conv(x)

        x_max = F.adaptive_max_pool1d(x_conv, 1).permute(0,2,1)

        return x_max

class OutputLayer(nn.Module):
    def __init__(self, inputDimension, outputDimension ):

        super(OutputLayer,self).__init__()

        self.inputDimension = inputDimension
        self.outputDimension = outputDimension

        self.linear = nn.Linear(self.inputDimension, self.outputDimension)

    def forward(self, x_in):

        output = self.linear(x_in)
        return output


class AttentionLayer(nn.Module):
    def __init__(self, inputDimension ):

        super(AttentionLayer,self).__init__()

        self.inputDimension = inputDimension * 2
        self.hiddenDimension = inputDimension

        self.attentionLayer1 = nn.Linear(self.inputDimension, self.hiddenDimension)
        self.attentionLayerTanh = nn.Tanh()
        self.attentionLayer2 = nn.Linear(self.hiddenDimension, 1)

    def forward(self, x_in):

        output = self.attentionLayer2(self.attentionLayerTanh(self.attentionLayer1(x_in)))
        return output

class BiLSTM(nn.Module):
    def __init__(self, vocabularySize, embedDimension, minNgrams, maxNgrams, charInputDim, charOutDim, hiddenDim, tagVocabSize, init_embedding ):
        super(BiLSTM,self).__init__()

        self.vocabularySize = vocabularySize
        self.embedDimension = embedDimension

        self.bilstmInputDim = self.embedDimension
        self.hiddenDim = hiddenDim
        self.outputSize = tagVocabSize

        self.embedLayer = nn.Embedding(self.vocabularySize, self.embedDimension, padding_idx=0)
        self.embedLayer.weight = Parameter(torch.Tensor(init_embedding))

        self.dropout_in = nn.Dropout()

        self.bilstm = nn.LSTM(self.bilstmInputDim, self.hiddenDim, 1, batch_first = True, bidirectional = True)

        self.outputLayer = OutputLayer(self.hiddenDim * 2, tagVocabSize)

        self.softmax = nn.Softmax(dim=1)
        self.nll_loss = nn.CrossEntropyLoss(size_average=True)

    def loss(self, x, length_of_sequence, batchSize, maxLength, target, mask, use_gpu):
        embedOut = self.embedLayer(x[0])

        finalWordOut = self.dropout_in(embedOut)

# convert the list of lengths into a Tensor
        seq_lengths = length_of_sequence
# get the sorted list of lengths and the correpsonding indices
        sorted_length, sorted_index = torch.sort(seq_lengths, dim=0, descending=True)

        _, rev_order = torch.sort(sorted_index)
# convert the given input into sorted order based on sorted indices
        if use_gpu:
            rnn_input = finalWordOut.index_select(0, sorted_index.cuda())
            correctLabels = target.index_select(0, sorted_index.cuda())
        else:
            rnn_input = finalWordOut.index_select(0, sorted_index)
            correctLabels = target.index_select(0, sorted_index)

# pack the sequence with pads so that Bi-LSTM returns zero for the remaining entries
        x = torch.nn.utils.rnn.pack_padded_sequence(rnn_input, sorted_length.data.tolist(), batch_first=True)

# forward pass to get the output from LSTM layer
        seq_output, hn = self.bilstm(x)

        unpacked_bilstm_out, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(seq_output, batch_first=True)
        # batchSize * maxLength * hidDimension
        unpacked_bilstm_out = unpacked_bilstm_out.contiguous()

        if use_gpu:
            sentence_rep = Variable(torch.FloatTensor(batchSize, self.hiddenDim * 2).cuda())
        else:
            sentence_rep = Variable(torch.FloatTensor(batchSize, self.hiddenDim * 2))

        count = 0
        for i in range(batchSize):
            sentence_rep[i] = torch.max(unpacked_bilstm_out[i, 0:length_of_sequence.data[i], :], dim=0)[0]
            count = count + length_of_sequence.data[i]

        output_scores = self.outputLayer(sentence_rep)

        prob_output = self.softmax(output_scores)
        pred, predIndex = torch.max(prob_output, dim=1 )

        return self.nll_loss(output_scores, correctLabels).sum() / batchSize, predIndex, correctLabels


class BiCNNLSTM(nn.Module):
    def __init__(self, vocabularySize, embedDimension, minNgrams, maxNgrams, charInputDim, charOutDim, hiddenDim, tagVocabSize, init_embedding ):
        super(BiCNNLSTM,self).__init__()

        self.vocabularySize = vocabularySize
        self.embedDimension = embedDimension

        self.minNgrams = minNgrams
        self.maxNgrams = maxNgrams
        self.charInputDim = charInputDim
        self.charOutDim = charOutDim

        self.bilstmInputDim = self.embedDimension + (self.maxNgrams - self.minNgrams + 1) * self.charOutDim
        self.hiddenDim = hiddenDim
        self.outputSize = tagVocabSize

        self.embedLayer = nn.Embedding(self.vocabularySize, self.embedDimension, padding_idx=0)
        self.embedLayer.weight = Parameter(torch.Tensor(init_embedding))

        self.charLayers = nn.ModuleList( [SubwordModuleSigmoid(i, self.charInputDim, self.charOutDim) for i in range(self.minNgrams, self.maxNgrams + 1) ])

        self.dropout_in = nn.Dropout()

        self.bilstm = nn.LSTM(self.bilstmInputDim, self.hiddenDim, 1, batch_first = True, bidirectional = True)

        # self.maxpoolOut = SubwordModule(2, self.hiddenDim * 2, self.hiddenDim)

        self.outputLayer = OutputLayer(self.hiddenDim * 2, tagVocabSize)

        # self.logsoftmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)
        self.nll_loss = nn.CrossEntropyLoss(size_average=True)

    def loss(self, x, length_of_sequence, batchSize, maxLength, target, mask, use_gpu):
        embedOut = self.embedLayer(x[0])

        charOut = []
# extract the sub-word features and the input is batchSize, sequenceLength, (numberOfCharacters * characterFeatures)
        for i,l in enumerate(self.charLayers):
            charOut.append(l(x[1]))

        # concatenate all extracted character features based on the last dimension
        finalCharOut = torch.cat( [charOut[i] for i,l in enumerate(self.charLayers)], 2)

        finalCharOut = finalCharOut.view(batchSize, maxLength, finalCharOut.size()[2])

# concatenate word representation and subword features
        finalWordOut = torch.cat( [finalCharOut, embedOut], 2)

        finalWordOut = self.dropout_in(finalWordOut)

# convert the list of lengths into a Tensor
        seq_lengths = length_of_sequence
# get the sorted list of lengths and the correpsonding indices
        sorted_length, sorted_index = torch.sort(seq_lengths, dim=0, descending=True)

        _, rev_order = torch.sort(sorted_index)
# convert the given input into sorted order based on sorted indices
        if use_gpu:
            rnn_input = finalWordOut.index_select(0, sorted_index.cuda())
            correctLabels = target.index_select(0, sorted_index.cuda())
        else:
            rnn_input = finalWordOut.index_select(0, sorted_index)
            correctLabels = target.index_select(0, sorted_index)

# pack the sequence with pads so that Bi-LSTM returns zero for the remaining entries
        x = torch.nn.utils.rnn.pack_padded_sequence(rnn_input, sorted_length.data.tolist(), batch_first=True)

# forward pass to get the output from LSTM layer
        seq_output, hn = self.bilstm(x)

        unpacked_bilstm_out, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(seq_output, batch_first=True)
        # batchSize * maxLength * hidDimension
        unpacked_bilstm_out = unpacked_bilstm_out.contiguous()

        if use_gpu:
            sentence_rep = Variable(torch.FloatTensor(batchSize, self.hiddenDim * 2).cuda())
        else:
            sentence_rep = Variable(torch.FloatTensor(batchSize, self.hiddenDim * 2))

        count = 0
        for i in range(batchSize):
            sentence_rep[i] = torch.max(unpacked_bilstm_out[i, 0:length_of_sequence.data[i], :], dim=0)[0]
            count = count + length_of_sequence.data[i]

        output_scores = self.outputLayer(sentence_rep)

        prob_output = self.softmax(output_scores)
        pred, predIndex = torch.max(prob_output, dim=1 )

        # log_output = self.logsoftmax(output_scores)

        return self.nll_loss(output_scores, correctLabels).sum() / batchSize, predIndex, correctLabels


class BiCNNLSTMCNN(nn.Module):
    def __init__(self, vocabularySize, embedDimension, minNgrams, maxNgrams, charInputDim, charOutDim, hiddenDim, tagVocabSize, init_embedding ):
        super(BiCNNLSTMCNN,self).__init__()

        self.vocabularySize = vocabularySize
        self.embedDimension = embedDimension

        self.minNgrams = minNgrams
        self.maxNgrams = maxNgrams
        self.charInputDim = charInputDim
        self.charOutDim = charOutDim

        self.bilstmInputDim = self.embedDimension + (self.maxNgrams - self.minNgrams + 1) * self.charOutDim
        self.hiddenDim = hiddenDim
        self.outputSize = tagVocabSize

        self.embedLayer = nn.Embedding(self.vocabularySize, self.embedDimension, padding_idx=0)
        self.embedLayer.weight = Parameter(torch.Tensor(init_embedding))

        self.charLayers = nn.ModuleList( [SubwordModuleSigmoid(i, self.charInputDim, self.charOutDim) for i in range(self.minNgrams, self.maxNgrams + 1) ])

        self.dropout_in = nn.Dropout()

        self.bilstm = nn.LSTM(self.bilstmInputDim, self.hiddenDim, 1, batch_first = True, bidirectional = True)

        self.maxpoolOut = SubwordModule(2, self.hiddenDim * 2, self.hiddenDim * 2)
        self.outputLayer = OutputLayer(self.hiddenDim * 2 , tagVocabSize)

        # self.logsoftmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)
        self.nll_loss = nn.CrossEntropyLoss(size_average=False)

    def loss(self, x, length_of_sequence, batchSize, maxLength, target, mask, use_gpu):
        embedOut = self.embedLayer(x[0])

        charOut = []
# extract the sub-word features and the input is batchSize, sequenceLength, (numberOfCharacters * characterFeatures)
        for i,l in enumerate(self.charLayers):
            charOut.append(l(x[1]))

        # concatenate all extracted character features based on the last dimension
        finalCharOut = torch.cat( [charOut[i] for i,l in enumerate(self.charLayers)], 2)

        finalCharOut = finalCharOut.view(batchSize, maxLength, finalCharOut.size()[2])

# concatenate word representation and subword features
        finalWordOut = torch.cat( [finalCharOut, embedOut], 2)

        finalWordOut = self.dropout_in(finalWordOut)

# convert the list of lengths into a Tensor
        seq_lengths = length_of_sequence
# get the sorted list of lengths and the correpsonding indices
        sorted_length, sorted_index = torch.sort(seq_lengths, dim=0, descending=True)

        _, rev_order = torch.sort(sorted_index)
# convert the given input into sorted order based on sorted indices
        if use_gpu:
            rnn_input = finalWordOut.index_select(0, sorted_index.cuda())
            correctLabels = target.index_select(0, sorted_index.cuda())
        else:
            rnn_input = finalWordOut.index_select(0, sorted_index)
            correctLabels = target.index_select(0, sorted_index)

# pack the sequence with pads so that Bi-LSTM returns zero for the remaining entries
        x = torch.nn.utils.rnn.pack_padded_sequence(rnn_input, sorted_length.data.tolist(), batch_first=True)

# forward pass to get the output from LSTM layer
        seq_output, hn = self.bilstm(x)

        unpacked_bilstm_out, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(seq_output, batch_first=True)
        unpacked_bilstm_out = unpacked_bilstm_out.contiguous()

        sentence_rep_bilstm = unpacked_bilstm_out.view(batchSize,1, maxLength * self.hiddenDim * 2)

        count = 0
        for i in range(batchSize):
            count = count + length_of_sequence.data[i]

        sentence_rep = self.maxpoolOut(sentence_rep_bilstm)
        sentence_rep = sentence_rep.view(batchSize, self.hiddenDim * 2)

        # output_scores = self.outputLayer(self.tanh(sentence_rep))
        output_scores = self.outputLayer(sentence_rep)

        prob_output = self.softmax(output_scores)
        pred, predIndex = torch.max(prob_output, dim=1 )

        return self.nll_loss(output_scores, correctLabels).sum() / count, predIndex, correctLabels



class BiCNNLSTMAttention(nn.Module):
    def __init__(self, vocabularySize, embedDimension, minNgrams, maxNgrams, charInputDim, charOutDim, hiddenDim, tagVocabSize, init_embedding ):
        super(BiCNNLSTMAttention,self).__init__()

        self.vocabularySize = vocabularySize
        self.embedDimension = embedDimension

        self.minNgrams = minNgrams
        self.maxNgrams = maxNgrams
        self.charInputDim = charInputDim
        self.charOutDim = charOutDim

        self.bilstmInputDim = self.embedDimension + (self.maxNgrams - self.minNgrams + 1) * self.charOutDim
        self.hiddenDim = hiddenDim
        self.outputSize = tagVocabSize

        self.embedLayer = nn.Embedding(self.vocabularySize, self.embedDimension, padding_idx=0)
        self.embedLayer.weight = Parameter(torch.Tensor(init_embedding))

        self.charLayers = nn.ModuleList( [SubwordModuleSigmoid(i, self.charInputDim, self.charOutDim) for i in range(self.minNgrams, self.maxNgrams + 1) ])

        self.dropout_in = nn.Dropout()

        self.bilstm = nn.LSTM(self.bilstmInputDim, self.hiddenDim, 1, batch_first = True, bidirectional = True)

        self.attention = AttentionLayer(self.hiddenDim)

        if self.outputSize == 2:
            self.outputLayer = OutputLayer(self.hiddenDim * 2 , 1)
            self.bce_loss = nn.BCEWithLogitsLoss()
            self.sigmoid = nn.Sigmoid()
        else:
            self.outputLayer = OutputLayer(self.hiddenDim * 2 , self.outputSize)
            self.softmax = nn.Softmax(dim=1)
            self.nll_loss = nn.CrossEntropyLoss(size_average=False)

    def loss(self, x, length_of_sequence, batchSize, maxLength, target, mask, use_gpu):
        embedOut = self.embedLayer(x[0])

        charOut = []
# extract the sub-word features and the input is batchSize, sequenceLength, (numberOfCharacters * characterFeatures)
        for i,l in enumerate(self.charLayers):
            charOut.append(l(x[1]))

        # concatenate all extracted character features based on the last dimension
        finalCharOut = torch.cat( [charOut[i] for i,l in enumerate(self.charLayers)], 2)

        finalCharOut = finalCharOut.view(batchSize, maxLength, finalCharOut.size()[2])

# concatenate word representation and subword features
        finalWordOut = torch.cat( [finalCharOut, embedOut], 2)

        finalWordOut = self.dropout_in(finalWordOut)

# convert the list of lengths into a Tensor
        seq_lengths = length_of_sequence
# get the sorted list of lengths and the correpsonding indices
        sorted_length, sorted_index = torch.sort(seq_lengths, dim=0, descending=True)

        _, rev_order = torch.sort(sorted_index)
# convert the given input into sorted order based on sorted indices
        if use_gpu:
            rnn_input = finalWordOut.index_select(0, sorted_index.cuda())
            correctLabels = target.index_select(0, sorted_index.cuda())
        else:
            rnn_input = finalWordOut.index_select(0, sorted_index)
            correctLabels = target.index_select(0, sorted_index)

# pack the sequence with pads so that Bi-LSTM returns zero for the remaining entries
        x = torch.nn.utils.rnn.pack_padded_sequence(rnn_input, sorted_length.data.tolist(), batch_first=True)

# forward pass to get the output from LSTM layer
        seq_output, hn = self.bilstm(x)

        unpacked_bilstm_out, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(seq_output, batch_first=True)
        unpacked_bilstm_out = unpacked_bilstm_out.contiguous()

        attn_output = self.attention(unpacked_bilstm_out).view(batchSize, maxLength)

        attn_energies = Variable(torch.zeros(batchSize, maxLength))
        if use_gpu:
            attn_energies = attn_energies.cuda()

        for i in range(batchSize):
            attn_energies[i, 0:length_of_sequence.data[i]] = F.softmax(attn_output[i][:length_of_sequence.data[i]], dim=0)

        attn_energies = attn_energies.unsqueeze(1)

        weighted_rep = attn_energies.bmm(unpacked_bilstm_out).squeeze(1)

        count = 0
        for i in range(batchSize):
            count = count + length_of_sequence.data[i]

        output_scores = self.outputLayer(weighted_rep)

        if self.outputSize == 2:
            loss = self.bce_loss(output_scores, correctLabels)

            prob_output = self.sigmoid(output_scores)
            return loss, prob_output, correctLabels
        else:
            prob_output = self.softmax(output_scores)
            pred, predIndex = torch.max(prob_output, dim=1 )
            loss = self.nll_loss(output_scores, correctLabels).sum() / count
            return loss, predIndex, correctLabels

    def forward(self, x, length_of_sequence, batchSize, maxLength, use_gpu):
        embedOut = self.embedLayer(x[0])

        charOut = []
    # extract the sub-word features and the input is batchSize, sequenceLength, (numberOfCharacters * characterFeatures)
        for i,l in enumerate(self.charLayers):
            charOut.append(l(x[1]))

        # concatenate all extracted character features based on the last dimension
        finalCharOut = torch.cat( [charOut[i] for i,l in enumerate(self.charLayers)], 2)

        finalCharOut = finalCharOut.view(batchSize, maxLength, finalCharOut.size()[2])

    # concatenate word representation and subword features
        finalWordOut = torch.cat( [finalCharOut, embedOut], 2)

        finalWordOut = self.dropout_in(finalWordOut)

    # convert the list of lengths into a Tensor
        seq_lengths = length_of_sequence
    # get the sorted list of lengths and the correpsonding indices
        sorted_length, sorted_index = torch.sort(seq_lengths, dim=0, descending=True)

        _, rev_order = torch.sort(sorted_index)
    # convert the given input into sorted order based on sorted indices
        if use_gpu:
            rnn_input = finalWordOut.index_select(0, sorted_index.cuda())
        else:
            rnn_input = finalWordOut.index_select(0, sorted_index)

    # pack the sequence with pads so that Bi-LSTM returns zero for the remaining entries
        x = torch.nn.utils.rnn.pack_padded_sequence(rnn_input, sorted_length.data.tolist(), batch_first=True)

    # forward pass to get the output from LSTM layer
        seq_output, hn = self.bilstm(x)

        unpacked_bilstm_out, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(seq_output, batch_first=True)
        unpacked_bilstm_out = unpacked_bilstm_out.contiguous()

        attn_output = self.attention(unpacked_bilstm_out).view(batchSize, maxLength)

        attn_energies = Variable(torch.zeros(batchSize, maxLength))
        if use_gpu:
            attn_energies = attn_energies.cuda()

        for i in range(batchSize):
            attn_energies[i, 0:length_of_sequence.data[i]] = F.softmax(attn_output[i][:length_of_sequence.data[i]], dim=0)

        attn_energies = attn_energies.unsqueeze(1)

        weighted_rep = attn_energies.bmm(unpacked_bilstm_out).squeeze(1)

        count = 0
        for i in range(batchSize):
            count = count + length_of_sequence.data[i]

        output_scores = self.outputLayer(weighted_rep)

        if self.outputSize == 2:
            prob_output = self.sigmoid(output_scores)
            return prob_output
        else:
            prob_output = self.softmax(output_scores)
            pred, predIndex = torch.max(prob_output, dim=1 )
            return predIndex
