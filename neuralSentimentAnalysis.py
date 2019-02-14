from __future__ import print_function

__author__ = 'rudramurthy'
"""
Implementation of Bi-directional LSTM-CNNs model for Sentiment Analysis.
"""

import os
import sys
import codecs
import math
import re
sys.path.append(".")
sys.path.append("..")

import time
import argparse

import numpy as np
import torch
import json
from tqdm import tqdm
from utils.vocab import Vocab, CharVocab
from torch.optim import SGD, ASGD
from models.modules import BiCNNLSTMAttention, BiCNNLSTMCNN
from torch.optim.lr_scheduler import ExponentialLR
from utils.logger import get_logger
from utils.utilsLocal import *

from sklearn.metrics import precision_recall_fscore_support as score

def main():
	parser = argparse.ArgumentParser(description='Implementation of Bi-directional LSTM-CNNs model for Sentiment Analysis')
	parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
	parser.add_argument('--batch_size', type=int, default=5, help='Number of sentences in each batch')
	parser.add_argument('--hidden_size', type=int, default=200, help='Number of hidden units in RNN')
	parser.add_argument('--num_filters', type=int, default=15, help='Number of filters in CNN')
	parser.add_argument('--min_filter_width', type=int, default=1, help='Number of filters in CNN')
	parser.add_argument('--max_filter_width', type=int, default=7, help='Number of filters in CNN')
	parser.add_argument('--embedDimension', type=int, default=300, help='embedding dimension')
	parser.add_argument('--learning_rate', type=float, default=0.4, help='Learning rate')
	parser.add_argument('--decay_rate', type=float, default=0.1, help='Decay rate of learning rate')
	parser.add_argument('--gamma', type=float, default=0.0, help='weight for regularization')
	parser.add_argument('--schedule', type=int, default=1, help='schedule for learning rate decay')
	parser.add_argument('--embedding_vectors', help='path for embedding dict')
	parser.add_argument('--data')

	parser.add_argument('--vocabChar')
	parser.add_argument('--vocabOutput')
	parser.add_argument('--vocabInput')

	parser.add_argument('--use_gpu', type=int, default=1, help='use gpu')
	parser.add_argument('--train_from', type=int, default=0, help='Train From')
	parser.add_argument('--perform_evaluation', type=int, default=0, help='Perform Evaluation')

	parser.add_argument('--save-dir')

	args = parser.parse_args()

	logger = get_logger("Neural Senti Analyzer")

	data_path = args.data

	num_epochs = args.num_epochs
	batch_size = args.batch_size
	hidden_size = args.hidden_size

	num_filters = args.num_filters
	min_filter_width = args.min_filter_width
	max_filter_width = args.max_filter_width

	learning_rate = args.learning_rate
	momentum = 0.001 * learning_rate
	decay_rate = args.decay_rate
	gamma = args.gamma
	schedule = args.schedule

	embedding_path = args.embedding_vectors

	save_dir = args.save_dir

	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	inputVocabulary = Vocab()
	charVocabulary = CharVocab()
	targetVocabulary = Vocab()

	embeddingDimension = args.embedDimension

	if args.vocabChar:
		with open(args.vocabChar, "r") as f:
			charVocabulary.__dict__ = json.load(f)
		charVocabulary.set_freeze()
		charVocabulary.process()

	if args.vocabOutput:
		with open(args.vocabOutput, "r") as f:
			targetVocabulary.__dict__ = json.load(f)
		targetVocabulary.set_freeze()
		targetVocabulary.process()

	embedding_vocab = None

	if args.embedding_vectors:
		print(args.embedding_vectors)
		embedd_dict, embedding_vocab, reverse_word_vocab, vocabularySize, embeddingDimension = load_embeddings(embedding_path)
		print("Read Word Embedding of dimension " + str(embeddingDimension) + " for " + str(vocabularySize) + " number of words")

		for everyWord in embedding_vocab:
			inputVocabulary.add(everyWord)
		inputVocabulary.set_freeze()
		inputVocabulary.process()
	else:
		if args.vocabInput:
			with open(args.vocabInput, "r") as f:
				inputVocabulary.__dict__ = json.load(f)
			inputVocabulary.set_freeze()
			inputVocabulary.process()
		else:
			print("Neither pre-trained word embeddings nor input vocabulary is specified")
			exit()

	if charVocabulary.__is_empty__():
		charVocabulary.add("<S>")
		charVocabulary.add("</S>")

	if args.perform_evaluation == 0:

		train_split, valid_split, test_split = readCorpus(data_path, targetVocabulary)
		targetVocabulary.set_freeze()
		targetVocabulary.process()

		train_sentences, train_labels, maxSequenceLength = extractData(train_split, charVocabulary, targetVocabulary)
		valid_sentences, valid_labels, maxSequenceLength = extractData(valid_split, charVocabulary, targetVocabulary)
		test_sentences, test_labels, maxSequenceLength = extractData(test_split, charVocabulary, targetVocabulary)

		print(targetVocabulary._tok_to_ind)

		tmp_filename = '%s/output.vocab' % (save_dir)
		with open(tmp_filename, "w") as f:
			json.dump(targetVocabulary.__dict__, f)
		targetVocabulary.set_freeze()

		if not charVocabulary.get_freeze():
			tmp_filename = '%s/char.vocab' % (save_dir)
			with open(tmp_filename, "w") as f:
				json.dump(charVocabulary.__dict__, f)
			charVocabulary.set_freeze()

	embeddingDimension = args.embedDimension
	word_embedding = np.random.uniform(-0.1, 0.1, (inputVocabulary.__len__(), embeddingDimension) )
	if not args.vocabInput:

		if args.embedding_vectors:
			for everyWord in inputVocabulary._tok_to_ind:
				if everyWord in embedding_vocab:
					word_embedding[ inputVocabulary.__get_word__(everyWord) ] = embedd_dict[embedding_vocab[everyWord]]

			tmp_filename = '%s/input.vocab' % (save_dir)
			with open(tmp_filename, "w") as f:
				json.dump(inputVocabulary.__dict__, f)
			inputVocabulary.set_freeze()

			del embedd_dict
			del reverse_word_vocab
			del vocabularySize
			del embedding_vocab

	print("Read " + str(targetVocabulary.__len__()) + " number of target words")
	print("Read " + str(inputVocabulary.__len__()) + " number of input words")
	print("Read " + str(charVocabulary.__len__()) + " number of characters")

	if args.perform_evaluation == 0:
		print("Number of epochs = " +  str(num_epochs))
		print("Mini-Batch size = " +  str(batch_size))
		print("LSTM Hidden size = " +  str(hidden_size))
		print("Features per CNN filter = " +  str(num_filters))
		print("Minimum ngrams for CNN filter = " +  str(min_filter_width))
		print("Maximum ngrams for CNN filter = " +  str(max_filter_width))
		print("Initial Learning Rate = " +  str(learning_rate))

	network = BiCNNLSTMAttention(inputVocabulary.__len__(), embeddingDimension, min_filter_width, max_filter_width, charVocabulary.__len__(), num_filters, hidden_size, targetVocabulary.__len__() , word_embedding)

	lr = learning_rate

	optim = SGD(network.parameters(), lr=lr, momentum=momentum, nesterov=True)
	# optim = ASGD(network.parameters(), lr=lr)
	scheduler = ExponentialLR(optim, gamma=0.9)

	if args.perform_evaluation == 1:
		network.eval()

		flag = True
		while flag:
			print("Enter a sample review")
			sentence = input()

			x_input, batch_length, current_batch_size, current_max_sequence_length = constructBatchOnline([sentence.split(" ")], inputVocabulary, charVocabulary, max_filter_width, args.use_gpu)
			pred_output = network.forward(x_input, batch_length, current_batch_size, current_max_sequence_length, args.use_gpu)

			if targetVocabulary.__len__() == 2:
				pred_output = pred_output.data.cpu().numpy()
				pred = []

				if pred_output[0] >= 0.5:
					print("Predicted Sentiment is " + targetVocabulary.__get_index__(1) + " with confidence of " + str(pred_output[0]))
				else:
					print("Predicted Sentiment is " + targetVocabulary.__get_index__(0) + " with confidence of " + str(pred_output[0]))
			else:
				predicted.extend(pred_output.data.cpu().numpy().tolist())

			print("Do you wish to continue (Y/N)")
			option = input().lower()
			if "n" in option:
				flag = False

	else:
		num_batches = len(train_sentences) / batch_size + 1
		print("Training....")

		if args.use_gpu == 1:
			network.cuda()

		if args.train_from == 1:
			print("Loading pre-trained model from " + args.train_from)

			network.load_state_dict(torch.load(args.train_from))

		prev_error = 1000.0
		epoch = 1

		while lr > 0.002:
			print('Epoch %d ( learning rate=%.4f): ' % (epoch, optim.defaults['lr']))

			train_err = 0.
			train_total = 0

			start_time = time.time()
			num_back = 0
			network.train()

			count = 0
			count_batch = 0

			with tqdm(total= ( len(train_sentences)) ) as pbar:

				for inputs,targets in batch2(train_sentences, train_labels, batch_size):

					x_input, batch_length, current_batch_size, current_max_sequence_length, y_output, mask = constructBatch(inputs, targets, inputVocabulary, charVocabulary, targetVocabulary, max_filter_width, args.use_gpu)

					optim.zero_grad()

					loss, pred_output, correct_labels = network.loss(x_input, batch_length, current_batch_size, current_max_sequence_length, y_output, mask, args.use_gpu)

					loss.backward()
					optim.step()

					train_err += loss.item()
					train_total += batch_length.data.sum()

					count = count + current_batch_size
					count_batch = count_batch + 1

					time_ave = (time.time() - start_time) / count
					time_left = (num_batches - count_batch) * time_ave

					pbar.update(batch_size)

				print('train: %d loss: %.4f, time: %.2fs' % (num_batches, train_err / count, time.time() - start_time))

			network.eval()
			current_epoch_loss = 0.0
			accuracy = 0.0
			prob = 0.0
			count = 0

			true = []
			predicted = []

			for inputs, targets in batch2(valid_sentences, valid_labels, 10):
				x_input, batch_length, current_batch_size, current_max_sequence_length, y_output, mask = constructBatch(inputs, targets, inputVocabulary, charVocabulary, targetVocabulary, max_filter_width, args.use_gpu)

				loss, pred_output, correct_labels = network.loss(x_input, batch_length, current_batch_size, current_max_sequence_length, y_output, mask, args.use_gpu)
				current_epoch_loss = current_epoch_loss + loss.item()

				true_labels = []
				temp_true = correct_labels.data.cpu().numpy()
				for i in range(len(temp_true)):
					true_labels.append(int(temp_true[i]))

				true.extend(true_labels)

				if targetVocabulary.__len__() == 2:
					pred_output = pred_output.data.cpu().numpy()
					pred = []

					for i in range(len(pred_output)):
						if pred_output[i] >= 0.5:
							pred.append(1)
						else:
							pred.append(0)
					predicted.extend(pred)
				else:
					predicted.extend(pred_output.data.cpu().numpy().tolist())

			precision, recall, fscore, support = score(true, predicted, average="macro")

			print('precision: {}'.format(precision))
			print('recall: {}'.format(recall))
			print('fscore: {}'.format(fscore))
			print('support: {}'.format(support))

			print("Development data loss " + str(current_epoch_loss))

			if epoch > 1:
				if current_epoch_loss > prev_error:
					lr = lr * 0.7
					momentum = momentum * 0.7
					optim = SGD(network.parameters(), lr=lr, momentum=momentum, nesterov=True)

					network.load_state_dict(torch.load(save_dir + "/model"))
					network.eval()
				else:
					prev_error = current_epoch_loss
					torch.save(network.state_dict(), save_dir + "/model")

					epoch = epoch + 1

					network.eval()
					true = []
					predicted = []

					print("\n" * 1)

					for inputs, targets in batch2(test_sentences, test_labels, 10):
						x_input, batch_length, current_batch_size, current_max_sequence_length, y_output, mask = constructBatch(inputs, targets, inputVocabulary, charVocabulary, targetVocabulary, max_filter_width, args.use_gpu)

						loss, pred_output, correct_labels = network.loss(x_input, batch_length, current_batch_size, current_max_sequence_length, y_output, mask, args.use_gpu)

						true_labels = []
						temp_true = correct_labels.data.cpu().numpy()
						for i in range(len(temp_true)):
							true_labels.append(int(temp_true[i]))

						true.extend(true_labels)

						if targetVocabulary.__len__() == 2:
							pred_output = pred_output.data.cpu().numpy()
							pred = []

							for i in range(len(pred_output)):
								if pred_output[i] >= 0.5:
									pred.append(1)
								else:
									pred.append(0)
							predicted.extend(pred)
						else:
							predicted.extend(pred_output.data.cpu().numpy().tolist())

					precision, recall, fscore, support = score(true, predicted, average="macro")

					print('precision: {}'.format(precision))
					print('recall: {}'.format(recall))
					print('fscore: {}'.format(fscore))
					print('support: {}'.format(support))

					from sklearn.metrics import classification_report as clsr
					scores = clsr(true, predicted)
					scores = list(map(lambda r: re.sub('\s\s+', '\t', r), scores.split("\n")))
					scores[-2] = '\t' + scores[-2]
					scores = '\n'.join(scores)
					print(scores)

					print("\n" * 1)

			else:
				prev_error = current_epoch_loss
				torch.save(network.state_dict(), save_dir + "/model")
				epoch = epoch + 1

		for inputs, targets in batch2(test_sentences, test_labels, 10):
			x_input, batch_length, current_batch_size, current_max_sequence_length, y_output, mask = constructBatch(inputs, targets, inputVocabulary, charVocabulary, targetVocabulary, max_filter_width, args.use_gpu)

			loss, pred_output, correct_labels = network.loss(x_input, batch_length, current_batch_size, current_max_sequence_length, y_output, mask, args.use_gpu)

			true_labels = []
			temp_true = correct_labels.data.cpu().numpy()
			for i in range(len(temp_true)):
				true_labels.append(int(temp_true[i]))

			true.extend(true_labels)

			if targetVocabulary.__len__() == 2:
				pred_output = pred_output.data.cpu().numpy()
				pred = []

				for i in range(len(pred_output)):
					if pred_output[i] >= 0.5:
						pred.append(1)
					else:
						pred.append(0)
				predicted.extend(pred)
			else:
				predicted.extend(pred_output.data.cpu().numpy().tolist())

		precision, recall, fscore, support = score(true, predicted, average="macro")

		print('precision: {}'.format(precision))
		print('recall: {}'.format(recall))
		print('fscore: {}'.format(fscore))
		print('support: {}'.format(support))

		from sklearn.metrics import classification_report as clsr
		scores = clsr(true, predicted)
		scores = list(map(lambda r: re.sub('\s\s+', '\t', r), scores.split("\n")))
		scores[-2] = '\t' + scores[-2]
		scores = '\n'.join(scores)
		print(scores)

		print("\n" * 1)


if __name__ == '__main__':
	main()
