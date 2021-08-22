from typing import ItemsView
import numpy as np

from util import accuracy
from hmm import HMM

# TODO:
def model_training(train_data, tags):
	"""
	Train HMM based on training data

	Inputs:
	- train_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
	- tags: (1*num_tags) a list of POS tags

	Returns:
	- model: an object of HMM class initialized with parameters(pi, A, B, obs_dict, state_dict) you calculated based on train_data
	"""

	# state_dict
	state_dict = dict()
	for idx, item in enumerate(tags):
		state_dict[item] = idx
	
	# print(state_dict)

	# obs_dict
	obs_dict = dict()
	pi_dict = dict()

	idx = 0
	for scentence in train_data:
		for i, item in enumerate(scentence.words):
			if item not in obs_dict:
				obs_dict[item] = idx
				idx += 1
	
	# print(obs_dict)

	# pi	
	for scentence in train_data:
		tags = scentence.tags
		if tags[0] not in pi_dict:
			pi_dict[tags[0]] = 1
		else:
			pi_dict[tags[0]] += 1

	pi = np.zeros(len(state_dict))

	count_first_tags = sum(pi_dict.values())

	for k, v in pi_dict.items():
		pi[state_dict[k]] = float(v)/float(count_first_tags)
	
	pi = np.asarray(pi)

	# print(pi)

	S = len(pi)
	L = len(obs_dict)

	# Transition matrix: A
	A = np.zeros([S, S])
	for scentence in train_data:
		tags = scentence.tags
		for i in range(len(tags)-1):
			s_dash = state_dict.get(tags[i])
			s = state_dict.get(tags[i+1])
			A[s_dash][s] += 1

	A = A/A.sum(axis=1, keepdims=True)
	
	# print(A)

	# Emission matrix: B
	B = np.zeros([S, L])

	for scentence in train_data:
		words = scentence.words
		tags = scentence.tags
		word_tag = list(map(lambda x, y:(x,y), words, tags))

		for w1, t1 in word_tag:
			ti = state_dict[t1]
			wi = obs_dict[w1]

			B[ti][wi] += 1
	
	B = B/B.sum(axis=1, keepdims=True)

	# print(B)

	return HMM(pi, A, B, obs_dict, state_dict)

# TODO:
def sentence_tagging(test_data, model, tags):
	"""
	Inputs:
	- test_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
	- model: an object of HMM class

	Returns:
	- tagging: (num_sentence*num_tagging) a 2D list of output tagging for each sentences on test_data
	"""
	tagging = []
	
	S = len(model.pi)
	L = len(model.obs_dict)

	for data in test_data:
		word_list = data.words

		# Check if there is a new observation in word_list.
		for w in word_list:
			if w not in model.obs_dict:
				model.obs_dict[w] = len(model.obs_dict)

				new_obs = np.array([np.repeat(10**-6, S)]).T
				model.B = np.hstack((model.B, new_obs))

		tagging.append(model.viterbi(word_list))

	return tagging