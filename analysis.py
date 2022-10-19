# Danny Zhang
# dz37
# COMP 182 Spring 2021 - Homework 8, Problem 2

# You can import any standard library, as well as Numpy and Matplotlib.
# You can use helper functions from provided.py, and autograder.py,
# but they have to be copied over here.

# Your code here...

import math
import numpy
from collections import *
import matplotlib.pyplot as plt
import pylab

#Provided functions
class HMM:
    """
    Simple class to represent a Hidden Markov Model.
    """
    def __init__(self, order, initial_distribution, emission_matrix, transition_matrix):
        self.order = order
        self.initial_distribution = initial_distribution
        self.emission_matrix = emission_matrix
        self.transition_matrix = transition_matrix

def read_pos_file(filename):
    """
    Parses an input tagged text file.
    Input:
    filename --- the file to parse
    Returns:
    The file represented as a list of tuples, where each tuple
    is of the form (word, POS-tag).
    A list of unique words found in the file.
    A list of unique POS tags found in the file.
    """
    file_representation = []
    unique_words = set()
    unique_tags = set()
    f = open(str(filename), "r")
    for line in f:
        if len(line) < 2 or len(line.split("/")) != 2:
            continue
        word = line.split("/")[0].replace(" ", "").replace("\t", "").strip()
        tag = line.split("/")[1].replace(" ", "").replace("\t", "").strip()
        file_representation.append( (word, tag) )
        unique_words.add(word)
        unique_tags.add(tag)
    f.close()
    return file_representation, unique_words, unique_tags

def bigram_viterbi(hmm, sentence):
    """
    Run the Viterbi algorithm to tag a sentence assuming a bigram HMM model.
    Inputs:
      hmm --- the HMM to use to predict the POS of the words in the sentence.
      sentence ---  a list of words.
    Returns:
      A list of tuples where each tuple contains a word in the
      sentence and its predicted corresponding POS.
    """
    # Initialization
    viterbi = defaultdict(lambda: defaultdict(int))
    backpointer = defaultdict(lambda: defaultdict(int))
    unique_tags = set(hmm.initial_distribution.keys()).union(set(hmm.transition_matrix.keys()))
    for tag in unique_tags:
        if (hmm.initial_distribution[tag] != 0) and (hmm.emission_matrix[tag][sentence[0]] != 0):
            viterbi[tag][0] = math.log(hmm.initial_distribution[tag]) + math.log(hmm.emission_matrix[tag][sentence[0]])
        else:
            viterbi[tag][0] = -1 * float('inf')

    # Dynamic programming.
    for t in range(1, len(sentence)):
        backpointer["No_Path"][t] = "No_Path"
        for s in unique_tags:
            max_value = -1 * float('inf')
            max_state = None
            for s_prime in unique_tags:
                val1= viterbi[s_prime][t-1]
                val2 = -1 * float('inf')
                if hmm.transition_matrix[s_prime][s] != 0:
                    val2 = math.log(hmm.transition_matrix[s_prime][s])
                curr_value = val1 + val2
                if curr_value > max_value:
                    max_value = curr_value
                    max_state = s_prime
            val3 = -1 * float('inf')
            if hmm.emission_matrix[s][sentence[t]] != 0:
                val3 = math.log(hmm.emission_matrix[s][sentence[t]])
            viterbi[s][t] = max_value + val3
            if max_state == None:
                backpointer[s][t] = "No_Path"
            else:
                backpointer[s][t] = max_state
    for ut in unique_tags:
        string = ""
        for i in range(0, len(sentence)):
            if (viterbi[ut][i] != float("-inf")):
                string += str(int(viterbi[ut][i])) + "\t"
            else:
                string += str(viterbi[ut][i]) + "\t"

    # Termination
    max_value = -1 * float('inf')
    last_state = None
    final_time = len(sentence) - 1
    for s_prime in unique_tags:
        if viterbi[s_prime][final_time] > max_value:
            max_value = viterbi[s_prime][final_time]
            last_state = s_prime
    if last_state == None:
        last_state = "No_Path"

    # Traceback
    tagged_sentence = []
    tagged_sentence.append((sentence[len(sentence)-1], last_state))
    for i in range(len(sentence)-2, -1, -1):
        next_tag = tagged_sentence[-1][1]
        curr_tag = backpointer[next_tag][i+1]
        tagged_sentence.append((sentence[i], curr_tag))
    tagged_sentence.reverse()
    return tagged_sentence

#Analysis: written auxiliary functions
def read_pos_file_percent(filename, percentage):
    """
    Parses an input tagged text file.
    Input:
    filename --- the file to parse
    percentage -- percentage that represents how much of the filename to consider
    Returns:
    The file represented as a list of tuples, where each tuple
    is of the form (word, POS-tag).
    A list of unique words found in the file.
    A list of unique POS tags found in the file.
    """
    file_representation = []
    unique_words = set()
    unique_tags = set()
    f = open(str(filename), "r")

    num_lines = 0
    for line in f:
        if len(line) < 2 or len(line.split("/")) != 2:
            continue
        num_lines += 1

    f.close()

    cutoff = num_lines * percentage
    f = open(str(filename), "r")
    track = 0
    for line in f:
        if track < cutoff:
            if len(line) < 2 or len(line.split("/")) != 2:
                continue
            word = line.split("/")[0].replace(" ", "").replace("\t", "").strip()
            tag = line.split("/")[1].replace(" ", "").replace("\t", "").strip()
            file_representation.append( (word, tag) )
            unique_words.add(word)
            unique_tags.add(tag)
            track += 1
        else:
            break
    f.close()
    return file_representation, unique_words, unique_tags

#TEST CASES FOR READ_POS_FILE_PERCENT:
# print(read_pos_file_percent('training.txt', 0.03))
# print(read_pos_file_percent('training.txt', 0.07))

def read_file(filename):
    """
    Parses an input tagged text file.
    Input:
    filename --- the file to parse
    Returns:
    The file represented as a list of words
    """
    file_representation = []
    f = open(str(filename), "r")
    for line in f:
        word = line.split()
        file_representation.append(word)
    f.close()
    return file_representation[0]

#TEST CASES FOR READ_FILE:
# print(read_file('training.txt'))
# print(read_file('testdata_untagged.txt'))

def accuracy_calc(word_tag, filename):
    """
    Computes the accuracy of the results from the HMM by comparing the output with the tagged data from the test data.

    Input: List of tuples that represent the word and its tag and the file name of the file containing the tagged test data.

    Output: A float that indicates the accuracy of the HMM.
    """
    tags = []
    total = 0
    accuracy = 0
    f = open(str(filename), "r")
    for line in f:
        if len(line) < 2 or len(line.split("/")) != 2:
            continue
        tag = line.split("/")[1].replace(" ", "").replace("\t", "").strip()
        tags.append(tag)
        total += 1
    for idx in range(len(word_tag)):
        if word_tag[idx][1] == tags[idx]:
            accuracy += float(1/total)
    f.close()
    return accuracy

#TEST CASES FOR ACCURACY_CALC:
# print(accuracy_calc([('The', 'DT'), ('New', 'NNP'), ('Deal', 'NNP')], 'testdata_tagged.txt'))
# print(accuracy_calc([('The', 'DT'), ('New', 'NNP'), ('Deal', 'NNP'), ('was', 'VBD'), ('a', 'DT'), ('series', 'NN'), ('of', 'IN'), ('domestic', 'JJ'), ('programs', 'NNS')], 'testdata_tagged.txt'))

#Written code
def compute_counts(training_data: list, order: int) -> tuple:
	"""
	Computes the counts of the number unique tokens, the number of times a word is tagged, the number of times a tag appears, the number of times a 2-tag-sequence occurs,
	and the number of times a 3-tag-sequence occurs.

	Input: List that represents the training data and an integer that represents the order.

	Output: Returns a tuple of number of unique tokens, and returns 3 dictionaries that represent the number of times a word is tagged, the number of times a tag appears, and
    the number of times a 2-tag-sequence occurs if the order is 2 or returns 4 dictionaries that represent the number of times a word is tagged, the number of times a tag appears, 
    the number of times a 2-tag-sequence occurs, and the number of times a 3-tag-sequence occurs if the order is 3.
	"""  
	num_tokens = len(training_data)

	dict1 = defaultdict(lambda: defaultdict(int)) #Number of times a word is tagged
	for ele in training_data:
		dict1[ele[1]][ele[0]] += 1

	dict2 = defaultdict(int) #Number of times a tag appears
	for ele in training_data:
		dict2[ele[1]] += 1

	dict3 = defaultdict(lambda: defaultdict(int)) #Number of times a 2-tag-sequence occurs
	for idx in range(1, len(training_data)):
		dict3[training_data[idx-1][1]][training_data[idx][1]] += 1

	dict4 = defaultdict(lambda: defaultdict(lambda: defaultdict(int))) #Number of times a 3-tag-sequence occurs
	for idx in range(2, len(training_data)):
		dict4[training_data[idx-2][1]][training_data[idx-1][1]][training_data[idx][1]] += 1

	if order == 2:
		return (num_tokens, dict1, dict2, dict3)
	elif order == 3:
		return (num_tokens, dict1, dict2, dict3, dict4)

#TEST CASES FOR COMPUTE_COUNTS:
# print(compute_counts([('The', 'DT'), ('homework', 'NN'), ('was', 'VBD'), ('quite', 'RB'), ('tough', 'JJ'), ('.', '.')], 2))
# print(compute_counts([('There', 'EX'), ('was', 'VBD'), ('a', 'DT'), ('Great', 'NNP'), ('Depression', 'NNP'), ('.', '.')], 3))
 
def compute_initial_distribution(training_data: list, order: int) -> dict:
    """
    Computes the initial distribution by counting the tags that are at the beginning of sentences.

    Input: List that represents the training data and an integer that represents the order.

    Output: Dictionary that represents the probability that a tag occurs at the beginning of a sentence in the training data.
    """
    if order == 2:
        dict_pi_1 = defaultdict(int)
        total = 1 #Account for the beginning of the training data
        for idx in range(len(training_data) - 1):
            if training_data[idx][1] == ".": #Number of periods
                total += 1
        dict_pi_1[training_data[0][1]] = float(1/total)        
        for idx in range(len(training_data) - 1):
            if training_data[idx][1] == ".":
                dict_pi_1[training_data[idx+1][1]] += float(1/total) #Add 1/total
        return dict_pi_1

    elif order == 3:
        dict_pi_2 = defaultdict(lambda: defaultdict(int))
        total = 1 #Account for the beginning of the training data
        for idx in range(len(training_data) - 2):
            if training_data[idx][1] == ".": #Number of periods
                total += 1
        dict_pi_2[training_data[0][1]][training_data[1][1]] = float(1/total)           
        for idx in range(len(training_data) - 2):
            if training_data[idx][1] == ".":
                dict_pi_2[training_data[idx+1][1]][training_data[idx+2][1]] += float(1/total) #Add 1/total 
        return dict_pi_2

#TEST CASES FOR COMPUTE_INITIAL_DISTRIBUTION:
# print(compute_initial_distribution([('The', 'DT'), ('assignment', 'NN'), ('was', 'VBD'), ('quite', 'RB'), ('easy', 'JJ'), ('.', '.')], 2))
# print(compute_initial_distribution([('There', 'EX'), ('is', 'VB'), ('a', 'DT'), ('Great', 'NNP'), ('Depression', 'NNP'), ('.', '.')], 3))

def compute_emission_probabilities(unique_words: list, unique_tags: list, W: dict, C: dict) -> dict:
    """
    Computes the emission probabilities of the training data.

    Input: List that represents the unique words, list that represents the unique tags, dictionary that represents the number of times a word is tagged with 
    a particular tag, dictionary represents the number of times a tag occurs.

    Output: Dictionary that represents the emission matrix.
    """
    emit = defaultdict(lambda: defaultdict(int))
    for tag in unique_tags: #Iterate through the unique tags
        for word in unique_words: #Iterate through the unique words 
            emit[tag][word] = float(W[tag][word]) / float(C[tag]) #Eq. 7 in the description document
    return emit

#TEST CASES FOR COMPUTE_EMISSION_PROBABILITIES:
# print(compute_emission_probabilities(['The', 'homework' ,'was', 'quite', 'tough', '.'], ['DT', 'NN', 'VBD', 'RB', 'JJ', '.'], compute_counts([('The', 'DT'), ('homework', 'NN'), ('was', 'VBD'), ('quite', 'RB'), ('tough', 'JJ'), ('.', '.')], 2)[1], compute_counts([('The', 'DT'), ('homework', 'NN'), ('was', 'VBD'), ('quite', 'RB'), ('tough', 'JJ'), ('.', '.')], 2)[2]))
# print(compute_emission_probabilities(['There', 'a', 'was', 'Great', 'Depression', '.'], ['EX', 'DT', 'VBD', 'NNP', '.'], compute_counts([('There', 'EX'), ('was', 'VBD'), ('a', 'DT'), ('Great', 'NNP'), ('Depression', 'NNP'), ('.', '.')], 3)[1], compute_counts([('There', 'EX'), ('was', 'VBD'), ('a', 'DT'), ('Great', 'NNP'), ('Depression', 'NNP'), ('.', '.')], 3)[2]))

def compute_lambdas(unique_tags: list, num_tokens: int, C1: dict, C2: dict, C3: dict, order: int) -> list:
    """
    Computes the lambdas, or the coefficients that are necessary for the computation of the transition matrix.

    Input: List that represents the unique tags, integer that represents the number of unique tokens, dictionary that represents the number of times that a tag occurs,
    dictionary that represents the number of times a 2-tag-sequence occurs, dictionary that represents the number of times a 3-tag-sequence occurs, and an integer that 
    represents the order.

    Output: List that represents the three lambdas that are calculated.
    """
    if order == 2: #Computations from Algorithm 1 in the description document
        lambda_0 = float(0)
        lambda_1 = float(0)
        for t_1 in unique_tags: #Iterate through unique tags
            for t in unique_tags: #Iterate through unique tags
                if C2[t_1][t] > 0:
                    a_0 = float((C1[t] - 1)) / float(num_tokens)
                    if (C1[t_1] - 1) != 0:
                        a_1 = float((C2[t_1][t] - 1)) / float((C1[t_1] - 1))
                    else:
                        a_1 = 0
                    i = numpy.argmax([a_0, a_1])
                    if i == 0:
                        lambda_0 += C2[t_1][t]
                    elif i == 1:
                        lambda_1 += C2[t_1][t]
        (lambda_0, lambda_1) = (float(lambda_0) / float((lambda_0 + lambda_1)), float(lambda_1) / float((lambda_0 + lambda_1)))
        return [(lambda_0), (lambda_1), float(0)]
    elif order == 3: #Computations from Algorithm 1 in the description document
        lambda_0 = float(0)
        lambda_1 = float(0)
        lambda_2 = float(0)
        for t_2 in unique_tags: #Iterate through unique tags
            for t_1 in unique_tags: #Iterate through unique tags
                for t in unique_tags:
                    if C3[t_2][t_1][t] > 0:
                        a_0 = float((C1[t] - 1) / num_tokens)
                        if (C1[t_1] - 1) != 0:
                            a_1 = float((C2[t_1][t] - 1) / (C1[t_1] - 1))
                        else:
                            a_1 = 0
                        if (C2[t_2][t_1] - 1) != 0:
                            a_2 = float((C3[t_2][t_1][t] - 1) / (C2[t_2][t_1] - 1))
                        else: 
                            a_2 = 0
                        i = numpy.argmax([a_0, a_1, a_2])
                        if i == 0:
                            lambda_0 += C3[t_2][t_1][t]
                        elif i == 1:
                            lambda_1 += C3[t_2][t_1][t]
                        elif i == 2:
                            lambda_2 += C3[t_2][t_1][t]
        (lambda_0, lambda_1, lambda_2) = (float(lambda_0) / float((lambda_0 + lambda_1 + lambda_2)), float(lambda_1) / float((lambda_0 + lambda_1 + lambda_2)), float(lambda_2) / float((lambda_0 + lambda_1 + lambda_2)))
        return [(lambda_0), (lambda_1), (lambda_2)]

#TEST CASES FOR COMPUTE_LAMBDAS:
# c = compute_counts([('The', 'DT'), ('homework', 'NN'), ('was', 'VBD'), ('quite', 'RB'), ('tough', 'JJ'), ('.', '.')], 2)
# print(compute_lambdas(['DT', 'NN', 'VBD', 'RB', 'JJ', '.'], 6, c[2], c[3], {}, 2))
# c1 = compute_counts([('There', 'EX'), ('was', 'VBD'), ('a', 'DT'), ('Great', 'NNP'), ('Depression', 'NNP'), ('.', '.')], 3)
# print(compute_lambdas(['EX', 'DT', 'VBD', 'NNP', '.'], 6, c1[2], c1[3], c1[4], 3))

def build_hmm(training_data: list, unique_tags: list, unique_words: list, order: int, use_smoothing: bool):
	"""
	Builds the hidden markov model.

	Input: List that represents the training data, list that represents the unique tags, list that represents the unique words, integer that represents the order, and
	boolean that indicates whether smoothing is used or not.

	Output: HMM with populated attributes.
	"""
	compute_count_output = compute_counts(training_data, 3) #Call compute_counts

	if order == 2:
		if use_smoothing == True:
			(lambda_0, lambda_1, lambda_2) = compute_lambdas(unique_tags, compute_count_output[0], compute_count_output[2], compute_count_output[3], {}, order) #Call compute_lambdas if smoothing is used
		else:
			lambda_0 = 0
			lambda_1 = 1
			lambda_2 = 0
	elif order == 3:
		if use_smoothing == True:
			(lambda_0, lambda_1, lambda_2) = compute_lambdas(unique_tags, compute_count_output[0], compute_count_output[2], compute_count_output[3], compute_count_output[4], order) #Call compute_lambdas if smoothing is used
		else:
			lambda_0 = 0
			lambda_1 = 0
			lambda_2 = 1

	if order == 2: #Creating transition matrix using Eq. 8 from the description document
		transition = defaultdict(lambda: defaultdict(int))
		for tag1 in unique_tags:
			for tag2 in unique_tags:
				if compute_count_output[2][tag1] != 0:
					transition[tag1][tag2] = (float(lambda_1) * float(compute_count_output[3][tag1][tag2]/compute_count_output[2][tag1])) + (float(lambda_0) * float(compute_count_output[2][tag2]/compute_count_output[0]))
				else:
					transition[tag1][tag2] = 0
	elif order == 3: #Creating transition matrix using Eq. 9 from the description document
		transition = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
		for tag1 in unique_tags:
			for tag2 in unique_tags:
				for tag3 in unique_tags:
					if compute_count_output[3][tag1][tag2] != 0 and compute_count_output[2][tag2] != 0:
						transition[tag1][tag2][tag3] = (lambda_2 * (compute_count_output[4][tag1][tag2][tag3]/compute_count_output[3][tag1][tag2])) + (lambda_1 * (compute_count_output[3][tag2][tag3]/compute_count_output[2][tag2])) + (lambda_0 * (compute_count_output[2][tag3]/compute_count_output[0]))
					else:
						transition[tag1][tag2][tag3] = 0

	hmm =  HMM(order, compute_initial_distribution(training_data, order), compute_emission_probabilities(unique_words, unique_tags, compute_count_output[1], compute_count_output[2]), transition)

	return hmm

#TEST CASES FOR BUILD_HMM:
# hmm1 = build_hmm([('The', 'DT'), ('homework', 'NN'), ('was', 'VBD'), ('quite', 'RB'), ('tough', 'JJ'), ('.', '.')], ['DT', 'NN', 'VBD', 'RB', 'JJ', '.'], ['The','homework', 'was', 'quite', 'tough', '.'], 3, True)
# hmm2 = build_hmm([('There', 'EX'), ('was', 'VBD'), ('a', 'DT'), ('Great', 'NNP'), ('Depression', 'NNP'), ('.', '.')], ['EX', 'DT', 'VBD', 'NNP', '.'], ['There', 'was', 'a', 'Great', 'Depression', '.'], 3, False)

def trigram_viterbi(hmm, sentence: list) -> list:
	"""
	Run the Viterbi algorithm to tag a sentence assuming a trigram HMM model.

	Inputs: A HMM to use to predict the POS of the words in the sentence and a sentence represented by a list of words.

	Output: A list of tuples where each tuple contains a word in the sentence and its predicted corresponding POS.
	"""
	# Initialization
	viterbi = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
	backpointer = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
	unique_tags = set(hmm.initial_distribution.keys()).union(set(hmm.transition_matrix.keys()))
	for tag1 in unique_tags:
		for tag2 in unique_tags:
			if (hmm.initial_distribution[tag1][tag2] != 0) and (hmm.emission_matrix[tag1][sentence[0]] != 0) and (hmm.emission_matrix[tag2][sentence[1]] != 0):
				viterbi[tag1][tag2][1] = math.log(hmm.initial_distribution[tag1][tag2]) + math.log(hmm.emission_matrix[tag1][sentence[0]]) + math.log(hmm.emission_matrix[tag2][sentence[1]])
			else:
				viterbi[tag1][tag2][1] = -1 * float('inf')

	# Dynamic programming.
	for t in range(2, len(sentence)):
		backpointer["No_Path"]["No_Path"][t] = "No_Path"      
		for s in unique_tags:
			for s_prime in unique_tags:
				max_value = -1 * float('inf')
				max_state = None
				for s_doubleprime in unique_tags:
					val1 = viterbi[s_doubleprime][s_prime][t-1]
					val2 = -1 * float('inf')
					if hmm.transition_matrix[s_doubleprime][s_prime][s] != 0:
						val2 = math.log(hmm.transition_matrix[s_doubleprime][s_prime][s])
					curr_value = val1 + val2
					if curr_value > max_value:
						max_value = curr_value
						max_state = s_doubleprime
				val3 = -1 * float('inf')
				if hmm.emission_matrix[s][sentence[t]] != 0:
					val3 = math.log(hmm.emission_matrix[s][sentence[t]])
				viterbi[s_prime][s][t] = max_value + val3
				if max_state == None:
					backpointer[s_prime][s][t] = "No_Path"
				else:
					backpointer[s_prime][s][t] = max_state

	# Termination
	last_max_value = -1 * float('inf')
	last_state = None
	second_last_state = None
	final_time = len(sentence) - 1
	for s_prime in unique_tags:
		for s_doubleprime in unique_tags:
			if viterbi[s_doubleprime][s_prime][final_time] > last_max_value:
				last_max_value = viterbi[s_doubleprime][s_prime][final_time]
				last_state = s_prime
				second_last_state = s_doubleprime
	if last_state == None:
		last_state = "No_Path"
	if second_last_state == None:
		second_last_state = "No_Path"
	
	tagged_sentence = []
	tagged_sentence.append((sentence[len(sentence)-1], last_state))
	tagged_sentence.append((sentence[len(sentence)-2], second_last_state))
	for i in range(len(sentence)-3, -1, -1):
		next_tag_1 = tagged_sentence[-1][1]
		next_tag_2 = tagged_sentence[-2][1]
		curr_tag = backpointer[next_tag_1][next_tag_2][i+2]
		tagged_sentence.append((sentence[i], curr_tag))
	tagged_sentence.reverse()

	return tagged_sentence

#TEST CASES FOR TRIGRAM_VITERBI:
# print(trigram_viterbi(hmm1, ['That', 'was', 'a', 'great', 'experience', '.']))
# print(trigram_viterbi(hmm2, ['That', 'was', 'not', 'a', 'great', 'experience', '.']))

def update_hmm(hmm, test_data):
    """
    Update the emission matrix of a HMM to account for words that are not encountered in the training data.

    Input: A HMM and a sentence represented by a list of words.

    Output: None
    """
    total_update_per_tag = defaultdict(float)
    unique_words = set()
    unique_tags = set()

    #Find all unique tags and unique words
    for tag in hmm.emission_matrix:
        unique_tags.add(tag)
        for word in hmm.emission_matrix[tag]:
            unique_words.add(word)

    #Assign epsilon to every word that is not in the emission matrix
    for word in test_data:
        if word not in unique_words:
            for tag in unique_tags:
                hmm.emission_matrix[tag][word] += float(0.00001)

    #Adjust emission probabilities for all other nonzero emissions       
    for tag in hmm.emission_matrix:
        for word in hmm.emission_matrix[tag]:
            if hmm.emission_matrix[tag][word] != 0: #necessary
                hmm.emission_matrix[tag][word] += float(0.00001)

    #Normalize by the sum
    for tag in hmm.emission_matrix:
        count = float(0)
        for word in hmm.emission_matrix[tag]:
            count += hmm.emission_matrix[tag][word]
        total_update_per_tag[tag] = count

    for tag in hmm.emission_matrix:
        for word in hmm.emission_matrix[tag]:
            hmm.emission_matrix[tag][word] /= total_update_per_tag[tag]

#TEST CASES FOR UPDATE_HMM:
# update_hmm(hmm1, read_file('testdata_untagged.txt'))
# update_hmm(hmm2, read_file('testdata_untagged.txt'))

#Provided from Homework 4 to plot the curves
def show():
    """
    Do not use this function unless you have trouble with figures.

    It may be necessary to call this function after drawing/plotting
    all figures.  If so, it should only be called once at the end.

    Arguments:
    None

    Returns:
    None
    """
    plt.show()

def _dict2lists(data):
    """
    Convert a dictionary into a list of keys and values, sorted by
    key.  

    Arguments:
    data -- dictionary

    Returns:
    A tuple of two lists: the first is the keys, the second is the values
    """
    xvals = list(data.keys())
    xvals.sort()
    yvals = []
    for x in xvals:
        yvals.append(data[x])
    return xvals, yvals

def _plot_dict_line(d, label=None):
    """
    Plot data in the dictionary d on the current plot as a line.

    Arguments:
    d     -- dictionary
    label -- optional legend label

    Returns:
    None
    """
    xvals, yvals = _dict2lists(d)
    if label:
        pylab.plot(xvals, yvals, label=label)
    else:
        pylab.plot(xvals, yvals)

def plot_lines(data, title, xlabel, ylabel, labels=None, filename=None):
    """
    Plot a line graph with the provided data.

    Arguments: 
    data     -- a list of dictionaries, each of which will be plotted 
                as a line with the keys on the x axis and the values on
                the y axis.
    title    -- title label for the plot
    xlabel   -- x axis label for the plot
    ylabel   -- y axis label for the plot
    labels   -- optional list of strings that will be used for a legend
                this list must correspond to the data list
    filename -- optional name of file to which plot will be
                saved (in png format)

    Returns:
    None
    """
    ### Check that the data is a list
    if not isinstance(data, list):
        msg = "data must be a list, not {0}".format(type(data).__name__)
        raise TypeError(msg)

    ### Create a new figure
    fig = pylab.figure()

    ### Plot the data
    if labels:
        mylabels = labels[:]
        for _ in range(len(data)-len(labels)):
            mylabels.append("")
        for d, l in zip(data, mylabels):
            _plot_dict_line(d, l)
        # Add legend
        pylab.legend(loc='best')
        gca = pylab.gca()
        legend = gca.get_legend()
        pylab.setp(legend.get_texts(), fontsize='medium')
    else:
        for d in data:
            _plot_dict_line(d)

    ### Set the lower y limit to 0 or the lowest number in the values
    mins = [min(l.values()) for l in data]
    ymin = min(0, min(mins))
    pylab.ylim(ymin=ymin)

    ### Label the plot
    pylab.title(title)
    pylab.xlabel(xlabel)
    pylab.ylabel(ylabel)

    ### Draw grid lines
    pylab.grid(True)

    ### Show the plot
    fig.show()

    ### Save to file
    if filename:
        pylab.savefig(filename)

#Experiments
test_data = read_file('testdata_untagged.txt')
train1 = read_pos_file_percent('training.txt', 0.01)
train2 = read_pos_file_percent('training.txt', 0.05)
train3 = read_pos_file_percent('training.txt', 0.1)
train4 = read_pos_file_percent('training.txt', 0.25)
train5 = read_pos_file_percent('training.txt', 0.5)
train6 = read_pos_file_percent('training.txt', 0.75)
train7 = read_pos_file('training.txt')

# #Experiment 1
# hmm_1_1 = build_hmm(train1[0], train1[2], train1[1], 2, False)
# update_hmm(hmm_1_1, read_file('testdata_untagged.txt'))
# bigram_1_1 = []
# sentence = []
# pos = 0
# while pos < len(test_data):
#     if test_data[pos] != '.':
#         sentence.append(test_data[pos])
#     else:
#         sentence.append('.')
#         bigram_1_1.extend(bigram_viterbi(hmm_1_1, sentence))
#         sentence = []
#     pos += 1

# hmm_1_2 = build_hmm(train2[0], train2[2], train2[1], 2, False)
# update_hmm(hmm_1_2, read_file('testdata_untagged.txt'))
# bigram_1_2 = []
# sentence = []
# pos = 0
# while pos < len(test_data):
#     if test_data[pos] != '.':
#         sentence.append(test_data[pos])
#     else:
#         sentence.append('.')
#         bigram_1_2.extend(bigram_viterbi(hmm_1_2, sentence))
#         sentence = []
#     pos += 1

# hmm_1_3 = build_hmm(train3[0], train3[2], train3[1], 2, False)
# update_hmm(hmm_1_3, read_file('testdata_untagged.txt'))
# bigram_1_3 = []
# sentence = []
# pos = 0
# while pos < len(test_data):
#     if test_data[pos] != '.':
#         sentence.append(test_data[pos])
#     else:
#         sentence.append('.')
#         bigram_1_3.extend(bigram_viterbi(hmm_1_3, sentence))
#         sentence = []
#     pos += 1

# hmm_1_4 = build_hmm(train4[0], train4[2], train4[1], 2, False)
# update_hmm(hmm_1_4, read_file('testdata_untagged.txt'))
# bigram_1_4 = []
# sentence = []
# pos = 0
# while pos < len(test_data):
#     if test_data[pos] != '.':
#         sentence.append(test_data[pos])
#     else:
#         sentence.append('.')
#         bigram_1_4.extend(bigram_viterbi(hmm_1_4, sentence))
#         sentence = []
#     pos += 1

# hmm_1_5 = build_hmm(train5[0], train5[2], train5[1], 2, False)
# update_hmm(hmm_1_5, read_file('testdata_untagged.txt'))
# bigram_1_5 = []
# sentence = []
# pos = 0
# while pos < len(test_data):
#     if test_data[pos] != '.':
#         sentence.append(test_data[pos])
#     else:
#         sentence.append('.')
#         bigram_1_5.extend(bigram_viterbi(hmm_1_5, sentence))
#         sentence = []
#     pos += 1

# hmm_1_6 = build_hmm(train6[0], train6[2], train6[1], 2, False)
# update_hmm(hmm_1_6, read_file('testdata_untagged.txt'))
# bigram_1_6 = []
# sentence = []
# pos = 0
# while pos < len(test_data):
#     if test_data[pos] != '.':
#         sentence.append(test_data[pos])
#     else:
#         sentence.append('.')
#         bigram_1_6.extend(bigram_viterbi(hmm_1_6, sentence))
#         sentence = []
#     pos += 1

# hmm_1_7 = build_hmm(train7[0], train7[2], train7[1], 2, False)
# update_hmm(hmm_1_7, read_file('testdata_untagged.txt'))
# bigram_1_7 = []
# sentence = []
# pos = 0
# while pos < len(test_data):
#     if test_data[pos] != '.':
#         sentence.append(test_data[pos])
#     else:
#         sentence.append('.')
#         bigram_1_7.extend(bigram_viterbi(hmm_1_7, sentence))
#         sentence = []
#     pos += 1

# dict_exp_1 = {0.01: accuracy_calc(bigram_1_1, 'testdata_tagged.txt'), 0.05: accuracy_calc(bigram_1_2, 'testdata_tagged.txt'), 0.1: accuracy_calc(bigram_1_3, 'testdata_tagged.txt'), 
# 0.25: accuracy_calc(bigram_1_4, 'testdata_tagged.txt'), 0.5: accuracy_calc(bigram_1_5, 'testdata_tagged.txt'), 0.75: accuracy_calc(bigram_1_6, 'testdata_tagged.txt'), 1: accuracy_calc(bigram_1_7, 'testdata_tagged.txt')}
# # print(bigram_1_1)
# # print("ACCURACY: ", accuracy_calc(bigram_1_1, 'testdata_tagged.txt'))

# #Experiment 2
# hmm_2_1 = build_hmm(train1[0], train1[2], train1[1], 3, False)
# update_hmm(hmm_2_1, read_file('testdata_untagged.txt'))
# trigram_2_1 = []
# sentence = []
# pos = 0
# while pos < len(test_data):
#     if test_data[pos] != '.':
#         sentence.append(test_data[pos])
#     else:
#         sentence.append('.')
#         trigram_2_1.extend(trigram_viterbi(hmm_2_1, sentence))
#         sentence = []
#     pos += 1

# hmm_2_2 = build_hmm(train2[0], train2[2], train2[1], 3, False)
# update_hmm(hmm_2_2, read_file('testdata_untagged.txt'))
# trigram_2_2 = []
# sentence = []
# pos = 0
# while pos < len(test_data):
#     if test_data[pos] != '.':
#         sentence.append(test_data[pos])
#     else:
#         sentence.append('.')
#         trigram_2_2.extend(trigram_viterbi(hmm_2_2, sentence))
#         sentence = []
#     pos += 1

# hmm_2_3 = build_hmm(train3[0], train3[2], train3[1], 3, False)
# update_hmm(hmm_2_3, read_file('testdata_untagged.txt'))
# trigram_2_3 = []
# sentence = []
# pos = 0
# while pos < len(test_data):
#     if test_data[pos] != '.':
#         sentence.append(test_data[pos])
#     else:
#         sentence.append('.')
#         trigram_2_3.extend(trigram_viterbi(hmm_2_3, sentence))
#         sentence = []
#     pos += 1

# hmm_2_4 = build_hmm(train4[0], train4[2], train4[1], 3, False)
# update_hmm(hmm_2_4, read_file('testdata_untagged.txt'))
# trigram_2_4 = []
# sentence = []
# pos = 0
# while pos < len(test_data):
#     if test_data[pos] != '.':
#         sentence.append(test_data[pos])
#     else:
#         sentence.append('.')
#         trigram_2_4.extend(trigram_viterbi(hmm_2_4, sentence))
#         sentence = []
#     pos += 1

# hmm_2_5 = build_hmm(train5[0], train5[2], train5[1], 3, False)
# update_hmm(hmm_2_5, read_file('testdata_untagged.txt'))
# trigram_2_5 = []
# sentence = []
# pos = 0
# while pos < len(test_data):
#     if test_data[pos] != '.':
#         sentence.append(test_data[pos])
#     else:
#         sentence.append('.')
#         trigram_2_5.extend(trigram_viterbi(hmm_2_5, sentence))
#         sentence = []
#     pos += 1

# hmm_2_6 = build_hmm(train6[0], train6[2], train6[1], 3, False)
# update_hmm(hmm_2_6, read_file('testdata_untagged.txt'))
# trigram_2_6 = []
# sentence = []
# pos = 0
# while pos < len(test_data):
#     if test_data[pos] != '.':
#         sentence.append(test_data[pos])
#     else:
#         sentence.append('.')
#         trigram_2_6.extend(trigram_viterbi(hmm_2_6, sentence))
#         sentence = []
#     pos += 1

# hmm_2_7 = build_hmm(train7[0], train7[2], train7[1], 3, False)
# update_hmm(hmm_2_7, read_file('testdata_untagged.txt'))
# trigram_2_7 = []
# sentence = []
# pos = 0
# while pos < len(test_data):
#     if test_data[pos] != '.':
#         sentence.append(test_data[pos])
#     else:
#         sentence.append('.')
#         trigram_2_7.extend(trigram_viterbi(hmm_2_7, sentence))
#         sentence = []
#     pos += 1

# dict_exp_2 = {0.01: accuracy_calc(trigram_2_1, 'testdata_tagged.txt'), 0.05: accuracy_calc(trigram_2_2, 'testdata_tagged.txt'), 0.1: accuracy_calc(trigram_2_3, 'testdata_tagged.txt'), 
# 0.25: accuracy_calc(trigram_2_4, 'testdata_tagged.txt'), 0.5: accuracy_calc(trigram_2_5, 'testdata_tagged.txt'), 0.75: accuracy_calc(trigram_2_6, 'testdata_tagged.txt'), 1: accuracy_calc(trigram_2_7, 'testdata_tagged.txt')}
# # print(trigram)
# # print(accuracy_calc(trigram_2_7, 'testdata_tagged.txt'))

# #Experiment 3
# hmm_3_1 = build_hmm(train1[0], train1[2], train1[1], 2, True)
# update_hmm(hmm_3_1, read_file('testdata_untagged.txt'))
# test_data = read_file('testdata_untagged.txt')
# bigram_3_1 = []
# sentence = []
# pos = 0
# while pos < len(test_data):
#     if test_data[pos] != '.':
#         sentence.append(test_data[pos])
#     else:
#         sentence.append('.')
#         bigram_3_1.extend(bigram_viterbi(hmm_3_1, sentence))
#         sentence = []
#     pos += 1

# hmm_3_2 = build_hmm(train2[0], train2[2], train2[1], 2, True)
# update_hmm(hmm_3_2, read_file('testdata_untagged.txt'))
# test_data = read_file('testdata_untagged.txt')
# bigram_3_2 = []
# sentence = []
# pos = 0
# while pos < len(test_data):
#     if test_data[pos] != '.':
#         sentence.append(test_data[pos])
#     else:
#         sentence.append('.')
#         bigram_3_2.extend(bigram_viterbi(hmm_3_2, sentence))
#         sentence = []
#     pos += 1

# hmm_3_3 = build_hmm(train3[0], train3[2], train3[1], 2, True)
# update_hmm(hmm_3_3, read_file('testdata_untagged.txt'))
# test_data = read_file('testdata_untagged.txt')
# bigram_3_3 = []
# sentence = []
# pos = 0
# while pos < len(test_data):
#     if test_data[pos] != '.':
#         sentence.append(test_data[pos])
#     else:
#         sentence.append('.')
#         bigram_3_3.extend(bigram_viterbi(hmm_3_3, sentence))
#         sentence = []
#     pos += 1

# hmm_3_4 = build_hmm(train4[0], train4[2], train4[1], 2, True)
# update_hmm(hmm_3_4, read_file('testdata_untagged.txt'))
# test_data = read_file('testdata_untagged.txt')
# bigram_3_4 = []
# sentence = []
# pos = 0
# while pos < len(test_data):
#     if test_data[pos] != '.':
#         sentence.append(test_data[pos])
#     else:
#         sentence.append('.')
#         bigram_3_4.extend(bigram_viterbi(hmm_3_4, sentence))
#         sentence = []
#     pos += 1

# hmm_3_5 = build_hmm(train5[0], train5[2], train5[1], 2, True)
# update_hmm(hmm_3_5, read_file('testdata_untagged.txt'))
# test_data = read_file('testdata_untagged.txt')
# bigram_3_5 = []
# sentence = []
# pos = 0
# while pos < len(test_data):
#     if test_data[pos] != '.':
#         sentence.append(test_data[pos])
#     else:
#         sentence.append('.')
#         bigram_3_5.extend(bigram_viterbi(hmm_3_5, sentence))
#         sentence = []
#     pos += 1

# hmm_3_6 = build_hmm(train6[0], train6[2], train6[1], 2, True)
# update_hmm(hmm_3_6, read_file('testdata_untagged.txt'))
# test_data = read_file('testdata_untagged.txt')
# bigram_3_6 = []
# sentence = []
# pos = 0
# while pos < len(test_data):
#     if test_data[pos] != '.':
#         sentence.append(test_data[pos])
#     else:
#         sentence.append('.')
#         bigram_3_6.extend(bigram_viterbi(hmm_3_6, sentence))
#         sentence = []
#     pos += 1

# hmm_3_7 = build_hmm(train7[0], train7[2], train7[1], 2, True)
# update_hmm(hmm_3_7, read_file('testdata_untagged.txt'))
# test_data = read_file('testdata_untagged.txt')
# bigram_3_7 = []
# sentence = []
# pos = 0
# while pos < len(test_data):
#     if test_data[pos] != '.':
#         sentence.append(test_data[pos])
#     else:
#         sentence.append('.')
#         bigram_3_7.extend(bigram_viterbi(hmm_3_7, sentence))
#         sentence = []
#     pos += 1

# dict_exp_3 = {0.01: accuracy_calc(bigram_3_1, 'testdata_tagged.txt'), 0.05: accuracy_calc(bigram_3_2, 'testdata_tagged.txt'), 0.1: accuracy_calc(bigram_3_3, 'testdata_tagged.txt'), 
# 0.25: accuracy_calc(bigram_3_4, 'testdata_tagged.txt'), 0.5: accuracy_calc(bigram_3_5, 'testdata_tagged.txt'), 0.75: accuracy_calc(bigram_3_6, 'testdata_tagged.txt'), 1: accuracy_calc(bigram_3_7, 'testdata_tagged.txt')}
# # print(bigram_3_7)
# # print(accuracy_calc(bigram_3_7, 'testdata_tagged.txt'))

# #Experiment 4
# hmm_4_1 = build_hmm(train1[0], train1[2], train1[1], 3, True)
# update_hmm(hmm_4_1, read_file('testdata_untagged.txt'))
# trigram_4_1 = []
# sentence = []
# pos = 0
# while pos < len(test_data):
#     if test_data[pos] != '.':
#         sentence.append(test_data[pos])
#     else:
#         sentence.append('.')
#         trigram_4_1.extend(trigram_viterbi(hmm_4_1, sentence))
#         sentence = []
#     pos += 1

# hmm_4_2 = build_hmm(train2[0], train2[2], train2[1], 3, True)
# update_hmm(hmm_4_2, read_file('testdata_untagged.txt'))
# trigram_4_2 = []
# sentence = []
# pos = 0
# while pos < len(test_data):
#     if test_data[pos] != '.':
#         sentence.append(test_data[pos])
#     else:
#         sentence.append('.')
#         trigram_4_2.extend(trigram_viterbi(hmm_4_2, sentence))
#         sentence = []
#     pos += 1

# hmm_4_3 = build_hmm(train3[0], train3[2], train3[1], 3, True)
# update_hmm(hmm_4_3, read_file('testdata_untagged.txt'))
# trigram_4_3 = []
# sentence = []
# pos = 0
# while pos < len(test_data):
#     if test_data[pos] != '.':
#         sentence.append(test_data[pos])
#     else:
#         sentence.append('.')
#         trigram_4_3.extend(trigram_viterbi(hmm_4_3, sentence))
#         sentence = []
#     pos += 1

# hmm_4_4 = build_hmm(train4[0], train4[2], train4[1], 3, True)
# update_hmm(hmm_4_4, read_file('testdata_untagged.txt'))
# trigram_4_4 = []
# sentence = []
# pos = 0
# while pos < len(test_data):
#     if test_data[pos] != '.':
#         sentence.append(test_data[pos])
#     else:
#         sentence.append('.')
#         trigram_4_4.extend(trigram_viterbi(hmm_4_4, sentence))
#         sentence = []
#     pos += 1

# hmm_4_5 = build_hmm(train5[0], train5[2], train5[1], 3, True)
# update_hmm(hmm_4_5, read_file('testdata_untagged.txt'))
# trigram_4_5 = []
# sentence = []
# pos = 0
# while pos < len(test_data):
#     if test_data[pos] != '.':
#         sentence.append(test_data[pos])
#     else:
#         sentence.append('.')
#         trigram_4_5.extend(trigram_viterbi(hmm_4_5, sentence))
#         sentence = []
#     pos += 1

# hmm_4_6 = build_hmm(train6[0], train6[2], train6[1], 3, True)
# update_hmm(hmm_4_6, read_file('testdata_untagged.txt'))
# trigram_4_6 = []
# sentence = []
# pos = 0
# while pos < len(test_data):
#     if test_data[pos] != '.':
#         sentence.append(test_data[pos])
#     else:
#         sentence.append('.')
#         trigram_4_6.extend(trigram_viterbi(hmm_4_6, sentence))
#         sentence = []
#     pos += 1

hmm_4_7 = build_hmm(train7[0], train7[2], train7[1], 3, True)
update_hmm(hmm_4_7, read_file('testdata_untagged.txt'))
trigram_4_7 = []
sentence = []
pos = 0
while pos < len(test_data):
    if test_data[pos] != '.':
        sentence.append(test_data[pos])
    else:
        sentence.append('.')
        trigram_4_7.extend(trigram_viterbi(hmm_4_7, sentence))
        sentence = []
    pos += 1
# dict_exp_4 = {0.01: accuracy_calc(trigram_4_1, 'testdata_tagged.txt'), 0.05: accuracy_calc(trigram_4_2, 'testdata_tagged.txt'), 0.1: accuracy_calc(trigram_4_3, 'testdata_tagged.txt'), 
# 0.25: accuracy_calc(trigram_4_4, 'testdata_tagged.txt'), 0.5: accuracy_calc(trigram_4_5, 'testdata_tagged.txt'), 0.75: accuracy_calc(trigram_4_6, 'testdata_tagged.txt'), 1: accuracy_calc(trigram_4_7, 'testdata_tagged.txt')}
print(trigram_4_7)
print(accuracy_calc(trigram_4_7, 'testdata_tagged.txt'))

# print("EXPERIMENT 1:", dict_exp_1)
# print("EXPERIMENT 2:", dict_exp_2)
# print("EXPERIMENT 3:", dict_exp_3)
# print("EXPERIMENT 4:", dict_exp_4)

# plot_lines([dict_exp_1, dict_exp_2, dict_exp_3, dict_exp_4], "Percent of Training Data VS Accuracy", "Percent of Training Data", "Accuracy", 
#             ["Bigram, no smoothing", "Trigram, no smoothing", "Bigram, smoothing", "Trigram, smoothing"])
# show()

