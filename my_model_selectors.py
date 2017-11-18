import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
	'''
	base class for model selection (strategy design pattern)
	'''

	def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
				 n_constant=3,
				 min_n_components=2, max_n_components=10,
				 random_state=14, verbose=False):
		self.words = all_word_sequences
		self.hwords = all_word_Xlengths
		self.sequences = all_word_sequences[this_word]
		self.X, self.lengths = all_word_Xlengths[this_word]
		self.this_word = this_word
		self.n_constant = n_constant
		self.min_n_components = min_n_components
		self.max_n_components = max_n_components
		self.random_state = random_state
		self.verbose = verbose

	def select(self):
		raise NotImplementedError

	def base_model(self, num_states):
		# with warnings.catch_warnings():
		warnings.filterwarnings("ignore", category=DeprecationWarning)
		# warnings.filterwarnings("ignore", category=RuntimeWarning)
		try:
			hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
									random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
			if self.verbose:
				print("model created for {} with {} states".format(self.this_word, num_states))
			return hmm_model
		except:
			if self.verbose:
				print("failure on {} with {} states".format(self.this_word, num_states))
			return None


class SelectorConstant(ModelSelector):
	""" select the model with value self.n_constant

	"""

	def select(self):
		""" select based on n_constant value

		:return: GaussianHMM object
		"""
		best_num_components = self.n_constant
		return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
	""" select the model with the lowest Bayesian Information Criterion(BIC) score

	http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
	Bayesian information criteria: BIC = -2 * logL + p * logN
	"""

	def select(self):
		""" select the best model for self.this_word based on
		BIC score for n between self.min_n_components and self.max_n_components

		:return: GaussianHMM object
		"""
		warnings.filterwarnings("ignore", category=DeprecationWarning)

		# TODO implement model selection based on BIC scores
		try:
			selected_model = None
			lowest_score = float('Inf')

			for n_components in range(self.min_n_components, self.max_n_components+1):
				model = self.base_model(n_components)
				logL = model.score(self.X, self.lengths)
				# Number of parameters increases quadratically with the number of states
				# plus (number of states - 1) * number of features
				# This may not be accurate
				p = n_components * n_components + (n_components-1) * len(self.X[0])
				logN = np.log(len(self.sequences))

				BIC_score = -2 * logL + p * logN

				# Update model if lower BIC_score is found
				if BIC_score < lowest_score:
					selected_model = model
					lowest_score = BIC_score

			return selected_model

		except:
			pass

		return self.base_model(self.n_constant)



class SelectorDIC(ModelSelector):
	""" select best model based on Discriminative Information Criterion

	Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
	Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
	http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
	DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
	"""

	def select(self):
		warnings.filterwarnings("ignore", category=DeprecationWarning)

		# TODO implement model selection based on DIC scores
		try:
			highest_score = float('-Inf')
			selected_model = None
			all_words = self.words.keys()

			for n_components in range(self.min_n_components, self.max_n_components + 1):
				model = self.base_model(n_components)
				this_word_score = model.score(self.X, self.lengths)
				other_words_score = 0
				for word in all_words:
					if word != self.this_word:
						word_X, word_lengths = self.hwords[word]
						other_words_score += model.score(word_X, word_lengths)

				score = this_word_score - other_words_score/(len(all_words) - 1)

				if score > highest_score:
					selected_model = model
					highest_score = score

			return selected_model

		except:
			pass

		return self.base_model(self.n_constant)

class SelectorCV(ModelSelector):
	''' select best model based on average log Likelihood of cross-validation folds

	'''

	def select(self):
		warnings.filterwarnings("ignore", category=DeprecationWarning)

		# TODO implement model selection using CV
		try:
			selected_model = None
			highest_score = float('-Inf')

			for n_components in range(self.min_n_components, self.max_n_components + 1):
				n_splits = min(3, len(self.sequences))
				split_method = KFold(n_splits)
				score = 0

				# Calculate the score for each test fold
				for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
					self.X, self.lengths = combine_sequences(cv_train_idx, self.sequences)
					X_test, lengths_test = combine_sequences(cv_test_idx, self.sequences)
					model = self.base_model(n_components)
					score += model.score(X_test, lengths_test)
				
				# Calculate the average score
				score = score / n_splits

				# Update the model if current model is best so far
				if score > highest_score:
					selected_model = model
					highest_score = score

			return selected_model
		except:
			pass

		return self.base_model(self.n_constant)

