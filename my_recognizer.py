import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
		""" Recognize test word sequences from word models set

	 :param models: dict of trained models
			 {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
	 :param test_set: SinglesData object
	 :return: (list, list)  as probabilities, guesses
			 both lists are ordered by the test set word_id
			 probabilities is a list of dictionaries where each key a word and value is Log Liklihood
					 [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
						{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
						]
			 guesses is a list of the best guess words ordered by the test set word_id
					 ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
	 """
		warnings.filterwarnings("ignore", category=DeprecationWarning)
		probabilities = []
		guesses = []
		# TODO implement the recognizer
		sequences = test_set.get_all_sequences()
		for i in range(test_set.num_items):
			temp = {}
			X, length = test_set.get_item_Xlengths(i)
			best_score = float('-Inf')
			best_guess = None
			# Calculate the score for each key word and its corresponding selected model
			for key, model in models.items():
				try:
					score = model.score(X, length)
					temp[key] = score
					if score > best_score:
						best_score = score
						best_guess = key
				except:
					temp[key] = float('-Inf')
			probabilities.append(temp)
			guesses.append(best_guess)
		return probabilities, guesses

