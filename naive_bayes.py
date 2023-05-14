import matplotlib.pyplot as plt
import nltk
from nltk import *
from nltk.corpus import stopwords
import glob
from collections import defaultdict, Counter
import math
import os
from datetime import datetime
import csv

nltk.download('stopwords')

# Global class labels.
POS_LABEL = 'pos'
NEG_LABEL = 'neg'


def main():
    print("Hello, this may take a little bit...")
    PATH_TO_DATA = 'large_movie_review_dataset'  # set this variable to point to the location of the IMDB corpus
    POS_LABEL = 'pos'
    NEG_LABEL = 'neg'
    TRAIN_DIR = os.path.join(PATH_TO_DATA, "train")
    TEST_DIR = os.path.join(PATH_TO_DATA, "test")
    word_counts = Counter()

    for label in [POS_LABEL, NEG_LABEL]:
        for directory in [TRAIN_DIR, TEST_DIR]:
            for fn in glob.glob(directory + "/" + label + "/*txt"):
                doc = open(fn, 'r', encoding='utf8') # Open the file with UTF-8 encoding
                word_count = tokenize_doc_and_more(doc.read())
                for word in word_count:
                    word_counts[word] += word_count[word]

    print("Finished Word Count")
    nb = NaiveBayes(PATH_TO_DATA, tokenizer=tokenize_doc_and_more)
    nb.train_model()
    print("Finished Training. Evaluating...")
    print("Final accuracy: " + str(nb.evaluate_classifier_accuracy(10.0)) + "%")
    print("Creating Graph and CSV output file...")
    plot_likelihood_create_csv(nb, word_counts)
    print("Complete!")


def tokenize_doc_and_more(doc): 
    """
    Return some representation of a document.
    Uses stopwords to remove most common words that do not contribute meaning
    Uses a Regular Expression tokenizer to better tokenize the words.
    """
    stopset = set(stopwords.words('english'))
    bow = defaultdict(float)
    tokenizer = RegexpTokenizer('\s+', gaps=True)
    tokens = tokenizer.tokenize(doc)
    lowered_tokens = map(lambda t: t.lower(), tokens)
    for token in lowered_tokens:
        if(token not in stopset):
            bow[token] += 1.0
    return bow


def tokenize_doc(doc):
    """
    Tokenize a document and return its bag-of-words representation.
    doc - a string representing a document.
    returns a dictionary mapping each word to the number of times it appears in doc.
    """
    bow = defaultdict(float)
    tokens = doc.split()
    lowered_tokens = map(lambda t: t.lower(), tokens)
    for token in lowered_tokens:
        bow[token] += 1.0
    return dict(bow)


def n_word_types(word_counts):
    """
    return a count of all word types in the corpus
    using information from word_counts
    """
    return len(word_counts)


def n_word_tokens(word_counts):
    """
    return a count of all word tokens in the corpus
    using information from word_counts
    """
    return sum(word_counts.values())


def plot_likelihood_create_csv(nb, word_counts):
    x = []  # x-axis
    y = []  # y-axis
    header = ["Word", "Occurrences", "Likelihood Ratio"]
    date_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = 'vocab_statistics_large_' + date_time + '.csv'
    with open(filename, 'w', newline='', encoding='utf8') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for word in word_counts:
            likelihood = nb.likelihood_ratio(word, word_counts[word])
            count = word_counts[word]
            x.append(count)
            y.append(likelihood)
            writer.writerow([str(word), count, likelihood])
    file.close()
    plt.scatter(x, y, s=10.0, color='b', marker='.')
    plt.xlabel("Occurrences")
    plt.ylabel("Likelihood Ratio")
    plt.title("Occurrences and Likelihood Ratios of Vocabulary")
    plt.savefig("Likelihood_ratio_graph_large_" + date_time + ".png")


class NaiveBayes:
    """A Naive Bayes model for text classification."""

    def __init__(self, path_to_data, tokenizer):
        # Vocabulary is a set that stores every word seen in the training data
        self.vocab = set()
        self.path_to_data = path_to_data
        self.tokenize_doc = tokenizer
        self.train_dir = os.path.join(path_to_data, "train")
        self.test_dir = os.path.join(path_to_data, "test")
        # class_total_doc_counts is a dictionary that maps a class (i.e., pos/neg) to
        # the number of documents in the training set of that class
        self.class_total_doc_counts = { POS_LABEL: 0.0,
                                        NEG_LABEL: 0.0 }

        # class_total_word_counts is a dictionary that maps a class (i.e., pos/neg) to
        # the number of words in the training set in documents of that class
        self.class_total_word_counts = { POS_LABEL: 0.0,
                                         NEG_LABEL: 0.0 }

        # class_word_counts is a dictionary of dictionaries. It maps a class (i.e.,
        # pos/neg) to a dictionary of word counts. For example:
        #    self.class_word_counts[POS_LABEL]['awesome']
        # stores the number of times the word 'awesome' appears in documents
        # of the positive class in the training documents.
        self.class_word_counts = { POS_LABEL: defaultdict(float),
                                   NEG_LABEL: defaultdict(float) }

    def train_model(self):
        """
        This function processes the entire training set using the global PATH
        variable above.  It makes use of the tokenize_doc and update_model
        functions you will implement.
        """

        pos_path = os.path.join(self.train_dir, POS_LABEL)
        neg_path = os.path.join(self.train_dir, NEG_LABEL)
        for (p, label) in [ (pos_path, POS_LABEL), (neg_path, NEG_LABEL) ]:
            for f in os.listdir(p):
                with open(os.path.join(p,f),'r', encoding = 'utf8') as doc:
                    content = doc.read()
                    self.tokenize_and_update_model(content, label)
        self.report_statistics_after_training()

    def report_statistics_after_training(self):
        """
        Report a number of statistics after training.
        """

        print("REPORTING CORPUS STATISTICS")
        print("NUMBER OF DOCUMENTS IN POSITIVE CLASS:", self.class_total_doc_counts[POS_LABEL])
        print("NUMBER OF DOCUMENTS IN NEGATIVE CLASS:", self.class_total_doc_counts[NEG_LABEL])
        print("NUMBER OF TOKENS IN POSITIVE CLASS:", self.class_total_word_counts[POS_LABEL])
        print("NUMBER OF TOKENS IN NEGATIVE CLASS:", self.class_total_word_counts[NEG_LABEL])
        print("VOCABULARY SIZE: NUMBER OF UNIQUE WORDTYPES IN TRAINING CORPUS:", len(self.vocab))

    def update_model(self, bow, label):
        """
        Update internal statistics given a document represented as a bag-of-words
        bow - a map from words to their counts
        label - the class of the document whose bag-of-words representation was input
        This function doesn't return anything but should update a number of internal
        statistics. Specifically, it updates:
          - the internal map the counts, per class, how many times each word was
            seen (self.class_word_counts)
          - the number of words seen for each label (self.class_total_word_counts)
          - the vocabulary seen so far (self.vocab)
          - the number of documents seen of each label (self.class_total_doc_counts)
        """
        for word in bow:
            self.class_word_counts[label][word] += bow[word]
            self.class_total_word_counts[label] += bow[word]
            self.vocab.add(word)

        self.class_total_doc_counts[label] += 1

    def tokenize_and_update_model(self, doc, label):
        """
        Tokenizes a document doc and updates internal count statistics.
        doc - a string representing a document.
        label - the sentiment of the document (either positive or negative)
        
        Make sure when tokenizing to lower case all tokens!
        """
        self.update_model(self.tokenize_doc(doc), label)

    def top_n(self, label, n):
        """
        Returns the most frequent n tokens for documents with class 'label'.
        """
        return sorted(self.class_word_counts[label].items(), key=lambda x : x[1], reverse=True)[:n]

    def p_word_given_label(self, word, label):
        """
        Returns the probability of word given label
        according to this NB model.
        """
        return self.class_word_counts[label][word]/self.class_total_word_counts[label]

    def p_word_given_label_and_alpha(self, word, label, alpha):
        """
        Returns the probability of word given label wrt psuedo counts.
        alpha - pseudocount parameter
        """
        return (self.class_word_counts[label][word]+alpha)/(self.class_total_word_counts[label] + (len(self.vocab) * alpha))

    def log_likelihood(self, bow, label, alpha):
        """
        Computes the log likelihood of a set of words give a label and pseudocount.
        bow - a bag of words (i.e., a tokenized document)
        label - either the positive or negative label
        alpha - float; pseudocount parameter
        """
        x = []
        retval = 0

        for word in bow:
            x.append(math.log(self.p_word_given_label_and_alpha(word, label, alpha)))

        for i in range(0, len(x)):
            retval += x[i]

        return retval

    def log_prior(self, label):
        """
        Returns the log prior of a document having the class 'label'.
        """
        return math.log(self.class_total_doc_counts[label]/(self.class_total_doc_counts[POS_LABEL] + self.class_total_doc_counts[NEG_LABEL]))

    def unnormalized_log_posterior(self, bow, label, alpha):
        """
        Computes the unnormalized log posterior (of doc being of class 'label').
        bow - a bag of words (i.e., a tokenized document)
        """
        return (self.log_likelihood(bow, label, alpha) + self.log_prior(label))

    def classify(self, bow, alpha):
        """
        Compares the unnormalized log posterior for doc for both the positive
        and negative classes and returns the either POS_LABEL or NEG_LABEL
        (depending on which resulted in the higher unnormalized log posterior)
        bow - a bag of words (i.e., a tokenized document)
        """
        if(self.unnormalized_log_posterior(bow, POS_LABEL, alpha) >= self.unnormalized_log_posterior(bow, NEG_LABEL, alpha)):
            return POS_LABEL
        else:
            return NEG_LABEL

    def likelihood_ratio(self, word, alpha):
        """
        Returns the ratio of P(word|pos) to P(word|neg).
        """
        return self.p_word_given_label_and_alpha(word, POS_LABEL, alpha)/self.p_word_given_label_and_alpha(word, NEG_LABEL, alpha)

    def largest_likelihood_ratio(self, alpha):
        """
        returns the largest likelihood ratio for the words in the vocabulary
        """
        max_ret = 0
        max_word = ""
        for word in self.vocab:
            val = self.likelihood_ratio(word, alpha)
            if(max_ret < val):
                max_word = word
                max_ret = val

        return max_word + " " + str(max_ret)

    def evaluate_classifier_accuracy(self, alpha):
        """
        alpha - pseudocount parameter.
        This function should go through the test data, classify each instance and
        compute the accuracy of the classifier (the fraction of classifications
        the classifier gets right.
        """
        correct = 0.0
        total = 0.0

        pos_path = os.path.join(self.test_dir, POS_LABEL)
        neg_path = os.path.join(self.test_dir, NEG_LABEL)
        for (p, label) in [ (pos_path, POS_LABEL), (neg_path, NEG_LABEL) ]:
            for f in os.listdir(p):
                with open(os.path.join(p,f),'r', encoding = 'utf8') as doc:
                    content = doc.read()
                    bow = self.tokenize_doc(content)
                    if self.classify(bow, alpha) == label:
                        correct += 1.0
                    total += 1.0
        return 100 * correct / total


if __name__ == '__main__':
    main()
    
