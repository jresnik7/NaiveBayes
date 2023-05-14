# NaiveBayes
Naive Bayes text classification

# Summary
This is a program which uses Naive Bayes and bag of words representation to do text classification of IMDB movie reviews as either positive reviews or negative reviews. It takes a set of movie reviews split into test and training splits, with positive or negative labels and trains on the training split, and is then tested on the testing split to come up with an approximate accuracy for the algorithm. The bag of words representation is created with simple tokenization and the use of stopwords. Then, these words are fed into a Naive Bayes algorithm to determine their sentiment.

# Motivation and Sources
This project was originally completed in class 490A at the University of Massachusetts Amherst as part of the coursework for the class. It was completed by Jeffrey Resnik (https://orcid.org/0009-0005-7763-4620). There are two datasets, the large_movie_review_dataset which is taken from https://ai.stanford.edu/~amaas/data/sentiment/, and contains 25000 positive and 25000 negative reviews. The second dataset, txt_sentoken, contains only 1000 positive reviews and 1000 negeative reviews and therefore is much smaller, and is taken from https://www.cs.cornell.edu/people/pabo/movie-review-data/. This is specifically the 2.0 version of the dataset. Credit to Brendon O'Connor and course staff for the outline of the methods in the source code. However, the content of those methods is written by me.

# Running the Code
This projects expects Python 3.9 or higher. I have not extensively tested different versions of python, and have only run versions 3.9 and 3.11, and therefore I can only recommend this software. All of the import statements are used, so please make sure all the libraries are downloaded, and that the stopwords are also downloaded, as otherwise this could cause issues. However, the program should download the stopwords automatically when run. Note that the dataset to be used should be downloaded and the path to the dataset must be set in the code to make sure the program can find the dataset. Once this is done, the program should run smoothly.

# Results
There are some unused methods in the Naive Bayes class, which can be used to get some interesting statistics. For example, there is a method to retrieve the top n words with the highest likelihood ratio. I suggest that if you do not understand the math behind the Naive Bayes algorithm that it is worth understanding before proceeding, as much of the program will likely be confusing without this understanding. The code will produce 2 outputs. A .csv file which contains all words in the vocabulary, along with the number of occurrences of each word as well as the likelihood ratio of that word, and the graph of occurrences and likelihood ratio for all words in the vocabulary. The accuracy of the classifier for the larger dataset is 85.132%, while for the smaller dataset it is 77.2%.

# Further Comments
This source code is licensed under the MIT license, and the produced datasets are licensed under the CC BY license.
