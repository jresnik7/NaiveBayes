# NaiveBayes
Naive Bayes text classification

# Summary
This is a program which uses Naive Bayes and bag of words representation to do text classification of IMDB movie reviews as either positive reviews or negative reviews. It takes a set of movie reviews split into test and training splits, with positive or negative labels and trains on the training split, and is then tested on the testing split to come up with an approximate accuracy for the algorithm. The bag of words representation is created with simple tokenization and the use of stopwords. Then, these words are fed into a Naive Bayes algorithm to determine their sentiment.

# Motivation and Sources
This project was originally completed in class 490A at the University of Massachusetts Amherst as part of the coursework for the class. It was completed by Jeffrey Resnik (https://orcid.org/0009-0005-7763-4620). There are two datasets, the large_movie_review_dataset which is taken from https://ai.stanford.edu/~amaas/data/sentiment/, and contains 25000 positive and 25000 negative reviews. The second dataset, txt_sentoken, contains only 1000 positive reviews and 1000 negeative reviews and therefore is much smaller, and is taken from https://www.cs.cornell.edu/people/pabo/movie-review-data/. This is specifically the 2.0 version of the dataset.
