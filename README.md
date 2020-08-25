# natural_language_processing
My work in Natural Language Processing, involving work in my NLP masters' course and after graduation projects.

# Token.java, Scanner.java, Online.java, Offline.java
A set of java classes that parsed text files, breaks them into tokens and then uses data structures (trees and lists) to build document term and term frequency matrices.

# CharLSTM.ipynb
Building a Long Short Term Memory Recurrent Neural Network (LSTM - RNN) to understand a piece of text and generate similar text, essentially a character generator.

# sentiment_cleaning_pipeline.ipynb
Building a pipeline that goes over the numerous cleaning procedures involved in cleaning textual data and also creating a dataset of "balanced by each class" data.

# sentiment_feature_classicalML.ipynb
Using the cleaned data to generate features based on document term and term frequency matrices followed by applying classical ML classifiers to perform sentiment analysis on both balanced and unbalanced data.

# sentiment_feature_neuralNet.ipynb
Using the cleaned data to generate features based on document term and term frequency matrices followed by building deep learning models to perform sentiment analysis on both the balanced and unbalanced data.

# tweets_Word2Vec.ipynb
Uploading data from twitter based on flight reviews to perform sentiment analysis on how good or bad the flight trip is. The data from twitter is loaded and cleaned up followed by building a custom word2vec model that gives each unique word a distributed represented vector trained from their respective context via the continuous bag of words model.
