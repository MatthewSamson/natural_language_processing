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

# text_classification_gensimlstm.ipynb
Performing text classification by using a deep learning (RNN-LSTM) model. The project applies word embeddings developed from word2vec pretrained embeddtings. These embeddings are further trained to the current dataset followed by classification using a batch wise training method (mini-batch size is 1).

# post_processing_gensimlstm.ipynb
Repeating text classification by using the same deep learning model. In this case pretrained word embeddings are loaded and further trained on the current dataset. The embeddings are further processed to remove the mean, principle axes and the summation of the product of the eigenvectors, embeddings and the eigenvectors transposes. These embeddings are then applied into classiciation using a batch wise training method (mini-batch size is 1).

# init_trained_wvvectors
The pretrained word embeddings that are then trained on the input dataset. These embeddings are stored and are loaded into either the text_classification_gensimlstm.ipynb and the post_processing_gensimlstm.ipynb. These are also the vectors that are loaded and from which the post_processed_vectors1 embeddings will be developed (within the post_processing_gensimlstm.ipynb)

# post_processed_vectors1
The post processed vectors are stored in this file and are loaded into the post_processing_gensimlstm.ipynb, so there is no need to run the post processing function found in the post_processing_gensimlstm.ipynb
