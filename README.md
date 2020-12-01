# ToxicCommentClassifier #

The goal of this project is to create a safer environment on social media. By detecting toxic comments on social media platforms, they can be more easily reported and removed. In the long term, this would allow people to better connect with each other in this increasingly digital world.

## The Data ##

In the dataset given, there are 150000+ comments, and each comment is labelled in 6 categories: toxic, severe_toxic, obscene, threat, insult, and identity hate. A comment labelled as 1,0,0,1,0,0 means toxic and threat, while 0,0,1,0,1,0 means obscene and insult. If all 6 categories are 0, then the comments is a "good" comment.

An overwhelming majority (89%) of the data contains good comments, so in each dataset, the number of good and bad comments are balanced. Each dataset also contains 4 files: train, valid, test, and overfit.

There are multiple datasets created for prototyping and testing, each catered towards a different model prototype:

1. Data: original and cleaned dataset from kaggle
2. Datasets that start with binary: for 6 binary classifiers (6 sub folders, each for a different class)
3. Datasets that start with processed_data: for multi-label classification
4. Datasets that start with multi: for multi-class classification

The suffixes have different meanings too:
1. Datasets that end with aug: with augmented data for the 3 minority classes: severe_toxic, threat, and indentity hate
1. Datasets that end with v2: same as data_aug but with more data for threats
3. loose vs strict: loose has repeated comments, while strict does not

In the end, two datasets were used: processed_data_aug_v2 and binary_data^6_aug_v2 for the two different approaches in the final model

After, the data can be fed into a globe embedding layer using processing.py, processing_binary.py, or processing_binary^6.py base on the data, and they are converted into batcheed vocab vectors.

## The Model ##
Here's a link to the google drive folder that contains all the models and vocabs. Models in folder that contains the word "Aug" are the best performed models.
https://drive.google.com/drive/folders/1nSGNEDN2mS2csaRHbWHTHipvGcxatwP7?usp=sharing 
### Baseline Model ###
Each word is tokenized, and then converted to a GloVe embedding vector. The average vector for the sentence is passed through a fully connected layer which outputs a prediction.

### Main Model ###
The main model we used were called "CNN_LSTM.py"
It has 2 Conv2d layer, 2 maxpool layers, 1 LSTM layer and 1 fully connected layer

There are three approaches we experimented with to train the model:
#### apporach 1 ####
General dataset with 6 classes --> 1 CNN_LSTM classifier --> 6 outputs

#### apporach 2 ####
General dataset with 6 classes --> 6 CNN_LSTM classifier --> each gives 1 outputs

#### apporach 3 (best approach) ####
6 sub-datasets each with 1 class --> 6 CNN_LSTM classifier --> each gives 1 outputs

## The Results ##
#### Baseline Model ####
train loss: 0.617, train acc: 0.797, test loss: 0.619, test acc 0.794

#### apporach 1 ####
train loss: 0.612, train acc: 0.812,  test loss: 0.618, test acc 0.806 

#### apporach 2 ####
train loss: 0.612,train acc: 0.900,  test loss: 0.537, test acc 0.879 

#### apporach 3 (best approach) ####
train loss: 0.510,train acc: 0.986,  test loss: 0.528, test acc 0.953  


