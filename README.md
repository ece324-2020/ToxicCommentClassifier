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

## The Results ##
