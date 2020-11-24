# ToxicCommentClassifier

The goal of this project is to create a safer environment on social media. By detecting toxic comments on social media platforms, they can be more easily reported and removed. In the long term, this would allow people to better connect with each other in this increasingly digital world.

In the dataset given, there are 100000+ comments, and each comment is labelled in 6 categories: toxic, severe_toxic, obscene, threat, insult, and indentity hate. A comment labelled as 1,0,0,1,0,0 means toxic and threat, while 0,0,1,0,1,0 means obscene and insult. If all 6 categories are 0, then the comments is a "good" comment.

There are multiple datasets created for prototyping and testing, each catered towards a different model:
1. processed_data: unbalanced set of all bad comments, with equal amounts of good comments (16225 each). This is the first attempt at multi-label classification. Outputted with preprocessing.py.
2. binary_data: balanced set between good and obsence comments to test out binary classification (8449 each). Outputted with preprocessing_binary.py.
3. multi_strict_data: unbalanced set between all the bad comments that are labelled as a single category only, with an equal amount of good comments. Some bad comment categories have thousands of examples, while some have none. Outputted with preprocessing_multi_strict.py.
4. multi_loose_data: balanced set between all bad comment categories with 478 comments in each category. There are 478 good comments also, which is lower than the datasets before. Because the model gets 50% accuracyguesses all comments are good before, compared to ~14% now. This is generated for multi-class classification. Outputted with preprocessing_multi_loose.py.
5. future dataset coming with balanced multi-label classes.

After, the data can be fed into a globe embedding layer using processing.py and processing_binary.py. The former can be used for datasets 1, 3, and 4 (with different paths for data), while the latter is only for dataset 2. The piece of code is copied into the model files for integration. 

