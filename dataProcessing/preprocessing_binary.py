# this file takes the original train.csv and generates 4 csv files: train, valid, test, and overfit (for prototyping)

import pandas as pd
from sklearn.model_selection import train_test_split
seed = 1

train = pd.read_csv("./data/train.csv")
# print(train) #159570 rows

# find number of certain toxic comments (there is overlap between labels)
# print(train["toxic"].value_counts(), #15294
#     train["severe_toxic"].value_counts(),  #1595
#     train["obscene"].value_counts(), #8449
#     train["threat"].value_counts(), #478
#     train["insult"].value_counts(), #7877
#     train["identity_hate"].value_counts()) #1405

# find number of good comments: 143345, this means bad comments = 16225
good = train[(train["toxic"]==0)&
    (train["severe_toxic"]==0)&
    (train["obscene"]==0)&
    (train["threat"]==0)&
    (train["insult"]==0)&
    (train["identity_hate"]==0)]
# print(good)
# takes an equal amount of good comments to match the bad
sample_good = good.sample(n = 8449, random_state = seed, replace = False) #choose same number of good vs bad comments
# print(sample_good)
good_overfit = good.sample(n = 10, random_state = seed, replace = False)


# bad comments (opposite of isolating good)
bad = train[(train["obscene"]==1)]
# print(bad["obscene"].value_counts())
# print(bad) 
bad_overfit = bad.sample(n = 10, random_state = seed, replace = False)

# #verify that bad is correctly isolated
# print(bad["toxic"].value_counts(), #15294
#     bad["severe_toxic"].value_counts(),  #1595
#     bad["obscene"].value_counts(), #8449
#     bad["threat"].value_counts(), #478
#     bad["insult"].value_counts(), #7877
#     bad["identity_hate"].value_counts()) #1405

#split train into train, valid, and test
good_train, good_temp = train_test_split(sample_good, test_size=0.2, random_state = seed)
good_valid, good_test = train_test_split(good_temp, test_size=0.5, random_state = seed)

bad_train, bad_temp = train_test_split(bad, test_size=0.2, random_state = seed)
bad_valid, bad_test = train_test_split(bad_temp, test_size=0.5, random_state = seed)

# print(good_test)

# stack good and bad as a single dataframe
train = pd.concat([good_train, bad_train])
valid = pd.concat([good_valid, bad_valid])
test = pd.concat([good_test, bad_test])
overfit = pd.concat([good_overfit, bad_overfit])
# print(overfit)

# shuffle and save to csv
train.sample(frac=1).to_csv("./binary_data/train.csv", index=False)
valid.sample(frac=1).to_csv("./binary_data/valid.csv", index=False)
test.sample(frac=1).to_csv("./binary_data/test.csv", index=False)
overfit.sample(frac=1).to_csv("./binary_data/overfit.csv", index=False)