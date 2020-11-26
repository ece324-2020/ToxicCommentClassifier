# this file takes the original train.csv and generates 4 csv files: train, valid, test, and overfit (for prototyping)
# it takes all the bad comments + equal amount of good comments
from matplotlib import pyplot as plt
import seaborn as sns
import re
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
good = train[(train["toxic"]==0)&(train["severe_toxic"]==0)&(train["obscene"]==0)&(train["threat"]==0)&(train["insult"]==0)&(train["identity_hate"]==0)]
# takes an equal amount of good comments to match the bad
sample_good = good.sample(n = 4593, random_state = seed, replace = False) #choose total of bad/6 # of good comments
# print(sample_good)
good_overfit = good.sample(n = 10, random_state = seed, replace = False)


# bad comments (opposite of isolating good)
bad = train[~((train["toxic"]==0)&(train["severe_toxic"]==0)&(train["obscene"]==0)&(train["threat"]==0)&(train["insult"]==0)&(train["identity_hate"]==0))]
bad = bad[~((bad["toxic"]==1)&(bad["severe_toxic"]==0)&(bad["obscene"]==0)&(bad["threat"]==0)&(bad["insult"]==0)&(bad["identity_hate"]==0))]
# print(bad["toxic"].value_counts(), #9628
#     bad["severe_toxic"].value_counts(),  #1595
#     bad["obscene"].value_counts(), #8449
#     bad["threat"].value_counts(), #478
#     bad["insult"].value_counts(), #7877
#     bad["identity_hate"].value_counts()) #1405

severe_toxic    = bad[bad["severe_toxic"]==1].sample(n = 4500, random_state = seed, replace = True)
threat          = bad[bad["threat"]==1].sample(n = 7000, random_state = seed, replace = True)
identity_hate   = bad[bad["identity_hate"]==1].sample(n = 4500, random_state = seed, replace = True)
threat1         = train[(train["toxic"]==0)&(train["obscene"]==0)&(train["threat"]==1)&(train["insult"]==0)]
# print(threat1.shape)
threat1         = threat1.sample(n = 500, random_state = seed, replace = True)
identity_hate1  = train[(train["toxic"]==0)&(train["obscene"]==0)&(train["insult"]==0)&(train["identity_hate"]==1)]
# print(identity_hate1.shape)
identity_hate1  = identity_hate1.sample(n = 500, random_state = seed, replace = True)

# print(severe_toxic.shape)
# print(threat.shape)
# print(identity_hate.shape)

bad = pd.concat([bad, severe_toxic, threat, identity_hate, threat1, identity_hate1])
bad_overfit = bad.sample(n = 60, random_state = seed, replace = False)
print(bad["toxic"].value_counts(), #24918
    bad["severe_toxic"].value_counts(),  #8718
    bad["obscene"].value_counts(), #20433
    bad["threat"].value_counts(), #8609
    bad["insult"].value_counts(), #19992
    bad["identity_hate"].value_counts()) #8715

#split train into train, valid, and test
good_train, good_temp = train_test_split(sample_good, test_size=0.2, random_state = seed)
good_valid, good_test = train_test_split(good_temp, test_size=0.5, random_state = seed)
bad_train, bad_temp = train_test_split(bad, test_size=0.2, random_state = seed)
bad_valid, bad_test = train_test_split(bad_temp, test_size=0.5, random_state = seed)

# stack good and bad as a single dataframe
train = pd.concat([good_train, bad_train])
valid = pd.concat([good_valid, bad_valid])
test = pd.concat([good_test, bad_test])
overfit = pd.concat([good_overfit, bad_overfit])
# print(overfit)

# shuffle and save to csv
train.sample(frac=1).to_csv("./processed_data/train.csv", index=False)
valid.sample(frac=1).to_csv("./processed_data/valid.csv", index=False)
test.sample(frac=1).to_csv("./processed_data/test.csv", index=False)
overfit.sample(frac=1).to_csv("./processed_data/overfit.csv", index=False)

