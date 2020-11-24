# this file takes the original train.csv and generates 4 csv files: train, valid, test, and overfit (for prototyping)

import pandas as pd
from sklearn.model_selection import train_test_split
seed = 1

train = pd.read_csv("./data/train.csv")
# print(train) #159570 rows

# find number of good comments: 143345, this means bad comments = 16225
good = train[(train["toxic"]==0)&(train["severe_toxic"]==0)&(train["obscene"]==0)&(train["threat"]==0)&(train["insult"]==0)&(train["identity_hate"]==0)]
sample_good = good.sample(n = 1060, random_state = seed, replace = False) #choose same number of good vs bad comments 6360/6 = 1060
good_overfit = good.sample(n = 10, random_state = seed, replace = False)

# bad comments 
toxic           = train[(train["toxic"]==1)&(train["severe_toxic"]==0)&(train["obscene"]==0)&(train["threat"]==0)&(train["insult"]==0)&(train["identity_hate"]==0)]
severe_toxic    = train[(train["toxic"]==0)&(train["severe_toxic"]==1)&(train["obscene"]==0)&(train["threat"]==0)&(train["insult"]==0)&(train["identity_hate"]==0)]
obscene         = train[(train["toxic"]==0)&(train["severe_toxic"]==0)&(train["obscene"]==1)&(train["threat"]==0)&(train["insult"]==0)&(train["identity_hate"]==0)]
threat          = train[(train["toxic"]==0)&(train["severe_toxic"]==0)&(train["obscene"]==0)&(train["threat"]==1)&(train["insult"]==0)&(train["identity_hate"]==0)]
insult          = train[(train["toxic"]==0)&(train["severe_toxic"]==0)&(train["obscene"]==0)&(train["threat"]==0)&(train["insult"]==1)&(train["identity_hate"]==0)]
identity_hate   = train[(train["toxic"]==0)&(train["severe_toxic"]==0)&(train["obscene"]==0)&(train["threat"]==0)&(train["insult"]==0)&(train["identity_hate"]==1)]
# print(toxic["toxic"].value_counts()) #5666
# print(severe_toxic["severe_toxic"].value_counts()) #0
# print(obscene["obscene"].value_counts()) #317
# print(threat["threat"].value_counts()) #22
# print(insult["insult"].value_counts()) #301
# print(identity_hate["identity_hate"].value_counts()) #54

bad = pd.concat([toxic, severe_toxic, obscene, threat, insult, identity_hate])
bad_overfit = bad.sample(n = 10, random_state = seed, replace = False)
# print(bad.shape)

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

# # shuffle and save to csv
train.sample(frac=1).to_csv("./multi_strict_data/train.csv", index=False)
valid.sample(frac=1).to_csv("./multi_strict_data/valid.csv", index=False)
test.sample(frac=1).to_csv("./multi_strict_data/test.csv", index=False)
overfit.sample(frac=1).to_csv("./multi_strict_data/overfit.csv", index=False)