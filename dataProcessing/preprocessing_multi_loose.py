# this file takes the original train.csv and generates 4 csv files: train, valid, test, and overfit (for prototyping)

import pandas as pd
from sklearn.model_selection import train_test_split
seed = 1

train = pd.read_csv("./data/train.csv")
# print(train.shape) #159570 rows

# find number of certain toxic comments (there is overlap between labels)
# print(train["toxic"].value_counts(), #15294
#     train["severe_toxic"].value_counts(),  #1595
#     train["obscene"].value_counts(), #8449
#     train["threat"].value_counts(), #478
#     train["insult"].value_counts(), #7877
#     train["identity_hate"].value_counts()) #1405

# good comments
good = train[(train["toxic"]==0)&(train["severe_toxic"]==0)&(train["obscene"]==0)&(train["threat"]==0)&(train["insult"]==0)&(train["identity_hate"]==0)]
sample_good = good.sample(n = 478, random_state = seed, replace = False) #choose 478 because the maximum for threat is 478, want to balance every class
good_overfit = good.sample(n = 10, random_state = seed, replace = False)




# bad comments 
toxic           = train[(train["toxic"]==1)&(train["severe_toxic"]==0)&(train["obscene"]==0)&(train["threat"]==0)&(train["insult"]==0)&(train["identity_hate"]==0)]
severe_toxic    = train[(train["toxic"]==0)&(train["severe_toxic"]==1)&(train["obscene"]==0)&(train["threat"]==0)&(train["insult"]==0)&(train["identity_hate"]==0)]
obscene         = train[(train["toxic"]==0)&(train["severe_toxic"]==0)&(train["obscene"]==1)&(train["threat"]==0)&(train["insult"]==0)&(train["identity_hate"]==0)]
threat          = train[(train["toxic"]==0)&(train["severe_toxic"]==0)&(train["obscene"]==0)&(train["threat"]==1)&(train["insult"]==0)&(train["identity_hate"]==0)]
insult          = train[(train["toxic"]==0)&(train["severe_toxic"]==0)&(train["obscene"]==0)&(train["threat"]==0)&(train["insult"]==1)&(train["identity_hate"]==0)]
identity_hate   = train[(train["toxic"]==0)&(train["severe_toxic"]==0)&(train["obscene"]==0)&(train["threat"]==0)&(train["insult"]==0)&(train["identity_hate"]==1)]
# print(toxic["toxic"].value_counts()) #5666 - pick from those
# print(severe_toxic["severe_toxic"].value_counts()) #0 - all 478 need to come from multi labelled
# print(obscene["obscene"].value_counts()) #317 - 161 from multi labelled
# print(threat["threat"].value_counts()) #22 - all 478 need to come from multi labelled (the constraint)
# print(insult["insult"].value_counts()) #301 - 177 from multi labelled
# print(identity_hate["identity_hate"].value_counts()) #54 - 424 from multi labelled

#custom selections based on results above
toxic           = toxic.sample(n = 478, random_state = seed, replace = False)
# print(toxic.shape) #make sure all classes are 478

severe_toxic    = train[(train["severe_toxic"]==1)].sample(n = 478, random_state = seed, replace = False)
severe_toxic    = severe_toxic.assign(toxic=0, obscene=0, threat=0, insult=0, identity_hate=0)
# print(severe_toxic.shape)

obscene2        = train[(train["obscene"]==1)].sample(n = 161, random_state = seed, replace = False)
obscene2        = obscene2.assign(toxic=0, severe_toxic=0, threat=0, insult=0, identity_hate=0)
obscene         = pd.concat([obscene, obscene2])
# print(obscene.shape)

threat          = train[(train["threat"]==1)].sample(n = 478, random_state = seed, replace = False)
threat          = threat.assign(toxic=0, severe_toxic=0, obscene=0, insult=0, identity_hate=0)
# print(threat.shape)

insult2         = train[(train["insult"]==1)].sample(n = 177, random_state = seed, replace = False)
insult2         = insult2.assign(toxic=0, severe_toxic=0, obscene=0, threat=0, identity_hate=0)
insult          = pd.concat([insult, insult2])
# print(insult.shape)

identity_hate2  = train[(train["identity_hate"]==1)].sample(n = 424, random_state = seed, replace = False)
identity_hate2  = identity_hate2.assign(toxic=0, severe_toxic=0, obscene=0, threat=0, insult=0)
identity_hate   = pd.concat([identity_hate, identity_hate2])
# print(identity_hate.shape)

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
train.sample(frac=1).to_csv("./multi_loose_data/train.csv", index=False)
valid.sample(frac=1).to_csv("./multi_loose_data/valid.csv", index=False)
test.sample(frac=1).to_csv("./multi_loose_data/test.csv", index=False)
overfit.sample(frac=1).to_csv("./multi_loose_data/overfit.csv", index=False)