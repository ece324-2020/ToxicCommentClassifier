# This file takes in the raw data and exaimines it + cleans it
from matplotlib import pyplot as plt
import seaborn as sns
import re
import pandas as pd
seed = 1

train = pd.read_csv("./data/train_old.csv")
cols_target = ['obscene','insult','toxic','severe_toxic','identity_hate','threat']

# Look at some examples of the data
print(train.sample(5))

# look at the overall value properties
print(train.describe())

# From before, some categories have very low mean, which implies most comments are good
# Find percentage of good comments
unlabelled_in_all = train[(train['toxic']!=1) & (train['severe_toxic']!=1) & (train['obscene']!=1) & (train['threat']!=1) & (train['insult']!=1) & (train['identity_hate']!=1)]
print('Percentage of good comments is ', len(unlabelled_in_all)/len(train)*100)

# Check for any null comments
no_comment = train[train['comment_text'].isnull()]
print('Number of null comments is ', len(no_comment))

# let's see the total rows in train, test data and the numbers for the various categories
print('Total rows in test is {}'.format(len(train)))
print(train[cols_target].sum())

# Next, let's examine the correlations among the target variables.
data = train[cols_target]
colormap = plt.cm.plasma
plt.figure(figsize=(7,7))
plt.title('Correlation of features & targets',y=1.05,size=14)
sns.heatmap(data.astype(float).corr(),linewidths=0.1,vmax=1.0,square=True,cmap=colormap,
           linecolor='white',annot=True)
plt.show()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip(' ')
    return text

# clean the comment_text in train
train['comment_text'] = train['comment_text'].map(lambda com : clean_text(com))
train.to_csv("./data/train.csv", index=False)

# Let's look at the character length for the rows in the training data and record these
# At the end because do not want to add another column to output data
train['char_length'] = train['comment_text'].apply(lambda x: len(str(x)))
sns.set()
train['char_length'].hist()
plt.title('distribution of character length in all comments')
plt.show()