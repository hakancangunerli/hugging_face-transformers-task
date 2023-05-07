# %%
import pandas as pd
import numpy as np
import sklearn as sk
# !pip install translate
from translate import Translator

# %%
# convert the _chat.txt file into a dataframe
df = pd.read_csv('./_chat.txt', sep='\t', header=None)

# %%
df[0] = df[df[0].str.contains("görüntü") == False]
df[0] = df[df[0].str.contains("video") == False]
df[0] = df[df[0].str.contains("Çıkartma") == False]
df

# %%
# split the dataframe into 3 columns, one for each column in the _chat.txt file
df = df[0].str.split(' ', expand=True)

# %%
df

# %%
# remove first three columns 
df = df.drop(df.columns[[0, 1, 2]], axis=1)
# combine the first four columns into one column    
df[3] = df[3].str.cat([df[4], df[5], df[6]], sep=" ")
# remove columns 4, 5, 6
df = df.drop([4,5,6], axis=1)
# remove the first row
df = df.drop(df.index[0])

# %%
# combine the coluumns into one column
# replace none with empty string 
df = df.replace(np.nan, '', regex=True)

# %%
# reset column numbers 
df = df.reset_index(drop=True)
df = df.rename(columns={3: "Who texted"})

# %%
# combine the rest of the columns into one column
df['Message'] = df[df.columns[1:]].apply( lambda x: ' '.join(x.dropna().astype(str)), axis=1)

# %%
# combine the message and who texted columns into a new dataframe
df_final = df[['Who texted', 'Message']]
df_final['Message'] =df_final['Message'].str.replace('[^\w\s]','')

# %%
df_final

# %%
# convert all message to lowercase
df_final['Message'] = df_final['Message'].str.lower()
# convert all messages to english 
translator= Translator(to_lang="en")
df_final['Message'] = df_final['Message'].apply(translator.translate)

# %%
df_final

# %%
#use nltk to tokenize the messages
import nltk

# %%
nltk.download('punkt')

# %%
from nltk.tokenize import word_tokenize

df_final['Message'] = df_final['Message'].apply(word_tokenize)

# %%
df_final

# %%
# stem the words
from nltk.stem import PorterStemmer 

ps = PorterStemmer()

df_final['Message'] = df_final['Message'].apply(lambda x: [ps.stem(y) for y in x])
df_final

# %%
# combine them back again into a string

df_final['Message'] = df_final['Message'].apply(lambda x: ' '.join(x))

df_final

# %%
from transformers  import pipeline

# perform sentiment analysis on the messages, TODO: this only does it to the entire message, not to each word which could yield better results, or worse since it would be more biased towards the sentiment of the each word, not entirely. 

classifier = pipeline('sentiment-analysis')

# put the label and score into a new dataframe
df_final['Sentiment'] = df_final['Message'].apply(classifier)

# %%
# df_copy = df_final.copy() 

# %%
df_final = df_copy.copy()


# %%
df_final

# %%
df_final.to_csv('output.csv')

# label goes into label column, score goes into score column 
df_final['label'] = df_final['Sentiment'].apply(lambda x: x[0]['label'])
df_final['score'] = df_final['Sentiment'].apply(lambda x: x[0]['score'])

# %%
# if the label is positive, then the score is positive, if the label is negative, then the score is negative

for index, row in df_final.iterrows():
    if row['label'] == 'NEGATIVE':
        df_final.at[index, 'score'] = df_final.at[index, 'score'] * -1

df_final

# %%
df_final.drop(['Sentiment'], axis=1, inplace=True)
df_final.drop(['label'], axis=1, inplace=True)

# %%
df_final

# %%
df_final.describe()

# %%
# plot the score column, they go between -1 and -0.5 and 0.5 and 1, so we can see that the messages are mostly positive, but there are some negative ones as well.

import matplotlib.pyplot as plt

# plot from -1 to -0.5 intervals (represents negative sentiment)
plt.hist(df_final['score'], bins=np.arange(-1, -0.5, 0.01))
# add a title to the plot

plt.title("Negative Sentiment")
# make another plot from 0.5 to 1 intervals (represents positive sentiment)

# %%
# plot positive sentiment from 0.5 to 1

plt.hist(df_final['score'], bins=np.arange(0.5, 1, 0.01))

plt.title("Positive Sentiment")

plt.show()


