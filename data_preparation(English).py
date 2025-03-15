import pandas as pd
import re
import string
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import contractions
import emoji


#functions

def clean_text(text):
    if isinstance(text, str):  # Check if the input is a string
        text = text.lower()  # Lowercase
        text = re.sub(r'\d+', '', text)  # Remove numbers
        text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
        text = re.sub(r'\W', ' ', text)  # Remove special characters
        text = BeautifulSoup(text, "html.parser").get_text()  # Remove HTML tags
        return text
    return ''  # Return an empty string for non-string inputs

def tokenize_english(text):
    return word_tokenize(text)

def remove_stopwords(tokenized_list):
    return [word for word in tokenized_list if word not in stop_words]

def lemmatize_words(words):
    return [lemmatizer.lemmatize(word) for word in words]

def stem_words(words):
    return [stemmer.stem(word) for word in words]

#Reading File
file = pd.read_excel('parallel-corpus.xlsx', header = None,usecols=[0, 1], skiprows=1)

#Text Cleaning Punctuation etc

file[0] = file[0].apply(clean_text)  # Clean English text 

#contractions

file[0] = [''.join(doc) for doc in file[0]]
file[0] = file[0].apply(contractions.fix)

# handling emojis
file[0] = file[0].apply(emoji.demojize)


#Saving File
file.iloc[:, 0].to_frame(name='English').to_excel('English_Sentences.xlsx', index=False, header=True)
