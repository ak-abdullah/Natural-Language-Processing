import pandas as pd
import re
import string

import re
def no_sentiment(reviews, sarcasm):
    i = 0
    while i < len(reviews):
        filtered_words = reviews[i].split()
        if len(filtered_words) < 3:
            reviews.pop(i)
            sarcasm.pop(i)  
        else:
            i += 1  
    return reviews, sarcasm

    
def remove_emoji(text):
    emoji_pattern = re.compile(
        "[" 
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"  # various symbols
        u"\U000024C2-\U0001F251"  # enclosed characters
        u"\U0001F900-\U0001F9FF"  # additional emojis
        u"\U0001F170-\U0001F171"  # A & B button emojis
        u"\U0001F18E"             # letter "N" button emoji
        u"\U0001F19A"             # "squared" variants
        "]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub(r'', text)


def load_stop_words(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        stop_words = set(line.strip() for line in file)
    return stop_words

stopwords1 = load_stop_words('stopwords-ur.txt')
stopwords2 = load_stop_words('stopwords.txt')

punc = string.punctuation
file_path = 'urdu_sarcastic_dataset.csv'  
data = pd.read_csv(file_path, header = None,encoding='utf-8')

data = data.iloc[0:, :2]
data.columns = ['Comments', 'Sarcasm']

filtered_sentences = []
is_sarcasm = []
combined_stopwords = set(stopwords1).union(set(stopwords2))
not_string = []
for i in range(1,len(data)):
    text = data.iloc[i, 0]
    flag = data.iloc[i, 1]

    if isinstance(text,str):
        text = text.translate(str.maketrans('', '', punc))
        text = remove_emoji(text)
        filtered_words = [word for word in text.split() if word not in combined_stopwords]
        filtered_sentence = ' '.join(filtered_words)
        filtered_sentences.append(filtered_sentence)
        is_sarcasm.append(flag)
        
filtered_data,is_sarcasm = no_sentiment(filtered_sentences, is_sarcasm)

filtered_data = pd.DataFrame({
    'Comments': filtered_sentences,  
    'Sarcasm': is_sarcasm
})

filtered_data.to_csv('filtered_output.csv', index=False, header=False,encoding='utf-8')






