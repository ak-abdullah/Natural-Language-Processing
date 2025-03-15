import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pandas as pd
from LughaatNLP import LughaatNLP


file_path = 'filtered_output.csv'  
data = pd.read_csv(file_path, header = None,encoding='utf-8')

reviews = []

urdu_text_processing = LughaatNLP()
for i in range(len(data)):
    normalized_text = urdu_text_processing.normalize(data.iloc[i,0])
    stemmed_sentence = urdu_text_processing.urdu_stemmer(normalized_text)
    lemmatized_sentence = urdu_text_processing.lemmatize_sentence(stemmed_sentence)
    filtered_text = urdu_text_processing.remove_stopwords(lemmatized_sentence)
    remove_english = urdu_text_processing.remove_english(filtered_text)
    pure_urdu = urdu_text_processing.pure_urdu(remove_english)
    special_char = urdu_text_processing.remove_special_characters(pure_urdu)
    reviews.append(special_char)

    
file_name = 'reviews.txt'


# Open the file in write mode with UTF-8 encoding and save the list
with open(file_name, 'w', encoding='utf-8') as file:

    for i in range(len(reviews)):
        file.write(f"{reviews[i]}\n")

print(f"List saved to {file_name}")






