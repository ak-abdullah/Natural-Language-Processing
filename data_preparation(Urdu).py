import pandas as pd
from LughaatNLP import NER_Urdu
from LughaatNLP import POS_urdu
from LughaatNLP import LughaatNLP


urdu_text_processing = LughaatNLP()


file = pd.read_excel('parallel-corpus.xlsx', header = None,usecols=[0, 1], skiprows=1)
duplicates = file[file.duplicated(keep=False)]

file[1] = file[1].apply(lambda x: urdu_text_processing.normalize(str(x)))
file[1] = file[1].apply(lambda x: urdu_text_processing.remove_english(str(x)))
file[1] = file[1].apply(lambda x: urdu_text_processing.remove_urls(str(x)))
file[1] = file[1].apply(lambda x: urdu_text_processing.remove_special_characters(x))




df = pd.read_excel('English_Sentences.xlsx')

df['Urdu'] = file.iloc[:, 1]

# Save the updated DataFrame back to the same file with both columns
df.to_excel('English_Urdu.xlsx', index=False, header=True)