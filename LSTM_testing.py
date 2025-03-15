import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize


# Load data
df = pd.read_excel('English_Urdu.xlsx')
df = df.astype(str)


# Create tokenizers
english_tokenizer = Tokenizer()
urdu_tokenizer = Tokenizer()


# Fit tokenizers to data
english_tokenizer.fit_on_texts(df['English'])
urdu_tokenizer.fit_on_texts(df['Urdu'])


# Get vocabulary
english_word_index = english_tokenizer.word_index
urdu_word_index = urdu_tokenizer.word_index
urdu_index_word = {v: k for k, v in urdu_word_index.items()}


# Define max sequence length
max_english_length = 40
max_urdu_length = 40


# Load pre-trained LSTM model
model = load_model('lstm_original.h5')


# Define translation function
def translate(sentence):
    sequence = english_tokenizer.texts_to_sequences([sentence])
    padded_sequence = pad_sequences(sequence, maxlen=max_english_length, padding='post')
    prediction = model.predict(padded_sequence)
    translation = []
    for word_prob in prediction[0]:
        predicted_index = np.argmax(word_prob)
        word = urdu_index_word.get(predicted_index, 'UNK')
        translation.append(word)
    return ' '.join(translation)


# Calculate BLEU score
smooth = SmoothingFunction()


# Test sentences with corresponding reference translations
test_data = df[['English', 'Urdu']].values


# Define the number of sentences to process
num_sentences = 10


# Calculate BLEU score for each sentence
bleu_scores = []
for i, (english_sentence, urdu_reference) in enumerate(test_data):
    if i >= num_sentences:
        break

    predicted_translation = translate(english_sentence)

    predicted_translation_tokens = word_tokenize(predicted_translation)
    reference_translation_tokens = word_tokenize(urdu_reference)

    bleu_score = sentence_bleu([reference_translation_tokens], predicted_translation_tokens, smoothing_function=smooth.method4)

    bleu_scores.append(bleu_score)

    print(f"Input: {english_sentence}")
    print(f"Predicted Translation: {predicted_translation}")
    print(f"Reference Translation: {urdu_reference}")
    print(f"BLEU Score: {bleu_score:.4f}")
    print()


# Calculate overall BLEU score
overall_bleu_score = sum(bleu_scores) / len(bleu_scores)
print(f"Overall BLEU Score: {overall_bleu_score:.4f}")


# Print BLEU scores for each sentence
print("BLEU Scores for each sentence:")
for i, bleu_score in enumerate(bleu_scores):
    print(f"Sentence {i+1}: {bleu_score:.4f}")


# Additional test sentences
additional_test_sentences = [
    "How are you today?",
    "What is your favorite food?",
    "I love reading books.",
    "Where do you live?",
    "What do you do?",
    "I am learning Urdu.",
    "Goodbye, take care.",
    "Hello, how can I help you?",
    "I am from Pakistan.",
    "What is your name?"
]

print("\nAdditional Test Sentences:")
for sentence in additional_test_sentences:
    predicted_translation = translate(sentence)
    print(f"Input: {sentence}")
    print(f"Predicted Translation: {predicted_translation}")
    print()