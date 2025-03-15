import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the dataset
df = pd.read_excel('English_Urdu.xlsx')
df = df.astype(str)
# Split the data into input (English) and output (Urdu)
english_sentences = df['English'].tolist()
urdu_sentences = df['Urdu'].tolist()

# Create tokenizers for English and Urdu
english_tokenizer = Tokenizer()
urdu_tokenizer = Tokenizer()

# Fit the tokenizers to the data
english_tokenizer.fit_on_texts(english_sentences)
urdu_tokenizer.fit_on_texts(urdu_sentences)

# Convert text to sequences
english_sequences = english_tokenizer.texts_to_sequences(english_sentences)
urdu_sequences = urdu_tokenizer.texts_to_sequences(urdu_sentences)

# Pad the sequences
max_english_length = 40
max_urdu_length = 40
padded_english = pad_sequences(english_sequences, maxlen=max_english_length, padding='post')
padded_urdu = pad_sequences(urdu_sequences, maxlen=max_urdu_length, padding='post')

# Split the data into training, validation, and test sets
train_size = int(0.8 * len(padded_english))
val_size = int(0.1 * len(padded_english))

train_english = padded_english[:train_size]
train_urdu = padded_urdu[:train_size]
val_english = padded_english[train_size:train_size+val_size]
val_urdu = padded_urdu[train_size:train_size+val_size]
test_english = padded_english[train_size+val_size:]
test_urdu = padded_urdu[train_size+val_size:]


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, TimeDistributed, Bidirectional

# Define the RNN-based model

rnn_model = Sequential([
    Embedding(input_dim=len(english_tokenizer.word_index)+1, output_dim=128),
    Bidirectional(SimpleRNN(128, return_sequences=True, activation='tanh')),
    TimeDistributed(Dense(len(urdu_tokenizer.word_index)+1, activation='softmax'))
])
# Compile the model
rnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# Train the model
from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint('english_to_urdu_rnn_model.keras', 
                             monitor='val_accuracy', 
                             mode='max', 
                             save_best_only=True, 
                             verbose=1)

# Train the model

rnn_model.fit(train_english, train_urdu, epochs=25, batch_size=64, 
              validation_data=(val_english, val_urdu), 
              callbacks=[checkpoint], 
              verbose=1)

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize


from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def calculate_bleu(y_true, y_pred):
    """
    Calculate BLEU score for machine translation model.

    Args:
    y_true (numpy array): True sentences.
    y_pred (numpy array): Predicted sentences.

    Returns:
    float: Mean BLEU score.
    """
    smoothing_func = SmoothingFunction()
    bleu_scores = []

    for i in range(len(y_true)):
        # Ignore padding tokens and handle unknown words
        true_sentence = []
        for word in y_true[i]:
            if word != 0:  # ignore padding tokens
                word = urdu_tokenizer.index_word.get(word, '')
                if word:  # ignore empty strings (unknown words)
                    true_sentence.append(word)

        # Get indices of top words in y_pred
        pred_indices = np.argmax(y_pred[i], axis=1)
        
        # Get words from indices
        pred_sentence = [urdu_tokenizer.index_word.get(word, '') for word in pred_indices]
        pred_sentence = [word for word in pred_sentence if word]  # ignore unknown words

        # Tokenize sentences
        true_sentence = word_tokenize(' '.join(true_sentence))
        pred_sentence = word_tokenize(' '.join(pred_sentence))

        # Calculate BLEU score with smoothing
        score = sentence_bleu([true_sentence], pred_sentence, smoothing_function=smoothing_func.method4)
        bleu_scores.append(score)

    return np.mean(bleu_scores)# Evaluate the model
test_pred = rnn_model.predict(test_english)
test_bleu = calculate_bleu(test_urdu, test_pred)
print(f'Test BLEU score: {test_bleu:.4f}')





def translate(sentence):
    sequence = english_tokenizer.texts_to_sequences([sentence])
    padded_sequence = pad_sequences(sequence, maxlen=max_english_length, padding='post')
    prediction = rnn_model.predict(padded_sequence)
    translation = []
    for word_prob in prediction[0]:
        predicted_index = np.argmax(word_prob)
        word = urdu_tokenizer.index_word.get(predicted_index, 'UNK')
        translation.append(word)
    return ' '.join(translation)


test_sentences = [
    "Hello, how are you?",
    "What is your name?",
    "I am from Pakistan.",
    "How old are you?",
    "What do you do?",
    "I love reading books.",
    "Where do you live?",
    "What is your favorite food?",
    "I am learning Urdu.",
    "Goodbye, take care."
]


for sentence in test_sentences:
    predicted_translation = translate(sentence)
    print(f"Input: {sentence}")
    print(f"Predicted Translation: {predicted_translation}")
    print()