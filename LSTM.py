import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import numpy as np
import nltk
nltk.download('punkt')
# Download and load the parallel-corpus.xlsx dataset
df = pd.read_excel('English_Urdu.xlsx')
df = df.astype(str)

# Preprocess data
english_texts = df['English'].tolist()
urdu_texts = df['Urdu'].tolist()

english_tokenizer = Tokenizer()
urdu_tokenizer = Tokenizer()

english_tokenizer.fit_on_texts(english_texts)
urdu_tokenizer.fit_on_texts(urdu_texts)

english_vocab_size = len(english_tokenizer.word_index) + 1
urdu_vocab_size = len(urdu_tokenizer.word_index) + 1

max_length = 40

english_sequences = english_tokenizer.texts_to_sequences(english_texts)
urdu_sequences = urdu_tokenizer.texts_to_sequences(urdu_texts)

english_padded = pad_sequences(english_sequences, maxlen=max_length, padding='post')
urdu_padded = pad_sequences(urdu_sequences, maxlen=max_length, padding='post')

# Split data into training, validation, and testing sets
train_size = int(0.8 * len(english_padded))
val_size = int(0.1 * len(english_padded))
test_size = len(english_padded) - train_size - val_size

train_english, val_english, test_english = english_padded[:train_size], english_padded[train_size:train_size+val_size], english_padded[train_size+val_size:]
train_urdu, val_urdu, test_urdu = urdu_padded[:train_size], urdu_padded[train_size:train_size+val_size], urdu_padded[train_size+val_size:]

# Create dataset and data loader
train_dataset = tf.data.Dataset.from_tensor_slices((train_english, train_urdu))
val_dataset = tf.data.Dataset.from_tensor_slices((val_english, val_urdu))
test_dataset = tf.data.Dataset.from_tensor_slices((test_english, test_urdu))

batch_size = 64
train_dataset = train_dataset.shuffle(100).batch(batch_size)
val_dataset = val_dataset.batch(batch_size)
test_dataset = test_dataset.batch(batch_size)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Masking, TimeDistributed, Bidirectional

# Define LSTM model
model_lstm = Sequential([
    Embedding(input_dim=english_vocab_size, output_dim=256, input_length=max_length),
    Masking(mask_value=0),
    Bidirectional(LSTM(units=512, return_sequences=True)),
    TimeDistributed(Dense(urdu_vocab_size, activation='softmax'))
])

model_lstm.compile(optimizer='adam', 
                    loss='sparse_categorical_crossentropy', 
                    metrics=['accuracy'])

# Train LSTM model
history = model_lstm.fit(train_dataset, epochs=50, validation_data=val_dataset)
# Save LSTM model
model_lstm.save('lstm_translator.h5')





from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
import numpy as np

from nltk.translate.bleu_score import SmoothingFunction

smoothing_func = SmoothingFunction()

def calculate_bleu(reference, prediction):
    reference = word_tokenize(reference)
    prediction = word_tokenize(prediction)
    return sentence_bleu([reference], prediction, smoothing_function=smoothing_func.method4)

# Evaluate LSTM model
max_test_size = 20 # Limit test size
bleu_scores_lstm = []
for i in range(min(max_test_size, len(test_english))):
    reference = urdu_tokenizer.sequences_to_texts([test_urdu[i]])[0]
    prediction = model_lstm.predict(test_english[i:i+1])
    predicted_sequence = np.argmax(prediction, axis=2)[0]
    prediction = urdu_tokenizer.sequences_to_texts([predicted_sequence])[0]
    bleu_scores_lstm.append(calculate_bleu(reference, prediction))

print(f'Test Average BLEU Score (LSTM): {sum(bleu_scores_lstm) / len(bleu_scores_lstm):.4f}')


# Evaluate LSTM model on original corpus (10 sentences)
print("Evaluation on Original Corpus")
evaluation_size = 10
corpus_english = english_texts[:evaluation_size]
corpus_urdu = urdu_texts[:evaluation_size]

corpus_english_sequences = english_tokenizer.texts_to_sequences(corpus_english)
corpus_urdu_sequences = urdu_tokenizer.texts_to_sequences(corpus_urdu)

corpus_english_padded = pad_sequences(corpus_english_sequences, maxlen=max_length, padding='post')
corpus_urdu_padded = pad_sequences(corpus_urdu_sequences, maxlen=max_length, padding='post')

predictions_lstm = model_lstm.predict(corpus_english_padded)

for i in range(min(evaluation_size, len(corpus_english))):
    reference = corpus_urdu[i]
    predicted_sequence = np.argmax(predictions_lstm[i], axis=1)
    prediction = urdu_tokenizer.sequences_to_texts([predicted_sequence])[0]
    print(f'English: {corpus_english[i]}')
    print(f'Reference Urdu: {reference}')
    print(f'Predicted Urdu (LSTM): {prediction}')
    print(f'BLEU Score: {calculate_bleu(reference, prediction):.4f}')
    print('---')

# Add a termination point
print("Evaluation completed.")