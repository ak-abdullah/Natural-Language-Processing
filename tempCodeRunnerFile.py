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