import nltk
from nltk import FreqDist, bigrams, trigrams


all_tokens = []
file_name = 'reviews.txt'
with open(file_name, 'r', encoding='utf-8') as file:
    reviews = [line.strip() for line in file]
    
for i in range(len(reviews)):
    tokens = nltk.word_tokenize(reviews[i])
    all_tokens.extend(tokens) 
    

    
unigram_list = all_tokens

print("Tokens:", tokens)

bigram_list = list(bigrams(all_tokens))


trigram_list = list(trigrams(all_tokens))


unigram_freq = FreqDist(unigram_list)
bigram_freq = FreqDist(bigram_list)
trigram_freq = FreqDist(trigram_list)


top_10_bigrams = bigram_freq.most_common(10)
top_10_trigrams = trigram_freq.most_common(10)

print("Top 10 Bigrams:", top_10_bigrams)
print("Top 10 Trigrams:", top_10_trigrams)
