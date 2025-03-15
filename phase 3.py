import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import ast
from gensim.models import Word2Vec

file_name = 'reviews.txt'
with open(file_name, 'r', encoding='utf-8') as file:
    reviews = [line.strip() for line in file]

tokenize = []
tokenize_reviews = []

for sentence in reviews:
    tokenize = word_tokenize(sentence)
    tokenize_reviews.append(tokenize)


file_name = 'tokenize_reviews.txt'
with open(file_name, 'w', encoding='utf-8') as file:
    for tokenized_review in tokenize_reviews:
        file.write(str(tokenized_review) + '\n')

with open(file_name, 'r', encoding='utf-8') as file:
        rev = [' '.join(ast.literal_eval(line.strip())) for line in file]



tfidf = TfidfVectorizer()
result = tfidf.fit_transform(rev)
feature_names = tfidf.get_feature_names_out()
tfidf_array = result.toarray()
tfidf_df = pd.DataFrame(tfidf_array, columns=feature_names)

# Find the top 10 highest TF-IDF values
top_terms = tfidf_df.max().nlargest(10)

# Print the top terms and their corresponding highest TF-IDF scores
print("Top 10 Terms with Highest TF-IDF Values:")
for term, score in top_terms.items():
    print(f"Term: {term} | Highest TF-IDF Value: {score}")
# count = 0
# for sen in tokenize_reviews:
#      for word in sen:
#           if word == "اچھا":
#                count+=1
# print('Count is ', count)
model = Word2Vec(sentences=tokenize_reviews, vector_size=100, window=5, min_count=1, workers=4, epochs=50)


# Step 3: Find similar words to "اچھا"
similar_words = model.wv.most_similar("اچھا", topn=5)

# Output the top 5 similar words
for word, similarity in similar_words:
    print(f"Word: {word}, Similarity: {similarity}")