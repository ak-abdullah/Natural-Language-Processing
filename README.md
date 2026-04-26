# Natural Language Processing Projects

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=flat&logo=huggingface&logoColor=black)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)

A collection of NLP projects covering Urdu text processing, sarcasm detection, sequence modeling for machine translation, and LLM fine-tuning. Built across coursework at FAST-NUCES.

---

## 📂 What is inside

### 1. Urdu Sarcasm Detection

End-to-end sarcasm classification pipeline built on an Urdu social media comments dataset.

**Preprocessing pipeline:**
- Removed emojis, punctuation, and comments shorter than 3 words
- Urdu normalization, stemming, and lemmatization using LughaatNLP
- Removed English words and special characters to keep pure Urdu text
- Combined Urdu and English stopword lists for filtering

**Feature extraction and modeling:**
- TF-IDF vectorization
- Word2Vec embeddings (vector size 100, window 5, 50 epochs)
- Unigram, bigram, and trigram frequency analysis
- Multinomial Naive Bayes classifier with 80/20 train-test split

**Evaluation:** accuracy, precision, recall, F1-score, confusion matrix

---

### 2. English to Urdu Machine Translation

Sequence-to-sequence translation models trained on paired English-Urdu sentence datasets.

- Vanilla RNN encoder-decoder
- LSTM encoder-decoder
- Attention mechanism integration for improved translation quality

---

### 3. LLM Fine-Tuning (Phi-3)

Fine-tuned Microsoft Phi-3 small language model on custom NLP tasks using prompt engineering and task-specific fine-tuning. Evaluated on few-shot and zero-shot performance.

---

### 4. NLP Architectures Study

In-depth exploration of modern NLP architectures:
- Transformer architecture and self-attention mechanism
- BERT: masked language modeling, embeddings, and fine-tuning
- Comparison of model tradeoffs across tasks

---

## 🛠️ Stack

| Task | Tools |
|---|---|
| Urdu NLP | LughaatNLP, NLTK |
| Embeddings | Word2Vec (gensim), TF-IDF |
| Classification | scikit-learn (Naive Bayes) |
| Sequence Modeling | TensorFlow/Keras (RNN, LSTM) |
| LLM Fine-Tuning | HuggingFace Transformers (Phi-3) |
| Visualization | Matplotlib, Seaborn |

---

## 📁 Project structure

```
nlp/
├── phase 1.py          # Urdu preprocessing: emoji removal, stopwords, filtering
├── phase 2.py          # LughaatNLP: normalization, stemming, lemmatization
├── phase 3.py          # Tokenization, TF-IDF, Word2Vec embeddings
├── phase 4.py          # N-gram frequency analysis
├── phase 5.py          # Naive Bayes sarcasm classifier
├── phase 6.py          # Evaluation metrics and confusion matrix
├── Assignment_1.ipynb  # Full documented pipeline
├── urdu_sarcastic_dataset.csv
├── stopwords-ur.txt
└── stopwords.txt
```

---

## 🚀 Running locally

```bash
pip install pandas scikit-learn nltk gensim tensorflow LughaatNLP matplotlib seaborn
```

Run phases in order:

```bash
python "phase 1.py"
python "phase 2.py"
python "phase 3.py"
python "phase 4.py"
python "phase 5.py"
python "phase 6.py"
```

Or open `Assignment_1.ipynb` for the full documented pipeline.

---

## 💡 What I learned building this

Working with Urdu text is genuinely different from English NLP. Arabic script reads right to left, characters connect differently, and standard English tokenizers break on it. LughaatNLP handles Urdu-specific normalization including character variants that look the same but have different Unicode points.

Sarcasm detection in any language is hard because sarcasm is contextual. The Naive Bayes classifier on TF-IDF features gives a reasonable baseline but misses the semantic layer entirely. Fine-tuning a transformer on this task would get much better results.

The RNN to LSTM improvement on translation was clear and measurable. Vanilla RNNs lose context over long sentences. LSTMs hold it significantly better which shows directly in translation quality on longer inputs.

---

## 📬 Contact

Built by Abdullah Khalid

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/-abdullah-khalid)
[![Email](https://img.shields.io/badge/Email-D14836?style=flat&logo=gmail&logoColor=white)](mailto:abdullahkh.cs@gmail.com)
[![Portfolio](https://img.shields.io/badge/Portfolio-000000?style=flat&logo=github&logoColor=white)](https://ak-abdullah.github.io/Resume/)
