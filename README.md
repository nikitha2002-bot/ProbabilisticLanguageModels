# Probabilistic Language Models with Word2Vec Extension

**Course:** PROG8245 – Machine Learning Programming  
**Team:** Team 5  
**Dataset:** Wikitext-2 (raw, cleaned)  
**Environment:** Python 3.10+, Jupyter Notebook, Gensim, Scikit-learn, NLTK

## Team Members
- Mandeep Singh (ID: 8989367)  
- Kumari Nikitha Singh (ID: 9053016)  
- Krishna Reddy Bovilla (ID: 905861)

## Overview

This project demonstrates the full pipeline of probabilistic and vector space language modeling using a real-world corpus. We implement:

- Text preprocessing (tokenization, stopword removal, stemming)
- Unigram and Bigram models (with and without smoothing)
- Sentence probability scoring using Chain Rule and bigram probabilities
- Word2Vec embeddings using Gensim's skip-gram model
- Cosine similarity-based query retrieval
- Bigram-based next-word prediction

Our work blends classical NLP techniques with modern word embedding methods.

## Project Structure

```
ProbabilisticLanguageModels/
│
├── ProbabilisticLanguageModels.ipynb   # Main Jupyter Notebook
├── requirements.txt                    # Python dependencies
├── README.md                           # Project documentation
```

## Installation & Setup

### 1. Clone the repository
```
git clone https://github.com/nikitha2002-bot/ProbabilisticLanguageModels.git
cd ProbabilisticLanguageModels
```

### 2. Set up a virtual environment (recommended)
```
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```
pip install -r requirements.txt
```

### 4. Launch Jupyter Notebook
```
jupyter notebook
```

Open `ProbabilisticLanguageModels.ipynb` to explore the implementation step-by-step.

## Model Highlights

- Tokenization using regex and lowercasing  
- Normalization with NLTK stopwords and Porter Stemmer  
- Unigram and Bigram Probabilities  
- Laplace Smoothing for bigrams  
- Chain Rule for sentence-level probability  
- Word2Vec Embeddings trained from scratch  
- Vector-based Document Retrieval  
- Bigram Next-Word Prediction

## Sample Output

- Sentence Probability (Laplace Bigram): 3.1e-9
- Most Similar Words to 'king': [('henri', 0.79), ('throne', 0.76), ...]
- Top 3 Query Matches for: "global government and war"

## Dependencies

- nltk
- gensim
- scikit-learn
- datasets
- numpy

All dependencies are listed in `requirements.txt`.

## Status

Completed – Ready for evaluation and demo.

This project showcases a hybrid language modeling pipeline integrating traditional probabilistic methods with modern vector semantics.

## License

This repository is for academic purposes and coursework under Conestoga College.
