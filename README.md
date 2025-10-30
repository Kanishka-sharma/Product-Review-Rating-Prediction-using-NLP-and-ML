# Product Review Rating Prediction using NLP and Machine Learning

## Project Overview

This project explores predicting Amazon product review ratings (1-5) based solely on review text using NLP and machine learning techniques. The goal is to build and compare models that effectively interpret customer sentiment to estimate satisfaction levels.

---

## Dataset Summary

The dataset (`dataset.csv`) contains Amazon product reviews with two columns:

| Column | Description |
|--------|-------------|
| `Score` | Rating (1 to 5) |
| `Text`  | Review text |

This dataset is well-suited for sentiment analysis due to its authentic and linguistically diverse content. Challenges include:
- **Noise**: Sarcasm, typos, and mixed sentiments.
- **Brief Reviews**: Limited context (e.g., "Good").
- **Rating Inconsistencies**: Overlap in sentiment across rating classes.
- **Class Imbalance**: 5-star reviews dominate (over 31k), while 1- and 2-star are underrepresented (~4k and ~3k), risking bias.

Despite these issues, the dataset remains valuable for learning sentiment-driven classification patterns.

---

## Data Preparation Highlights

- **Cleaning**: Lowercasing, punctuation removal, tokenization, stopword removal, and lemmatization.
- **Negation Handling**: Combining terms like "not good" into "not_good" to preserve sentiment.
- **Missing & Short Reviews**: Removed or imputed for consistency.
- **Label Encoding**: Ratings converted to numerical labels.
- **Class Imbalance**: Handled using stratified undersampling and class weights.
- **Train-Test Split**: 80-20 with stratified sampling for balanced representation.

---

## Models & Representations

- **TF-IDF**: Used with Naive Bayes and KNN.
- **Word Embeddings**: Used with CNN and LSTM for contextual learning.
  
### Algorithms:
- **Naive Bayes**: Efficient baseline for sparse TF-IDF data.
- **KNN**: Distance-based, interpretable but computationally expensive.
- **CNN**: Captures local patterns and performs well on short reviews.
- **LSTM**: Models long-term dependencies; effective for longer text.

---

## Experimental Results (Accuracy)

| Model        | Accuracy | Highlights |
|--------------|----------|------------|
| Naive Bayes  | 60%      | Strong baseline, struggles with mixed sentiments. |
| KNN (k=9)     | 52%      | High recall for class 5, but weak for classes 2 & 4. |
| CNN          | 67%      | Best performer, excels with structured and short reviews. |
| LSTM         | 62%      | Handles long text well, but imbalanced precision on minority classes. |

---

## Discussion

- **Naive Bayes** offers a fast and interpretable baseline but is limited by its simplistic assumptions.
- **KNN** underperforms due to high sensitivity to class imbalance and sparsity.
- **CNN** outperforms others by effectively capturing key patterns in short texts.
- **LSTM**, while powerful, provides limited additional value in this setup due to shorter review lengths and data imbalance.

**Overall**, CNN strikes the best balance between accuracy and complexity. Future improvements could include ensemble approaches or leveraging pre-trained language models (e.g., BERT).


