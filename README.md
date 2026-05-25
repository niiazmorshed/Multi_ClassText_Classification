# Multi-Class Text Classification of Q&A Topics

Classifies question–answer text into **10 topic categories** and benchmarks a
range of models — from a classical TF-IDF + Logistic Regression baseline to
deep recurrent networks (RNN/GRU/LSTM and their bidirectional variants) with
trainable, Word2Vec (skip-gram), and GloVe word embeddings.

## Problem

Given a piece of Q&A text, predict its topic among 10 balanced classes:

`Business & Finance` · `Computers & Internet` · `Education & Reference` ·
`Entertainment & Music` · `Family & Relationships` · `Health` ·
`Politics & Government` · `Science & Mathematics` · `Society & Culture` · `Sports`

## Dataset

| Split | Samples | Columns |
|-------|---------|---------|
| Train | 93,333  | `QA Text`, `Class` |
| Test  | 59,999  | `QA Text`, `Class` |

The 10 classes are roughly balanced (~9,100–9,500 samples each in training),
so **accuracy** and **macro F1** are both meaningful evaluation metrics.

## Pipeline

1. **Text preprocessing** — lowercasing, removing non-alphabetic characters,
   stop-word removal (NLTK).
2. **Feature representation**
   - **TF-IDF** (top 5,000 features) for the classical and dense models.
   - **Tokenization + padding** (`max_len = 100`, vocab capped at 10,000) for
     the sequence models.
3. **Label encoding** of the 10 target classes.
4. **Modeling & evaluation** (see below).

## Models

| Family | Models | Features / Embeddings |
|--------|--------|-----------------------|
| Classical | Logistic Regression (baseline + `GridSearchCV` tuned) | TF-IDF |
| Dense NN | Deep feed-forward network | TF-IDF |
| Recurrent | RNN, GRU, LSTM, BiRNN, BiGRU, BiLSTM (64 / 128 / 256 units) | Trainable, Word2Vec skip-gram, GloVe |

Deep models use **dropout tuning** (0.2 / 0.3 / 0.4), **early stopping** on
validation loss, and the Adam optimizer. Each recurrent architecture is
evaluated against all three embedding strategies for a full comparison.

## Results

Performance comparison across all evaluated models:

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| **Logistic Regression (TF-IDF)** | **0.6495** | **0.6479** |
| Deep Neural Network (TF-IDF)     | 0.5922 | 0.5941 |
| SimpleRNN                        | 0.5011 | 0.4820 |
| GRU                              | 0.5525 | 0.5425 |
| LSTM                             | 0.5603 | 0.5549 |
| Bidirectional SimpleRNN          | 0.2746 | 0.2513 |
| Bidirectional GRU                | 0.5544 | 0.5511 |
| Bidirectional LSTM               | 0.5457 | 0.5516 |

**Key finding:** Logistic Regression with TF-IDF achieved the best overall
performance (**Macro F1 = 0.6479**), outperforming all neural architectures.
On this large, well-structured, balanced dataset, strong class-specific
keywords make sparse TF-IDF features highly discriminative — neural models must
jointly learn embeddings and a classifier, which can dilute the learning signal
when lexical cues alone are sufficient. Among the recurrent models, GRU and LSTM
were the most stable, while the Bidirectional SimpleRNN performed poorly due to
vanishing gradients.

See the accompanying PDF report for the full comparison plots and confusion
matrices.

## Tech Stack

Python · pandas · NumPy · scikit-learn · TensorFlow / Keras · Gensim (Word2Vec) ·
NLTK · Matplotlib · Seaborn · GloVe embeddings · Google Colab

## Repository Contents

| File | Description |
|------|-------------|
| `Multi_ClassText_Classification_Report.ipynb` | Full notebook: preprocessing, EDA, modeling, evaluation |
| `Multi_ClassText_Classification_Report.pdf`   | Exported report with all outputs and plots |
| `Question Answer Classification Dataset 8.csv` | Training dataset |

## How to Run

The notebook was developed in **Google Colab**. To reproduce locally:

```bash
pip install pandas numpy scikit-learn tensorflow gensim nltk matplotlib seaborn
```

You will also need:
- The GloVe embeddings file `glove.6B.100d.txt`
  ([download](https://nlp.stanford.edu/projects/glove/)).
- The test split CSV referenced in the notebook.

Then open the notebook and run the cells top to bottom.
