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

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| Logistic Regression (baseline, TF-IDF) | 0.630 | 0.630 |
| Logistic Regression (tuned, TF-IDF)    | 0.637 | 0.636 |

The tuned Logistic Regression provides a strong, fast baseline at ~64%
accuracy. The notebook additionally trains and compares the full set of
recurrent architectures across the three embedding strategies and unit sizes;
see the notebook and the accompanying PDF report for the complete comparison
plots and confusion matrices.

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
