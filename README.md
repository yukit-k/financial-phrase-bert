# Predict sentiment on financial news

## Project Description

This project tests the advantages of fine-tuning pre-trained language models (BERT, ALBERT) for financial sentiment analysis, compared against classical NLP approaches (Lexicon-based, TF-IDF, LSTM).

Please refer to the [blog post](https://medium.com/@yukitakahashi_46678/nlp-in-the-financial-market-sentiment-analysis-9de0dda95dc?source=friends_link&sk=9bb03bc7e108c125499bc9cf1996bd49) for full background.

## Files

| Notebook | Description |
|---|---|
| `SA_Model_Comparison_Finphrase.ipynb` | Sentiment Analysis on FinancialPhrase Bank data — compares Lexicon, TF-IDF, LSTM, BERT, ALBERT |
| `SA_Model_Comparison_Tweet.ipynb` | Sentiment Analysis on Stocktwit tweet data — same model comparison |
| `SA_Tweet_LSTMvsBERT.ipynb` | Deep-dive LSTM vs BERT comparison on tweet data |

## Getting Started

### Requirements

- Python 3.9+
- See `requirements.txt` for all dependencies

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd financial-phrase-bert

# Install dependencies
pip install -r requirements.txt

# Download required NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('wordnet')"
```

### Data

Download the datasets and place them in the following directories before running the notebooks:

- **FinancialPhrase Bank**: Download from [this link](https://www.researchgate.net/profile/Pekka_Malo/publication/251231364_FinancialPhraseBank-v10/data/0c96051eee4fb1d56e000000/FinancialPhraseBank-v10.zip) and unzip to `./data/FinancialPhraseBank-v1.0/`
- **Loughran-McDonald Dictionary**: Place CSV files in `./data/LoughranMcDonald/`
- **GloVe embeddings**: Place pre-trained GloVe files in `./data/GloVe/`
- **Tweet data**: Place Stocktwit data in `./data/tweet/`

### Running the notebooks

```bash
jupyter notebook
```

Open any of the `.ipynb` files and run all cells. Google Colab is also supported — the notebooks detect the environment automatically.

## What was updated (May 2026)

The codebase was updated from the original 2020 versions to work with current library releases:

| Change | Old | New |
|---|---|---|
| **torch** | 1.6.0 | ≥2.12.0 |
| **transformers** | 3.3.1 | ≥5.8.1 |
| **scikit-learn** | (unversioned) | ≥1.8.0 |
| **seaborn** | 0.11.0 | ≥0.13.2 |
| **nltk** | 3.5 | ≥3.9.4 |
| **wordcloud** | 1.8.0 | ≥1.9.6 |
| **numpy / pandas / tqdm / matplotlib** | (old) | latest |

### API migrations applied to all notebooks

- **`sns.distplot`** → `sns.histplot(..., kde=True)` (removed in seaborn 0.12)
- **`sklearn.metrics.plot_confusion_matrix`** → `ConfusionMatrixDisplay.from_estimator` (removed in scikit-learn 1.2)
- **`skplt.metrics.plot_confusion_matrix`** → `ConfusionMatrixDisplay.from_predictions`
- **`transformers.AdamW`** → `torch.optim.AdamW` (removed from transformers 4.x)
- **`get_linear_schedule_with_warmup`** now imported from `transformers` directly
- **`correct_bias=False`** parameter removed from AdamW (not supported in `torch.optim.AdamW`)
- **`loss, logits = model(...)`** → `_out = model(...); loss, logits = _out.loss, _out.logits` (transformers 5.x returns `ModelOutput`, no longer tuple-unpackable directly)
- **`pd.set_option('display.max_colwidth', -1)`** → `None` (deprecated in pandas 1.0)
- **`from torch.autograd import Variable`** removed (deprecated since PyTorch 0.4; plain tensors replaced it)
- **`nltk.download('punkt_tab')`** added (required by NLTK 3.6+)
