# Netflix Data Analysis using Python and Deep Learning

Exploratory data analysis of the Netflix titles catalog, followed by a
Bidirectional LSTM (TensorFlow/Keras) that predicts a title's genre family
from its text synopsis.

## Dataset

`data/netflix_titles.csv` — 7,787 Netflix titles (movies + TV shows), with
columns for type, title, director, cast, country, date added, release year,
rating, duration, genres (`listed_in`), and a text `description`
(synopsis). Several fields — `country`, `cast`, and `listed_in` — are
comma-separated multi-value strings that need to be split before they're
useful for aggregation or modelling.

Source: Netflix Movies and TV Shows dataset (originally published on
Kaggle by Shivam Bansal), mirrored via the
[tidytuesday project](https://github.com/rfordatascience/tidytuesday/blob/main/data/2021/2021-04-20/readme.md).

## Project structure

```
netflix-data-analysis/
├── data/
│   └── netflix_titles.csv
├── src/
│   ├── 01_eda.py                        # catalog trends: genre, country, year, rating
│   ├── 02_feature_engineering.py        # structured features from multi-value fields
│   └── 03_genre_classification_lstm.py  # Bidirectional LSTM genre classifier
├── outputs/                             # generated on run: figures, features csv, metrics
├── requirements.txt
└── README.md
```

## How to run

```bash
pip install -r requirements.txt
python src/01_eda.py
python src/02_feature_engineering.py
python src/03_genre_classification_lstm.py
```

## 1. Exploratory Data Analysis (`01_eda.py`)

- Explodes the multi-value `listed_in` and `country` fields to get true
  genre and country frequencies (a title tagged "Dramas, International
  Movies" counts once toward *each* genre, not as a single combined label).
- Finds catalog growth is heavily skewed toward 2016–2020 (over 4,000 of
  the 7,787 titles), and that Dramas, Comedies, and Documentaries are the
  three largest genre families.
- Saves bar/line charts to `outputs/figures/` (top genres, top countries,
  titles by release year, movies vs TV shows added per year).

## 2. Feature Engineering (`02_feature_engineering.py`)

Turns the messy raw fields into structured, model-ready columns:

- `genre_list`, `cast_list`, `country_list` — parsed lists from the
  comma-separated raw strings.
- `primary_genre`, `primary_country` — first-listed value, used as a
  simplified label.
- `num_genres`, `num_cast`, `num_countries`, `is_international` — derived
  counts/flags.
- `duration_value` / `duration_unit` — splits `"90 min"` and `"3 Seasons"`
  into a numeric value and a unit, since movies and TV shows use
  incompatible duration formats.
- `release_era` — bucketed release-year ranges for easier grouping.
- `*_missing` flags for `director`, `cast`, `country`, `rating` instead of
  silently dropping rows with missing values.

Output: `outputs/netflix_features.csv`.

## 3. Genre Classification with a Bidirectional LSTM (`03_genre_classification_lstm.py`)

Predicts a title's **genre family** from its `description` text alone.

- The ~40 raw `listed_in` genre labels are collapsed into 8 coarse families
  (Drama, Comedy, Documentary, Action, Horror, Crime, Kids, Other) — a
  one-line synopsis essentially never states its genre outright, so the
  raw label space is too fine-grained and imbalanced to be learnable from
  text alone.
- Text pipeline: Keras `TextVectorization` (10k vocab, sequences padded/
  truncated to 60 tokens) → trainable embedding (64-dim) → two stacked
  **Bidirectional LSTM** layers (64 then 32 units) → dense + dropout →
  softmax over the 8 classes.
- Trained for 12 epochs, Adam optimizer, sparse categorical cross-entropy,
  80/20 stratified train/test split.

### Results (this run)

| Metric | Value |
|---|---|
| Test accuracy | **39.7%** |
| Majority-class baseline (always predict "Drama") | 29.3% |
| Train accuracy (epoch 12) | 93.7% |

Full per-class precision/recall/F1 is in `outputs/lstm_metrics.txt` and the
epoch-by-epoch training curve in `outputs/lstm_training_history.csv`.

**Honest caveat:** training accuracy climbs to ~94% while validation
accuracy plateaus around 35–39%, i.e. the model overfits — expected for an
8-way classification task learned from a single sentence of freeform text
with only ~6,200 training examples. Genres like Documentary and Kids,
which have distinctive vocabulary ("explores", "true story", "kids",
"family"), classify noticeably better than Crime or "Other", which overlap
heavily with Drama and Action in wording. Straightforward next steps to
improve this: pretrained embeddings (GloVe/word2vec), predicting the *set*
of genres per title (multi-label) instead of one collapsed family, or
adding `cast`/`director` as auxiliary features.

> **Note on numbers:** this README reports the actual metrics produced by
> running this exact code against the dataset in `data/`. If you're
> comparing against a different write-up of this project with different
> figures, that run likely used a different dataset snapshot, train/test
> split, or genre grouping — rerun `03_genre_classification_lstm.py` to
> reproduce the numbers above from scratch.
