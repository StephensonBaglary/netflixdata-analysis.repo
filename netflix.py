"""
03_genre_classifier.py
-----------------------
Trains a Bidirectional LSTM (TensorFlow/Keras) that reads a title's
"description" (its synopsis) and predicts its primary genre.

What this does, step by step:
1. Load netflix_titles.csv.
2. Get each title's primary genre (the first genre in "listed_in").
3. Keep only the TOP_N_GENRES most common genres — this keeps the problem
   balanced enough to learn, and gives us a clear "always guess the most
   common genre" baseline to beat.
4. Turn the description text into padded number sequences (tokenize).
5. Split into train/test sets.
6. Build an Embedding -> Bidirectional LSTM -> pooling -> Dense model.
7. Train it, then report test accuracy vs. the baseline.

HOW TO RUN:
Put this script and "netflix_titles.csv" in the SAME folder, then run:
    python 03_genre_classifier.py
This can take a few minutes on a laptop CPU — that's normal, LSTMs aren't fast.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, models, regularizers

# ---------------------------------------------------------------------------
# Settings — tweak these if you want to experiment
# ---------------------------------------------------------------------------
DATA_FILE = "/Users/stephensonbaglary/Downloads/github_projects/netflix-data-analysis/data/netflix_titles.csv"
TOP_N_GENRES = 10          # only classify the 10 most common genres
VOCAB_SIZE = 8000          # how many unique words to keep
MAX_LENGTH = 60            # max words per description (longer ones get cut)
EMBEDDING_DIM = 100
LSTM_UNITS = 64
EPOCHS = 25
BATCH_SIZE = 32
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# ---------------------------------------------------------------------------
# STEP 1-2: Load data and get each title's primary genre
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_FILE)
df["primary_genre"] = df["listed_in"].apply(lambda g: g.split(",")[0].strip())

# ---------------------------------------------------------------------------
# STEP 3: Keep only the top N genres
# ---------------------------------------------------------------------------
top_genres = df["primary_genre"].value_counts().head(TOP_N_GENRES).index
df = df[df["primary_genre"].isin(top_genres)].copy()

print(f"Rows kept after filtering to top {TOP_N_GENRES} genres: {len(df)}")

baseline_accuracy = df["primary_genre"].value_counts().iloc[0] / len(df)
print(f"Baseline accuracy (always guess most common genre): {baseline_accuracy:.1%}")
print()

# ---------------------------------------------------------------------------
# STEP 4: Encode text and labels
# ---------------------------------------------------------------------------
texts = df["description"].astype(str).values
labels = df["primary_genre"].values

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=MAX_LENGTH, padding="post", truncating="post")

# ---------------------------------------------------------------------------
# STEP 5: Train / test split (stratified)
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    padded_sequences,
    encoded_labels,
    test_size=0.15,
    random_state=RANDOM_SEED,
    stratify=encoded_labels,
)

# ---------------------------------------------------------------------------
# STEP 6: Build the Bidirectional LSTM model
# We combine average-pooling and max-pooling over the LSTM outputs — this
# tends to work better for short-text classification than either alone,
# since it captures both the overall tone and the single strongest signal.
# ---------------------------------------------------------------------------
num_classes = len(label_encoder.classes_)

inputs = layers.Input(shape=(MAX_LENGTH,))
x = layers.Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM)(inputs)
x = layers.SpatialDropout1D(0.3)(x)
x = layers.Bidirectional(layers.LSTM(LSTM_UNITS, return_sequences=True, recurrent_dropout=0.1))(x)
avg_pool = layers.GlobalAveragePooling1D()(x)
max_pool = layers.GlobalMaxPooling1D()(x)
x = layers.Concatenate()([avg_pool, max_pool])
x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = models.Model(inputs, outputs)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

# ---------------------------------------------------------------------------
# STEP 7: Train and evaluate
# ---------------------------------------------------------------------------
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy", patience=5, restore_best_weights=True
)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5
)

history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop, reduce_lr],
    verbose=2,
)

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

print()
print("=" * 60)
print("RESULTS")
print("=" * 60)
print(f"Baseline accuracy:        {baseline_accuracy:.1%}")
print(f"BiLSTM test accuracy:     {test_accuracy:.1%}")
print(f"Improvement over baseline: {test_accuracy / baseline_accuracy:.2f}x")