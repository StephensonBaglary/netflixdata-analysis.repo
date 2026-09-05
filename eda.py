"""
01_eda.py
---------
Simple exploratory data analysis (EDA) on the Netflix Titles dataset.

What this does, step by step:
1. Load netflix_titles.csv into a pandas DataFrame.
2. Print basic info (shape, missing values).
3. Look at genre trends (the "listed_in" column).
4. Look at country trends (the "country" column).
5. Look at release year trends.
6. Save simple seaborn charts for each into an "outputs" folder.

HOW TO RUN:
Put this script and "netflix_titles.csv" in the SAME folder, then run:
    python 01_eda.py
(or open it as a notebook cell-by-cell — just keep the CSV next to the notebook)
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # lets plots save to file without needing a display
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# STEP 1: Setup — simple, no fancy path tricks. Just keep the CSV next to
# this script/notebook and everything below works.
# ---------------------------------------------------------------------------
DATA_FILE = "/Users/stephensonbaglary/Downloads/github_projects/netflix-data-analysis/data/netflix_titles.csv"
OUTPUT_FOLDER = "outputs"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

sns.set_style("whitegrid")

# ---------------------------------------------------------------------------
# STEP 2: Load the data
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_FILE)

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"Rows x Columns: {df.shape}")
print()
print("Movies vs TV Shows:")
print(df["type"].value_counts())
print()
print("Missing values per column:")
print(df.isna().sum().sort_values(ascending=False))
print()

# ---------------------------------------------------------------------------
# STEP 3: Genre trends
# "listed_in" holds multiple genres per title, e.g. "Dramas, International Movies"
# We split on the comma so each genre gets counted separately.
# ---------------------------------------------------------------------------
all_genres = []
for genre_string in df["listed_in"].dropna():
    for genre in genre_string.split(","):
        all_genres.append(genre.strip())

genre_counts = pd.Series(all_genres).value_counts()

print("Top 15 genres:")
print(genre_counts.head(15))
print()

plt.figure(figsize=(10, 6))
top15_genres = genre_counts.head(15).sort_values()
sns.barplot(x=top15_genres.values, y=top15_genres.index, color="#B81D24")
plt.title("Top 15 Netflix Genres by Title Count")
plt.xlabel("Number of Titles")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "top_genres.png"), dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# STEP 4: Country trends
# "country" can also list multiple countries (co-productions), so we split
# the same way.
# ---------------------------------------------------------------------------
all_countries = []
for country_string in df["country"].dropna():
    for country in country_string.split(","):
        all_countries.append(country.strip())

country_counts = pd.Series(all_countries).value_counts()

print("Top 10 countries:")
print(country_counts.head(10))
print()

plt.figure(figsize=(10, 6))
top10_countries = country_counts.head(10).sort_values()
sns.barplot(x=top10_countries.values, y=top10_countries.index, color="#221F1F")
plt.title("Top 10 Countries by Netflix Title Count")
plt.xlabel("Number of Titles")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "top_countries.png"), dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# STEP 5: Release year trends
# ---------------------------------------------------------------------------
titles_per_year = df["release_year"].value_counts().sort_index()
recent_years = titles_per_year[titles_per_year.index >= 1990]

print("Titles released per year (1990+, last 15 years):")
print(recent_years.tail(15))
print()

plt.figure(figsize=(10, 5))
plt.plot(recent_years.index, recent_years.values, marker="o", color="#B81D24")
plt.title("Netflix Catalog: Titles by Release Year (1990+)")
plt.xlabel("Release Year")
plt.ylabel("Number of Titles")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "titles_by_release_year.png"), dpi=150)
plt.close()

print(f"Done! Charts saved in the '{OUTPUT_FOLDER}' folder.")