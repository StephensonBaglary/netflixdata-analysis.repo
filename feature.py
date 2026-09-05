"""
02_feature_engineering.py
--------------------------
Turns the messy multi-value columns (listed_in, cast, country) into clean,
structured features we can actually use for modeling and analysis.

What this does, step by step:
1. Load netflix_titles.csv.
2. Genres ("listed_in"): pick a single "primary_genre" (the first genre
   listed) + create one binary column per top genre (multi-hot encoding).
3. Countries ("country"): same idea — "primary_country" + binary columns
   for the top countries.
4. Cast ("cast"): count how many cast members are listed, and create
   binary "has_<actor>" columns for the most frequently appearing actors.
5. Save the result as netflix_titles_features.csv.

HOW TO RUN:
Put this script and "netflix_titles.csv" in the SAME folder, then run:
    python 02_feature_engineering.py
"""

import pandas as pd

DATA_FILE = "/Users/stephensonbaglary/Downloads/github_projects/netflix-data-analysis/data/netflix_titles.csv"
OUTPUT_FILE = "netflix_titles_features.csv"

TOP_N_GENRES = 15
TOP_N_COUNTRIES = 10
TOP_N_ACTORS = 20


def split_and_strip(value: str) -> list:
    """Turn 'Dramas, International Movies' into ['Dramas', 'International Movies']."""
    if pd.isna(value):
        return []
    return [piece.strip() for piece in value.split(",")]


# ---------------------------------------------------------------------------
# STEP 1: Load data
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_FILE)

# Break each multi-value column into a real Python list first — this list
# form is what steps 2-4 all build on.
df["genre_list"] = df["listed_in"].apply(split_and_strip)
df["country_list"] = df["country"].apply(split_and_strip)
df["cast_list"] = df["cast"].apply(split_and_strip)

# ---------------------------------------------------------------------------
# STEP 2: Genre features
# ---------------------------------------------------------------------------
df["primary_genre"] = df["genre_list"].apply(lambda genres: genres[0] if genres else "Unknown")

top_genres = (
    df["genre_list"].explode().dropna().value_counts().head(TOP_N_GENRES).index
)
for genre in top_genres:
    column_name = "genre_" + genre.replace(" ", "_").replace("&", "and")
    df[column_name] = df["genre_list"].apply(lambda genres: 1 if genre in genres else 0)

# ---------------------------------------------------------------------------
# STEP 3: Country features
# ---------------------------------------------------------------------------
df["primary_country"] = df["country_list"].apply(lambda countries: countries[0] if countries else "Unknown")

top_countries = (
    df["country_list"].explode().dropna().value_counts().head(TOP_N_COUNTRIES).index
)
for country in top_countries:
    column_name = "country_" + country.replace(" ", "_")
    df[column_name] = df["country_list"].apply(lambda countries: 1 if country in countries else 0)

# ---------------------------------------------------------------------------
# STEP 4: Cast features
# ---------------------------------------------------------------------------
df["cast_count"] = df["cast_list"].apply(len)

top_actors = (
    df["cast_list"].explode().dropna().value_counts().head(TOP_N_ACTORS).index
)
for actor in top_actors:
    column_name = "actor_" + actor.replace(" ", "_").replace(".", "")
    df[column_name] = df["cast_list"].apply(lambda cast: 1 if actor in cast else 0)

# ---------------------------------------------------------------------------
# STEP 5: Save the engineered dataset
# ---------------------------------------------------------------------------
df.to_csv(OUTPUT_FILE, index=False)

print("=" * 60)
print("FEATURE ENGINEERING SUMMARY")
print("=" * 60)
print(f"Rows: {len(df)}")
print(f"New genre columns: {len(top_genres)}")
print(f"New country columns: {len(top_countries)}")
print(f"New actor columns: {len(top_actors)}")
print()
print("Primary genre breakdown (top 10):")
print(df["primary_genre"].value_counts().head(10))
print()
print(f"Saved engineered dataset to: {OUTPUT_FILE}")