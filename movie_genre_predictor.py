"""
Movie Genre Predictor
=====================
Trains TF-IDF + Naive Bayes / Logistic Regression / Linear SVM models
to predict a movie's genre from its plot description.

Usage
-----
1. Train:
   python movie_genre_predictor.py --train movies.csv

2. Predict:
   python movie_genre_predictor.py --predict "A scientist discovers a portal to another dimension..."
"""

import argparse, json, joblib, warnings
import pandas as pd
import numpy as np
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report


# ── Train ─────────────────────────────────────────────────────────────────────
def train(csv_path: str, top_n: int = 10):
    df = pd.read_csv(csv_path).dropna(subset=['description', 'genre'])
    df['genre'] = df['genre'].str.strip().str.lower()

    top_genres = df['genre'].value_counts().head(top_n).index.tolist()
    df = df[df['genre'].isin(top_genres)].copy()
    print(f"Training on {len(df):,} samples across {len(top_genres)} genres.")

    X_train, X_test, y_train, y_test = train_test_split(
        df['description'], df['genre'],
        test_size=0.2, random_state=42, stratify=df['genre']
    )

    tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2),
                            stop_words='english', sublinear_tf=True)
    X_tr = tfidf.fit_transform(X_train)
    X_te = tfidf.transform(X_test)

    candidates = {
        "Naive Bayes":         MultinomialNB(alpha=0.1),
        "Logistic Regression": LogisticRegression(max_iter=1000, C=5, random_state=42),
        "Linear SVM":          LinearSVC(C=1.0, max_iter=2000, random_state=42),
    }

    best_acc, best_model, best_name = 0, None, ""
    for name, model in candidates.items():
        model.fit(X_tr, y_train)
        acc = accuracy_score(y_test, model.predict(X_te))
        print(f"  {name:25s}  Accuracy: {acc:.4f}")
        if acc > best_acc:
            best_acc, best_model, best_name = acc, model, name

    print(f"\nBest: {best_name} ({best_acc:.4f})")
    print(classification_report(y_test, best_model.predict(X_te), target_names=top_genres))

    joblib.dump(best_model, "best_model.pkl")
    joblib.dump(tfidf,      "tfidf.pkl")
    with open("genres.json", "w") as f:
        json.dump(top_genres, f)
    print("Saved: best_model.pkl, tfidf.pkl, genres.json")


# ── Predict ───────────────────────────────────────────────────────────────────
def predict(description: str):
    model = joblib.load("best_model.pkl")
    tfidf = joblib.load("tfidf.pkl")
    vec   = tfidf.transform([description])
    genre = model.predict(vec)[0]

    # Probability scores (not available for LinearSVC directly)
    try:
        probs = model.predict_proba(vec)[0]
        genres = json.load(open("genres.json"))
        top3 = sorted(zip(genres, probs), key=lambda x: -x[1])[:3]
        print(f"\nPredicted genre: {genre.upper()}")
        print("Top 3 probabilities:")
        for g, p in top3:
            print(f"  {g:15s} {p:.2%}")
    except AttributeError:
        print(f"\nPredicted genre: {genre.upper()}")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Movie Genre Predictor")
    parser.add_argument("--train",   metavar="CSV",  help="Path to movies.csv")
    parser.add_argument("--predict", metavar="TEXT", help="Plot description to classify")
    parser.add_argument("--top_n",   type=int, default=10, help="Number of top genres to use")
    args = parser.parse_args()

    if args.train:
        train(args.train, top_n=args.top_n)
    elif args.predict:
        predict(args.predict)
    else:
        parser.print_help()
