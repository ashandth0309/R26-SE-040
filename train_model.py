import json
import pickle
import os

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


INTENTS_FILE = "intents.json"
MODEL_PATH = "model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"


def load_dataset(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"{filepath} not found.")

    with open(filepath, "r", encoding="utf-8") as file:
        data = json.load(file)

    sentences = []
    labels = []

    for intent_block in data.get("intents", []):
        intent_name = intent_block.get("intent", "").strip()

        for example in intent_block.get("examples", []):
            example = example.lower().strip()

            if intent_name and example:
                sentences.append(example)
                labels.append(intent_name)

    if not sentences:
        raise ValueError("No training examples found in intents.json.")

    return sentences, labels


def train():
    print("=" * 50)
    print("Training Buddy Intent Classifier")
    print("=" * 50)

    sentences, labels = load_dataset(INTENTS_FILE)

    print(f"Loaded {len(sentences)} examples.")
    print(f"Loaded {len(set(labels))} intents.")

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1
    )

    x_train = vectorizer.fit_transform(sentences)

    model = LogisticRegression(
        C=2.0,
        max_iter=500,
        solver="lbfgs"
    )

    unique_labels = sorted(set(labels))

    if len(unique_labels) >= 2 and len(sentences) >= 10:
        smallest_class_count = min(labels.count(label) for label in unique_labels)
        cv_folds = min(5, smallest_class_count)

        if cv_folds >= 2:
            scores = cross_val_score(model, x_train, labels, cv=cv_folds, scoring="accuracy")
            print(f"Cross-validation accuracy: {np.mean(scores) * 100:.1f}%")
        else:
            print("Skipping cross-validation because some intents have too few examples.")

    model.fit(x_train, labels)

    with open(MODEL_PATH, "wb") as model_file:
        pickle.dump(model, model_file)

    with open(VECTORIZER_PATH, "wb") as vectorizer_file:
        pickle.dump(vectorizer, vectorizer_file)

    print(f"Saved {MODEL_PATH}")
    print(f"Saved {VECTORIZER_PATH}")
    print("Training complete. Now run: python speech.py")


if __name__ == "__main__":
    train()