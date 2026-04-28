"""
train_model.py — Model Training Script
=======================================
Set 4 Tuesday  — Created dataset (intents.json) with example sentences per intent.
                  Applied TF-IDF to convert text to numbers.
                  Used Logistic Regression as the classification algorithm.
Set 5 Monday   — Expanded dataset with more examples per intent.
                  Tuned model parameters (C=2.0, max_iter=300) for better accuracy.
Set 5 Thursday — Verified training pipeline produces correct model.pkl and vectorizer.pkl.

HOW TO RUN:
  python train_model.py

OUTPUT FILES:
  model.pkl       — Trained Logistic Regression classifier
  vectorizer.pkl  — Fitted TF-IDF vectorizer (must match what ml_brain.py uses)

WHY THESE CHOICES (viva explanation):
  - intents.json  : Teaches the model different ways users say the same thing.
  - TF-IDF        : Converts human language into numbers. Weights rare words higher
                    so meaningful keywords have more impact than common words.
  - Logistic Reg. : Simple, fast, and works well for multi-class text classification.
                    Does not overfit on small datasets like neural networks might.
  - C=2.0         : Regularization strength. Higher = less regularization = better
                    fit on training data. Tuned in Set 5 Monday.
  - max_iter=300  : Enough iterations for the solver to converge reliably.
"""

import json
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import numpy as np

# ==========================================
# FILE PATHS
# ==========================================
INTENTS_FILE    = "intents.json"
MODEL_PATH      = "model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"

# ==========================================
# STEP 1 — LOAD DATASET
# Set 4 Tuesday — intents.json contains example sentences per intent
# Set 5 Monday  — expanded with more examples for better accuracy
# ==========================================
def load_dataset(filepath):
    """
    Reads intents.json and returns two parallel lists:
      sentences — all example input strings
      labels    — the intent label for each sentence

    Example:
      "my name is ashan"  → "set_name"
      "what time is it"   → "time"
      "tell me a joke"    → "joke"
    """
    sentences = []
    labels    = []

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    for intent_block in data["intents"]:
        intent_label = intent_block["intent"]
        for example in intent_block["examples"]:
            sentences.append(example.lower().strip())
            labels.append(intent_label)

    print(f"✅ Dataset loaded: {len(sentences)} examples across {len(set(labels))} intents.")
    return sentences, labels

# ==========================================
# STEP 2 — TF-IDF VECTORIZATION
# Set 4 Tuesday — TF-IDF converts raw text into numerical feature vectors
#
# WHY TF-IDF (viva):
#   TF  = Term Frequency  — how often a word appears in the sentence
#   IDF = Inverse Document Frequency — penalises words that appear in
#         every sentence (like "i", "the") because they carry less meaning.
#   Result: meaningful words like "joke", "time", "name" get higher weight.
#
# ngram_range=(1,2) means we also look at two-word combos:
#   "my name", "call me", "what time" — better for intent matching.
# ==========================================
def build_vectorizer():
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),    # unigrams + bigrams (Set 5 Monday tuning)
        sublinear_tf=True,     # apply log normalization to term frequency
        min_df=1               # include words that appear at least once
    )
    return vectorizer

# ==========================================
# STEP 3 — TRAIN LOGISTIC REGRESSION MODEL
# Set 4 Tuesday  — baseline model with default parameters
# Set 5 Monday   — tuned C=2.0 and max_iter=300 for better accuracy
#
# WHY LOGISTIC REGRESSION (viva):
#   - Simple and fast to train on small text datasets.
#   - Outputs probability scores (used for confidence threshold in ml_brain.py).
#   - Works well for multi-class classification (one intent from many).
#   - Less likely to overfit compared to complex models like neural networks
#     when training data is limited.
# ==========================================
def build_model():
    model = LogisticRegression(
        C=2.0,           # Set 5 Monday — reduced regularization for better fit
        max_iter=300,    # Set 5 Monday — enough iterations to converge
        solver="lbfgs"   # efficient for multi-class problems (handles multiple intents natively)
    )
    return model

# ==========================================
# STEP 4 — EVALUATE WITH CROSS-VALIDATION
# Set 5 Monday — added cross-validation to measure real accuracy
# ==========================================
def evaluate_model(model, X, labels):
    """
    Runs 5-fold cross-validation to estimate real-world accuracy.
    Prints mean accuracy and standard deviation.
    """
    scores = cross_val_score(model, X, labels, cv=5, scoring="accuracy")
    print(f"\n📊 Cross-Validation Accuracy: {np.mean(scores)*100:.1f}% "
          f"(±{np.std(scores)*100:.1f}%)")
    if np.mean(scores) < 0.75:
        print("⚠️  Accuracy is below 75%. Consider adding more examples to intents.json.")
    else:
        print("✅ Accuracy looks good! Model is ready.")

# ==========================================
# STEP 5 — SAVE MODEL AND VECTORIZER
# Set 4 Tuesday — saved with pickle so ml_brain.py can load them
# ==========================================
def save_assets(model, vectorizer):
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"\n✅ Saved: {MODEL_PATH}")
    print(f"✅ Saved: {VECTORIZER_PATH}")

# ==========================================
# MAIN TRAINING PIPELINE
# ==========================================
def train():
    print("=" * 50)
    print("  Training ML Intent Classifier")
    print("  Usha — Offline Voice Assistant")
    print("=" * 50)

    # Load data
    sentences, labels = load_dataset(INTENTS_FILE)

    # Build and fit vectorizer
    vectorizer = build_vectorizer()
    X = vectorizer.fit_transform(sentences)
    print(f"✅ TF-IDF features: {X.shape[1]} dimensions")

    # Build and evaluate model (cross-validation before final fit)
    model = build_model()
    evaluate_model(model, X, labels)

    # Final fit on ALL data (after evaluation)
    model.fit(X, labels)
    print(f"\n✅ Model trained on {X.shape[0]} examples.")
    print(f"   Intents: {sorted(set(labels))}")

    # Save to disk
    save_assets(model, vectorizer)

    print("\n🎉 Training complete! You can now run speech.py.")
    print("=" * 50)

if __name__ == "__main__":
    train()