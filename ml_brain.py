import pickle
import os

# ==========================================
# FILE PATHS
# ==========================================
MODEL_PATH      = "model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"

# ==========================================
# LOAD MODEL AND VECTORIZER AT STARTUP
# Set 4 Wednesday — loaded once when module is imported (efficient)
# ==========================================
model      = None
vectorizer = None

def _load_assets():
    """
    Loads the trained Logistic Regression model and TF-IDF vectorizer from disk.
    Called automatically when this module is imported.
    If files are missing, model stays None and predict_intent() returns "unknown".
    """
    global model, vectorizer
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(VECTORIZER_PATH, "rb") as f:
            vectorizer = pickle.load(f)
        print("✅ ML model loaded successfully.")
    except FileNotFoundError:
        print("⚠️  model.pkl or vectorizer.pkl not found.")
        print("   Run train_model.py first to generate these files.")
    except Exception as e:
        print(f"⚠️  Failed to load ML model: {e}")

_load_assets()

# ==========================================
# INTENT PREDICTION
# Set 4 Thursday — used by speech.py in hybrid approach
# Set 5 Tuesday  — confidence threshold added (< 0.6 → "unknown")
# ==========================================
def predict_intent(text):
    """
    Predicts the intent of a given input string.

    Steps:
      1. Check model is loaded (if not, return "unknown")
      2. Transform text to TF-IDF vector (same as during training)
      3. Get probability scores for all classes
      4. If top confidence < 0.6, return "unknown" (Set 5 Tuesday)
      5. Otherwise return the predicted intent label

    WHY CONFIDENCE THRESHOLD (viva explanation):
      The model might guess wrong with high certainty on unusual input.
      By requiring ≥ 60% confidence, we ensure only reliable predictions
      are used. Anything below 60% falls back to the rule-based system
      in speech.py, which is more reliable for edge cases.

    Args:
        text (str): User's speech input (already lowercased in speech.py)

    Returns:
        str: Intent label e.g. "greeting", "time", "joke", or "unknown"
    """
    if model is None or vectorizer is None:
        return "unknown"

    # Step 1 — Convert text to TF-IDF numerical vector
    # WHY TF-IDF: ML models cannot process raw text. TF-IDF converts words
    # into numbers that represent how important each word is in the sentence.
    X = vectorizer.transform([text])

    # Step 2 — Get confidence scores for all intent classes
    probs = model.predict_proba(X)[0]
    top_confidence = max(probs)

    # Step 3 — Apply confidence threshold (Set 5 Tuesday)
    if top_confidence < 0.6:
        return "unknown"

    # Step 4 — Return the predicted intent
    return model.predict(X)[0]