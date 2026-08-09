import os
import pickle


MODEL_PATH = "model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"

model = None
vectorizer = None


def load_assets():
    global model, vectorizer

    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        print("ML model files not found. Run: python train_model.py")
        return False

    try:
        with open(MODEL_PATH, "rb") as model_file:
            model = pickle.load(model_file)

        with open(VECTORIZER_PATH, "rb") as vectorizer_file:
            vectorizer = pickle.load(vectorizer_file)

        print("ML intent model loaded successfully.")
        return True

    except Exception as error:
        print(f"Failed to load ML model: {error}")
        model = None
        vectorizer = None
        return False


def predict_intent(text, threshold=0.60):
    if model is None or vectorizer is None:
        return "unknown"

    if not text or not text.strip():
        return "unknown"

    try:
        text_vector = vectorizer.transform([text.lower().strip()])
        probabilities = model.predict_proba(text_vector)[0]
        confidence = max(probabilities)

        if confidence < threshold:
            return "unknown"

        return model.predict(text_vector)[0]

    except Exception as error:
        print(f"Intent prediction error: {error}")
        return "unknown"


load_assets()