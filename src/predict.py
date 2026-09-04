from sentence_transformers import SentenceTransformer
import joblib


embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
classifier = joblib.load("models/intent_classifier.joblib")
label_map = joblib.load("models/label_map.joblib")


def predict_intent(message):
    message_embedding = embedding_model.encode([message])

    prediction = classifier.predict(message_embedding)

    predicted_category = label_map.get(prediction[0])

    return predicted_category


message = "my transaction hasn't gone through, how come?"
predicted_category = predict_intent(message)

print(f"Customer message: {message}")
print(f"Predicted category: {predicted_category}")