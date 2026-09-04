from sentence_transformers import SentenceTransformer
import joblib


# Load the trained SentenceTransformer model, the trained CalibratedClassifierCV classifier, and the label map from their respective files
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
classifier = joblib.load("models/calibrated_intent_classifier.joblib")
label_map = joblib.load("models/label_map.joblib")

# Define a function to predict the intent of a given message using the trained classifier and the label map
def predict_intent(message):

    # Encode the message into embeddings using the SentenceTransformer model and make predictions on the embeddings
    message_embedding = embedding_model.encode([message])

    # Get the predicted class and the predicted probabilities for each class for the message
    prediction = classifier.predict(message_embedding)[0]
    probabilities = classifier.predict_proba(message_embedding)[0]

    # Get the position of the predicted class in the list of classes and calculate the confidence score for the prediction
    class_position = classifier.classes_.tolist().index(prediction)

    # Get the confidence score for the predicted class from the predicted probabilities
    confidence = float(probabilities[class_position])

    # Get the predicted category from the label map using the predicted class
    predicted_category = label_map.get(prediction)

    # Return the predicted category and the confidence score as a dictionary
    return {
        "intent": predicted_category,
        "confidence": confidence
    }

# Test the predict_intent function with a sample message
if __name__ == "__main__":
    message = "The person I sent money to yesterday still has not received it"

    result = predict_intent(message)

    print(f"Customer message: {message}")
    print(f"Predicted category: {result['intent']}")
    print(f"Confidence: {result['confidence']:.2%}")