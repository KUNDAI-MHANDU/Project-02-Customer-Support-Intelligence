from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import joblib
from sklearn.metrics import classification_report

# Load the banking77 dataset
dataset = load_dataset("mteb/banking77")

# Create a SentenceTransformer model and a LinearSVC classifier
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
base_classifier = LinearSVC()

# Create a CalibratedClassifierCV classifier using the LinearSVC classifier as the base classifier
classifier = CalibratedClassifierCV(base_classifier, method="sigmoid", cv=5)

# Extract the training and test data from the dataset
X_train_text = dataset["train"]["text"]
y_train = dataset["train"]["label"]
X_test_text = dataset["test"]["text"]
y_test = dataset["test"]["label"]

# Encode the training and test data into embeddings using the SentenceTransformer model
X_train = embedding_model.encode(
    X_train_text,
    show_progress_bar=True)

X_test = embedding_model.encode(
    X_test_text,
    show_progress_bar=True)

# Fit the CalibratedClassifierCV classifier to the training embeddings and make predictions on the test embeddings
classifier.fit(X_train, y_train)

# Save the trained CalibratedClassifierCV classifier to a file using joblib
joblib.dump(
    classifier,
    "models/calibrated_intent_classifier.joblib"
)

# Make predictions on the test embeddings
y_pred = classifier.predict(X_test)

# Calculate the accuracy and generate a classification report for the predictions
report = classification_report(
    y_test,
    y_pred,
    output_dict=True,
    zero_division=0
)

# Test the trained classifier on a new message
message = "my transaction hasn't gone through, how come?"

# Encode the message into embeddings using the SentenceTransformer model and make predictions on the embeddings
message_embedding = embedding_model.encode([message])
message_prediction = classifier.predict(message_embedding)[0]

# Get the predicted probabilities for each class for the message
message_probability = classifier.predict_proba(message_embedding)[0]

# Load the label map from a file using joblib
label_map = joblib.load("models/label_map.joblib")

# Print the accuracy, macro F1 score, and predicted intent for the new message
print(f"Calibrated model accuracy: {report['accuracy']:.4f}")
print(f"Calibrated model macro F1: {report['macro avg']['f1-score']:.4f}")
print(f"Predicted intent for a new message {label_map.get(message_prediction)}")
print(f"\nmessage1: {message}")

# Get the top 3 predicted intents and their probabilities for the new message
top_indices = message_probability.argsort()[-3:][::-1]
for index in top_indices:
    label_id = classifier.classes_[index]

    print(
        label_map[label_id],
        f"{message_probability[index]:.2%}"
    )

# Test the trained classifier on another new message
message = "The person I sent money to yesterday still has not received it"
message_embedding = embedding_model.encode([message])
message_prediction = classifier.predict(message_embedding)[0]
message_probability = classifier.predict_proba(message_embedding)[0]
print(f"Predicted intent for a new message {label_map.get(message_prediction)}")
print(f"\nmessage1: {message}")

# Get the top 3 predicted intents and their probabilities for the new message
top_indices = message_probability.argsort()[-3:][::-1]

for index in top_indices:
    label_id = classifier.classes_[index]

    print(
        label_map[label_id],
        f"{message_probability[index]:.2%}"
    )