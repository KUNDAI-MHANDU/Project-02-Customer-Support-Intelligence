from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the banking77 dataset
dataset = load_dataset("mteb/banking77")

# Create a SentenceTransformer model and a LinearSVC classifier
model = SentenceTransformer("all-MiniLM-L6-v2")
classifier = LinearSVC()

# Extract the training and test data from the dataset
X_train_text = dataset["train"]["text"]
y_train = dataset["train"]["label"]

# Extract the test data from the dataset
X_test_text = dataset["test"]["text"]
y_test = dataset["test"]["label"]

# Encode the training and test data into embeddings using the SentenceTransformer model
X_train = model.encode(
    X_train_text,
    show_progress_bar=True
)

X_test = model.encode(
    X_test_text,
    show_progress_bar=True
)

# Fit the LinearSVC classifier to the training embeddings and make predictions on the test embeddings
classifier.fit(X_train, y_train)

# Save the trained LinearSVC classifier to a file using joblib
joblib.dump(classifier, "models/intent_classifier.joblib")

# Make predictions on the test embeddings
y_pred = classifier.predict(X_test)

# Calculate the accuracy and generate a classification report for the predictions
accuracy = accuracy_score(y_test, y_pred)

# Generate a classification report for the predictions
report = classification_report(
    y_test,
    y_pred,
    output_dict=True,
    zero_division=0
)

# Print the shapes of the training and test embeddings, as well as the accuracy and F1 scores
print("Training embeddings shape:", X_train.shape)
print("Test embeddings shape:", X_test.shape)

# Print the accuracy and F1 scores for the predictions
print("Accuracy:", accuracy)
print("Macro F1:", report["macro avg"]["f1-score"])
print("Weighted F1:", report["weighted avg"]["f1-score"])

# Create a label map to map label indices to label texts
label_map = dict(
    zip(
        dataset["train"]["label"],
        dataset["train"]["label_text"]
    )
)

# Create a list to store the F1 scores for each category and sort them by the SVM F1 score
category_scores = []
for label_id in range(77):

    category_name = label_map[label_id]

    f1_score = report[str(label_id)]["f1-score"]

    category_scores.append((category_name, label_id, f1_score))

# Sort the category scores by the SVM F1 score in ascending order
sorted_scores = sorted(
    category_scores,
    key=lambda item: item[2]
)

# Print the top 5 categories with the lowest SVM F1 scores
print("\nTop 5 categories with the lowest SVM F1 scores:")
for category in sorted_scores[:5]:
    print(category)

# Print 5 mistakes made by the SVM model for the category "balance_not_updated_after_bank_transfer"
svm_mistakes_found = 0
print("\n5 SVM mistakes for balance_not_updated_after_bank_transfer:")
for i in range(len(y_test)):

    if y_test[i]==5 and y_pred[i]!=5:

        print("\nMessage:", X_test_text[i])
        print("Actual:", label_map[y_test[i]])
        print("Predicted:", label_map[y_pred[i]])

        svm_mistakes_found += 1

    if svm_mistakes_found == 5:
        break

new_message = [
    "My virtual card keeps getting declined when I try to use it"
]

# Encode the new message into embeddings using the SentenceTransformer model
new_message_embedding = model.encode(new_message)

# Make a prediction for the new message using the trained LinearSVC classifier
new_message_prediction = classifier.predict(new_message_embedding)

# Print the predicted category for the new message
print("\nPredicted category for the new message:")
print("\nMessage:", new_message[0])
print(label_map[new_message_prediction[0]])