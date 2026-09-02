from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# Load the banking77 dataset
dataset = load_dataset("mteb/banking77")

# Extract the training data from the dataset
X_train_text = dataset["train"]["text"]
y_train = dataset["train"]["label"]

# Extract the test data from the dataset
X_test_text = dataset["test"]["text"]
y_test = dataset["test"]["label"]

# Create a TfidfVectorizer instance
vectorizer = TfidfVectorizer()

# Fit the vectorizer to the training data and transform both training and test data into TF-IDF features
X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

# Print the shapes of the training and test data
print("Training data shape:", X_train.shape)
print("Test data shape:", X_test.shape)

# Train a logistic regression model on the training data
model = LogisticRegression(max_iter=1000)

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate the accuracy of the model on the test data
accuracy = accuracy_score(y_test, y_pred)

# Print the first 5 actual and predicted labels, and the accuracy
print(y_test[:5], y_pred[:5])
print(f"Accuracy on the test set: {accuracy:.4f}")

# Generate a classification report for the model's performance on the test data
report = classification_report(
    y_test,
    y_pred,
    output_dict=True,
    zero_division=0
)

# Print the classification report keys and specific metrics
print(report.keys())
print(report["accuracy"])
print(report["macro avg"])
print(report["weighted avg"])

# Create a label map to convert label indices to label texts
label_map = dict(
    zip(
        dataset["train"]["label"],
        dataset["train"]["label_text"]
    )
)

# Print a few examples of customer messages, their actual categories, and the predicted categories
print("\nCustomer message:", X_test_text[0])
print("Actual category:", label_map[y_test[0]])
print("Predicted category:", label_map[y_pred[0]])