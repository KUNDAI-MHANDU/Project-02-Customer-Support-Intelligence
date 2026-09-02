from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC

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

# Create instances of LogisticRegression and LinearSVC models
logistic_model = LogisticRegression(max_iter=1000)
svm_model = LinearSVC()

# Fit the models to the training data
logistic_model.fit(X_train, y_train)
svm_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred_logistic = logistic_model.predict(X_test)
y_pred_svm = svm_model.predict(X_test)

# Calculate the accuracy of the models on the test data
accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
accuracy_svm = accuracy_score(y_test, y_pred_svm)

# Print the accuracy of the models on the test data
print(f"Accuracy on the test set: {accuracy_logistic:.4f}")
print(f"Accuracy on the test set (SVM): {accuracy_svm:.4f}")


# Generate classification reports for both models
report_logistic = classification_report(
    y_test,
    y_pred_logistic,
    output_dict=True,
    zero_division=0
)

svm_report = classification_report(
    y_test,
    y_pred_svm,
    output_dict=True,
    zero_division=0
)

# Print the classification report keys and some metrics for both models
print("\nLogistic Regression Classification Report Keys:")
print(report_logistic.keys())
print(report_logistic["accuracy"])
print(report_logistic["macro avg"])
print(report_logistic["weighted avg"])

print("\nSVM Classification Report Keys:")
print(svm_report.keys())
print(svm_report["accuracy"])
print(svm_report["macro avg"])
print(svm_report["weighted avg"])

# Create a label map to convert label IDs to label texts
label_map = dict(
    zip(
        dataset["train"]["label"],
        dataset["train"]["label_text"]
    )
)

# Print a few examples of customer messages and their corresponding categories
print("\nCustomer message:", X_test_text[0])
print("Actual category:", label_map[y_test[0]])
print("Logistic Regression category:", label_map[y_pred_logistic[0]])
print("SVM category:", label_map[y_pred_svm[0]])


# Compare the F1-scores for each category for both models
logistic_category_scores = []
svm_category_scores = []

for label_id in range(77):

    category_name = label_map[label_id]

    logistic_f1_score = report_logistic[str(label_id)]["f1-score"]
    svm_f1_score = svm_report[str(label_id)]["f1-score"]

    logistic_category_scores.append(
        (category_name, label_id, logistic_f1_score)
    )

    svm_category_scores.append(
        (category_name, label_id, svm_f1_score)
    )   


logistic_sorted_scores = sorted(
    logistic_category_scores,
    key=lambda item: item[2]
)

svm_sorted_scores = sorted(
    svm_category_scores,
    key=lambda item: item[2]
)

# Print the 5 categories with the lowest F1-scores for both models
print("\n5 logistic categories with lowest F1-score:")
for category in logistic_sorted_scores[:5]:
    print(category)

# Print the 5 categories with the lowest F1-scores for SVM
print("\n5 SVM categories with lowest F1-score:")
for category in svm_sorted_scores[:5]:
    print(category)


# Print 5 mistakes made by the Logistic Regression model for the category "virtual_card_not_working"
logistic_mistakes_found = 0
print("\n5 Logistic Regression mistakes for virtual_card_not_working:")
for i in range(len(y_test)):

    if y_test[i]==72 and y_pred_logistic[i]!=72:

        print("\nMessage:", X_test_text[i])
        print("Actual:", label_map[y_test[i]])
        print("Predicted:", label_map[y_pred_logistic[i]])

        logistic_mistakes_found += 1

    if logistic_mistakes_found == 5:
        break

# Print 5 mistakes made by the SVM model for the category "balance_not_updated_after_bank_transfer"
svm_mistakes_found = 0
print("\n5 SVM mistakes for balance_not_updated_after_bank_transfer:")
for i in range(len(y_test)):

    if y_test[i]==5 and y_pred_svm[i]!=5:

        print("\nMessage:", X_test_text[i])
        print("Actual:", label_map[y_test[i]])
        print("Predicted:", label_map[y_pred_svm[i]])

        svm_mistakes_found += 1

    if svm_mistakes_found == 5:
        break