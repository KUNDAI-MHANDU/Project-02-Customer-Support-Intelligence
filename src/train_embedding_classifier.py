from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

dataset = load_dataset("mteb/banking77")

model = SentenceTransformer("all-MiniLM-L6-v2")
classifier = LinearSVC()

X_train_text = dataset["train"]["text"]
y_train = dataset["train"]["label"]

X_test_text = dataset["test"]["text"]
y_test = dataset["test"]["label"]

X_train = model.encode(
    X_train_text,
    show_progress_bar=True
)

X_test = model.encode(
    X_test_text,
    show_progress_bar=True
)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

report = classification_report(
    y_test,
    y_pred,
    output_dict=True,
    zero_division=0
)

print("Training embeddings shape:", X_train.shape)
print("Test embeddings shape:", X_test.shape)

print("Accuracy:", accuracy)
print("Macro F1:", report["macro avg"]["f1-score"])
print("Weighted F1:", report["weighted avg"]["f1-score"])