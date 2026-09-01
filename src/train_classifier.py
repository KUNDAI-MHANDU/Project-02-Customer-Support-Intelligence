from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer

dataset = load_dataset("mteb/banking77")

# Extract the text data from the dataset
X_train_text = dataset["train"]["text"]
y_train = dataset["train"]["label"]

X_test_text = dataset["test"]["text"]
y_test = dataset["test"]["label"]

vectorizer = TfidfVectorizer()

X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

print("Training data shape:", X_train.shape)
print("Test data shape:", X_test.shape)