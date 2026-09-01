from sklearn.feature_extraction.text import TfidfVectorizer

# Define a list of sentences to vectorize
sentences = [
    "need my card",
    "my card is late",
    "need help now"
]

# Create a TfidfVectorizer instance
vectorizer = TfidfVectorizer()

# Fit the vectorizer to the sentences and transform them into TF-IDF features
X = vectorizer.fit_transform(sentences)

# TF-IDF feature matrix
print(vectorizer.get_feature_names_out())
print(X.toarray())