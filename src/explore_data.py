from datasets import load_dataset
from collections import Counter

# Load the banking77 dataset
dataset = load_dataset("mteb/banking77")

# Get unique label_texts
label_texts = set(dataset["train"]["label_text"])

# Print the unique label_texts and their count
print(label_texts)
print(f"Number of unique label_texts: {len(label_texts)}")

# Count the occurrences of each label_text in the training set
label_counts = Counter(dataset["train"]["label_text"])

# Print the label counts
print(f"Label counts:\n{label_counts}")

# Print a few examples of customer messages and their corresponding categories
seen_examples = set()
for example in dataset["train"]:
    if example["label_text"] not in seen_examples:
        print(f"customer message: {example['text']}")
        print(f"category: {example['label_text']}")
        seen_examples.add(example["label_text"])
    if len(seen_examples    ) == 5:
        break
