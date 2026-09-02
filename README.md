# Customer Support Intelligence System

## Project Overview

An AI-powered system that will classify customer requests, determine the priority of each request, provide a confidence score, and assist customer support teams in handling customer issues.

## Current Goal

The current goal is to classify customer-support messages into the correct intent/category.

## Dataset

- Dataset: Banking77
- Training examples: 9,993
- Test examples: 3,076
- Number of categories: 77

## Current Approach

Customer-support messages are converted into numerical features using TF-IDF.

The project currently compares two machine-learning classifiers:

- Logistic Regression
- Linear SVM

## Current Results

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| Logistic Regression | 87.78% | 87.77% |
| Linear SVM | 89.47% | 89.45% |

Linear SVM is currently the best-performing classical machine-learning model.

## Experiments

I tested both Logistic Regression and Linear SVM using two TF-IDF configurations:

1. Unigrams
2. Unigrams + bigrams

| Model | TF-IDF Configuration | Accuracy | Macro F1 |
|---|---|---:|---:|
| Logistic Regression | Unigrams | 87.78% | 87.77% |
| Logistic Regression | Unigrams + Bigrams | 85.53% | 85.38% |
| Linear SVM | Unigrams | 89.47% | 89.45% |
| Linear SVM | Unigrams + Bigrams | 89.37% | 89.37% |

Adding bigrams increased the number of TF-IDF features from 2,319 to 23,605.

The additional bigram features did not improve either model. Logistic Regression performance decreased noticeably, while Linear SVM performance remained very similar but slightly lower.

Because unigram TF-IDF produced fewer features while also achieving the best overall performance, I selected unigram TF-IDF with Linear SVM as the current classical machine-learning baseline.

## What I Learned

I learned that machine-learning models cannot directly process customer messages as text, so the text first needs to be converted into numerical features.

I learned how Bag of Words and TF-IDF represent text numerically and that TF-IDF produces a sparse matrix, meaning that most of the values in the feature matrix are zero.

For this dataset, the TF-IDF training data contained 9,993 rows and 2,319 features, creating high-dimensional sparse data.

I also learned that Linear SVM can work very well for text-classification problems involving high-dimensional sparse features. In this project, it performed better than Logistic Regression.

I learned that accuracy alone is not enough when evaluating a classifier. I also used precision, recall, F1-score, Macro F1, and error analysis to understand where the model was making mistakes.

Error analysis showed that the model sometimes struggles to distinguish between closely related customer intents, such as:

- `balance_not_updated_after_bank_transfer`
- `pending_transfer`
- `transfer_not_received_by_recipient`
- `transfer_timing`

## Next Steps

- Test pretrained text embeddings
- Compare pretrained embeddings with TF-IDF
- Build an API using FastAPI
- Store customer-support tickets in a database
- Add priority classification
- Add confidence scoring
- Add LLM/RAG capabilities
- Deploy the system