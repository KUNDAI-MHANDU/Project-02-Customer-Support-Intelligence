# Customer Support Intelligence System

## Project Overview

An AI-powered system that will classify customer requests, determine the priority of each request, provide a confidence score, and assist customer support teams in handling customer issues.

## Current Goal

The current goal is to build a production-style customer-support intelligence service that classifies incoming customer messages, returns a calibrated confidence score, stores tickets and predictions in PostgreSQL, and exposes the system through a REST API.

## Dataset

- Dataset: Banking77
- Training examples: 9,993
- Test examples: 3,076
- Number of categories: 77

## Current Approach

The project currently uses pretrained sentence embeddings generated using `all-MiniLM-L6-v2` to represent customer-support messages as 384-dimensional dense vectors.

A Linear SVM classifier is trained on these embeddings to classify messages into one of the 77 Banking77 intents.

The project also uses `CalibratedClassifierCV` with Linear SVM to generate calibrated probability estimates that can be used as confidence scores.

The trained classifier and label mapping are saved using `joblib`, allowing the application to perform inference without retraining the model every time it starts.

A reusable `predict_intent()` function loads the saved model and returns both the predicted intent and confidence score.

The prediction system is exposed through a FastAPI REST API using a `POST /predict` endpoint.

Predictions are persisted in PostgreSQL using SQLAlchemy, allowing the system to store the original customer message together with its predicted intent, confidence score, and creation timestamp.

## Current Results

| Representation | Classifier | Accuracy | Macro F1 |
|---|---|---:|---:|
| TF-IDF | Logistic Regression | 87.78% | 87.77% |
| TF-IDF | Linear SVM | 89.47% | 89.45% |
| MiniLM Embeddings | Linear SVM | **92.95%** | **92.91%** |
| MiniLM Embeddings | Calibrated Linear SVM | 92.88% | 92.85% |

MiniLM embeddings combined with Linear SVM achieved the highest classification performance.

The calibrated version maintained nearly identical classification performance while also providing calibrated probability estimates that can be used as confidence scores.


## Experiments

### TF-IDF Experiments

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

Because unigram TF-IDF used far fewer features while achieving the best TF-IDF performance, I selected unigram TF-IDF with Linear SVM as the classical machine-learning baseline.

### Pretrained Embedding Experiment

I then replaced TF-IDF with pretrained sentence embeddings generated using `all-MiniLM-L6-v2`.

The same Linear SVM classifier was trained on the 384-dimensional dense sentence embeddings.

| Representation | Classifier | Accuracy | Macro F1 |
|---|---|---:|---:|
| TF-IDF | Logistic Regression | 87.78% | 87.77% |
| TF-IDF | Linear SVM | 89.47% | 89.45% |
| MiniLM Embeddings | Linear SVM | **92.95%** | **92.91%** |
| MiniLM Embeddings | Calibrated Linear SVM | 92.88% | 92.85% |

Using pretrained sentence embeddings improved accuracy from 89.47% to 92.95% compared with the best TF-IDF model.

MiniLM embeddings combined with Linear SVM are currently the best-performing classification approach.

## API

The intent-classification system is exposed through a FastAPI REST API.

### Endpoint

`POST /predict`
The endpoint accepts a customer-support message, stores the ticket and model prediction in PostgreSQL, and returns the generated ticket ID, predicted intent, and calibrated confidence score.

Example request:

```json
{
  "text": "My virtual card keeps getting declined"
}
```

Example response:

```json
{
  "ticket_id": 3,
  "intent": "virtual_card_not_working",
  "confidence": 0.7911729096745604
}
```

### Input Validation

The API validates incoming messages using Pydantic. Empty messages, whitespace-only messages, and messages exceeding the allowed length are rejected before reaching the machine-learning model.


### Automated API Tests

The API currently includes automated tests for:

- Valid request → `200`
- Empty message → `422`
- Whitespace-only message → `422`
- Message over 1000 characters → `422`

All 4 tests currently pass using `pytest`.

The valid-request test also uses a mocked database ticket ID so the API response can be tested without creating a real PostgreSQL record.

## PostgreSQL

The application uses a PostgreSQL database called `customer_support` to persist customer-support tickets and model predictions.

When a customer message is sent to the `/predict` endpoint:

1. The message is classified by the model.
2. The predicted intent and confidence score are generated.
3. The ticket is stored in PostgreSQL using SQLAlchemy.
4. PostgreSQL generates a unique ticket ID.
5. The API returns the ticket ID, intent, and confidence score.

### Tickets Table

The `tickets` table currently contains:

- `id` → `BIGINT` → Primary key, automatically generated
- `text` → `TEXT` → `NOT NULL`
- `predicted_intent` → `VARCHAR(100)` → `NOT NULL`
- `confidence` → `DOUBLE PRECISION` → `NOT NULL`
- `created_at` → `TIMESTAMPTZ` → `NOT NULL`, automatically generated

Example stored ticket:

| id | text | predicted_intent | confidence |
|---:|---|---|---:|
| 3 | My cash withdrawal is still showing as pending | pending_cash_withdrawal | 0.9119 |


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

I learned the difference between sparse TF-IDF vectors and dense pretrained sentence embeddings. Unlike TF-IDF, pretrained embeddings represent the semantic meaning of a sentence rather than mainly relying on word frequency.

In this project, replacing TF-IDF with MiniLM embeddings improved Linear SVM accuracy from 89.47% to 92.95%.

After performing error analysis on the MiniLM embeddings + Linear SVM model, I found that some of the previously weak categories improved compared with the TF-IDF + Linear SVM baseline.

For example:

- `balance_not_updated_after_bank_transfer`
  - TF-IDF + Linear SVM: F1 ≈ 0.72
  - MiniLM + Linear SVM: F1 ≈ 0.79

This showed me that pretrained semantic embeddings can help the classifier better distinguish between customer intents that have similar wording but different meanings.

I learned the difference between model training and inference. Training creates the classifier using the training dataset, while inference loads the saved classifier and uses it to make predictions on new customer messages without retraining.

I learned how to persist trained machine-learning models using `joblib` and load them later for production inference.

I also learned that Linear SVM decision scores are not probabilities. I used `CalibratedClassifierCV` to generate probability-based confidence estimates while maintaining similar classification performance.

I learned how to expose a machine-learning model through a FastAPI REST API, validate incoming JSON requests with Pydantic, return model predictions as JSON responses, and create automated API tests using `pytest`.

I learned how to connect a Python application to PostgreSQL using SQLAlchemy and `psycopg`.

I learned that SQLAlchemy sessions allow the application to interact with the database. In this project:

- `session.add(ticket)` adds a new ticket object to the current session.
- `session.commit()` permanently saves the transaction to PostgreSQL.
- `session.refresh(ticket)` reloads the stored row so Python can access values generated by PostgreSQL, such as the `id` and `created_at` fields.

I learned how to use SQLAlchemy ORM models to represent PostgreSQL tables as Python classes.

I also learned that automated API tests should not insert test data into the real application database. By mocking the database-saving function during API tests, the API behavior can be tested without creating unwanted PostgreSQL records. The tests also verify that `save_ticket()` is called with the correct customer message, predicted intent, and confidence score.

I learned how database-generated values can be used in an API response. After saving a ticket, `session.refresh(ticket)` retrieves values generated by PostgreSQL, including the ticket `id`, which can then be returned to the client through FastAPI.

## Next Steps

- Add endpoints for retrieving stored tickets
- Add priority classification
- Add routing logic for support departments
- Add LLM/RAG capabilities for generating support assistance
- Containerize the application with Docker
- Deploy the API