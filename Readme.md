# IMDB Sentiment Analysis using Simple RNN

This project performs binary sentiment classification on the IMDB Movie Reviews dataset using a **Recurrent Neural Network (Simple RNN)** built with TensorFlow/Keras. The model predicts whether a movie review is **Positive** or **Negative** based on text sequence data.

---

## 🚀 Project Description

The IMDB dataset contains **50,000** pre-tokenized movie reviews. Each review is represented as a sequence of integers corresponding to word indices. The workflow includes:

- Loading IMDB dataset
- Merging and splitting custom train/test sets
- Padding sequences to a fixed length
- Training a Simple RNN model with validation & early stopping
- Evaluating performance on test set
- Saving & loading trained model
- Predicting sentiment for a sample review

This project demonstrates the use of RNNs for Natural Language Processing tasks involving sequential data.

---

## 🧠 Model Architecture

The neural network structure includes:


**Layer Details:**

| Layer        | Purpose |
|--------------|---------|
| Embedding    | Converts tokens → 128-dimensional vectors |
| SimpleRNN    | Learns sequential context in the review |
| Dense(1)     | Outputs sentiment probability |

The output uses a **sigmoid activation** to classify sentiment:

- `> 0.5` → Positive
- `<= 0.5` → Negative

---

## 📂 Dataset Information

The dataset used is the **IMDB Movie Review dataset**, available via:

```python
from tensorflow.keras.datasets import imdb
