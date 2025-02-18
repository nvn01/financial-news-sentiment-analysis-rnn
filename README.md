# Sentiment Analysis on Financial News using RNN

## Overview

This project implements a **Sentiment Analysis Model** using a **Recurrent Neural Network (RNN)** to classify financial news as **positive** or **negative**. The model is trained on the **"Sentiment Analysis for Financial News"** dataset from Kaggle.

## Dataset

- **Source**: [Kaggle - Sentiment Analysis for Financial News](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news)
- **Description**: The dataset consists of financial news headlines labeled as `positive`, `negative`, or `neutral`.
- **File Used**: `all-data.csv`
- **Columns**:
  - **Label**: Sentiment class (Positive, Negative, Neutral)
  - **Headline**: The financial news headline

## Model Architecture

- **Embedding Layer**: Converts words into dense vector representations.
- **Recurrent Layer**: Uses **LSTM/GRU** to capture sequential dependencies.
- **Dense Output Layer**: Predicts sentiment classes.

## Requirements

Ensure you have the required dependencies installed:

```bash
pip install numpy pandas tensorflow scikit-learn matplotlib nltk
```

## Usage

### 1. Data Preprocessing

- Load the dataset from Kaggle.
- Perform text cleaning (remove stopwords, tokenize, lemmatize).
- Convert text data into embeddings using `Tokenizer` from TensorFlow/Keras.

### 2. Training the Model

Run the training script to train the RNN model:

```bash
python train_model.py
```

### 3. Evaluating the Model

Evaluate the model's performance using accuracy, precision, recall, and F1-score:

```bash
python evaluate.py
```

### 4. Predict Sentiment of New Text

To classify a new financial headline:

```bash
python predict.py "Stock market surges as investors gain confidence."
```

## Files

- `RNN.ipynb` - Jupyter Notebook with full training and evaluation pipeline.
- `train_model.py` - Python script for training the RNN model.
- `evaluate.py` - Evaluates model performance on test data.
- `predict.py` - Runs inference on new financial headlines.
- `dataset/` - Contains the dataset extracted from Kaggle.

## Results

- **Training Accuracy**: XX%
- **Validation Accuracy**: XX%
- **Test Accuracy**: XX%

## Future Improvements

- Implement **Bidirectional LSTM** for better contextual understanding.
- Experiment with **transformers (BERT/FinBERT)** for improved sentiment accuracy.
- Fine-tune **hyperparameters** (batch size, dropout rate, optimizer, learning rate).

## References

- [Kaggle Dataset](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news)
- [TensorFlow/Keras Documentation](https://www.tensorflow.org/api_docs)
- [NLP Preprocessing Techniques](https://www.nltk.org/)

## License

This project is for educational purposes. Feel free to modify and expand it as needed!
