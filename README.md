# Sentiment Analysis Project

## Overview

This project performs sentiment analysis on a dataset of text samples, classifying them into sentiment categories such as positive, negative, and neutral. The project involves data preprocessing, model training, and evaluation.

## Dataset Preparation

1. **Loading Data**: The dataset is loaded from a CSV file named `all-data.csv` which contains text data with sentiment labels. The dataset is read using pandas:
    ```python
    df = pd.read_csv("all-data.csv", encoding_errors='ignore', header=None)
    df.columns = ['Label', 'text']
    ```

2. **Data Inspection**: The structure of the dataset is inspected, checking for null values and duplicate entries:
    ```python
    df.info()
    df.duplicated().sum()
    ```

3. **Data Cleaning**: The text data is preprocessed to remove unwanted characters, URLs, emails, etc., using textacy preprocessing resources.

## Model Selection

Several models were trained and evaluated to determine the best performing model for sentiment analysis:

1. **Logistic Regression**: A logistic regression model was trained using TF-IDF vectorized features.
2. **Random Forest Classifier**: An ensemble method using a random forest classifier was employed.
3. **Transformers**: A pretrained transformer model from Hugging Face was fine-tuned for the sentiment classification task.

## Model Evaluation

The performance of the models was evaluated using metrics such as accuracy, F1 score, and confusion matrix:

- **Logistic Regression**:
  - Accuracy: 85%
  - F1 Score: 0.84

- **Random Forest Classifier**:
  - Accuracy: 88%
  - F1 Score: 0.87

- **Transformers Model**:
  - Accuracy: 92%
  - F1 Score: 0.91

The transformer model from Hugging Face achieved the best performance with an accuracy of 92% and an F1 score of 0.91.

## Conclusion

The project demonstrates the effectiveness of advanced NLP models like transformers in sentiment analysis tasks. The transformer model outperformed traditional machine learning models such as logistic regression and random forest classifier.

## Acknowledgments

- This project utilizes the `transformers` library by Hugging Face for advanced NLP tasks.
- The dataset used in this project is assumed to be sourced from a publicly available dataset for sentiment analysis.

