# Fake News Detection Using a CNN-LSTM Deep Learning Approach

## Project Overview
This project implements a comprehensive pipeline to detect fake news using a hybrid CNN-LSTM deep learning model. The pipeline includes:
- Data Augmentation: Synonym replacement to enhance data variability.
- Baseline Model: A logistic regression model using TF-IDF features as a foundational benchmark.
- Advanced Model: A CNN-LSTM model optimized with hyperparameter tuning using Optuna.
- Adversarial Testing: Synthetic errors (e.g., typos) introduced to evaluate robustness.
- Confidence Analysis: Assessment of model prediction reliability.

---

## Error Metric
- Specified Metric: 
  - Accuracy: Primary metric for overall classification performance.
  - Precision, Recall, and F1-Score: Secondary metrics for detailed class-specific analysis.
- Target Value*:
  - Achieve an accuracy of >98%.
-Achieved Value:
  - Final accuracy: 99.9% on validation data.
  - Perfect scores (1.00) for Precision, Recall, and F1-Score for both Fake and Real classes.

---

---

 ## Add "Dataset Details"
Provide details about the dataset for users and contributors.

Dataset Details
The dataset is fetched directly from Kaggle:
- [Fake and Real News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)
- Consists of two CSV files:
  - `Fake.csv`: Contains fake news articles.
  - `True.csv`: Contains real news articles.

The dataset is downloaded and cached using the `kagglehub` library as part of the pipeline.

## *Work Breakdown Structure**
 Task                               Time Spent (Hours)
Data Loading and Cleaning              2                     
Data Augmentation                      1.5                   
Baseline Model Implementation             1                  
Advanced Model Implementation            4                  
Hyperparameter Tuning                    3                     
Adversarial Testing                     1.5                   
Evaluation and Analysis                  2                     
Documentation and Code Cleanup            2                 

---

## Pipeline Summary
1. Data Loading and Augmentation:
   - Loaded raw data from `fake.csv` and `true.csv`.
   - Enhanced data variability with synonym replacement augmentation.

2. Baseline Model:
   - Logistic Regression with TF-IDF features achieved a baseline accuracy of 98.74%.

3. Advanced Model:
   - A CNN-LSTM model was implemented with:
     - Embedding Layer (trainable embeddings with optimized size).
     - Convolutional Layer for feature extraction.
     - Bidirectional LSTM for capturing temporal dependencies.
     - Dropout Layer for regularization.
     - Fully Connected Layer for binary classification.

4. Hyperparameter Tuning:
   - Used Optuna to optimize `embedding_dim`, `lstm_units`, and `dropout_rate`.

5. Evaluation:
   - Metrics included Accuracy, Precision, Recall, and F1-Score.
   - Confusion matrix and confidence analysis were performed for deeper insights.

6. Adversarial Testing:
   - Introduced typos into test samples to mimic real-world challenges.

