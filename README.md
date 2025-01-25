# Credit Card Fraud Detection using Sampling Techniques and Various Classifiers

This project explores the impact of different sampling methods on the performance of various machine learning classifiers for detecting fraudulent transactions in a credit card dataset. The dataset has been pre-processed and balanced using SMOTE (Synthetic Minority Over-sampling Technique) to handle class imbalance, followed by evaluations using several sampling techniques. Different classifiers were tested on the resulting samples to identify the best approach for each.

## Dataset

The dataset used for this project is the **Credit Card Fraud Detection** dataset, which includes transaction data and a label indicating whether the transaction is fraudulent (Class = 1) or not (Class = 0). The dataset can be loaded from a CSV file named `Creditcard_data.csv`.

## Sampling Techniques

To mitigate the class imbalance, we applied different **sampling techniques** to balance the dataset:
- **Systematic Sampling**: Selects samples in a systematic manner by choosing every nth observation.
- **Cluster Sampling**: Divides the data into clusters and selects a subset of clusters to form the sample.

The following sample sizes were used:
- **500 samples**
- **1000 samples**
- **1500 samples**
- **2000 samples**
- **2500 samples**

## Classifiers

We evaluated the performance of the following machine learning classifiers:
- **Logistic Regression**
- **Gradient Boosting**
- **K-Nearest Neighbors (KNN)**
- **Decision Tree**
- **Naive Bayes**

## Process Flow

1. **Data Loading**: The dataset was loaded into a pandas DataFrame.
2. **Data Preprocessing**: The data was balanced using SMOTE, which generates synthetic samples to balance the class distribution.
3. **Sampling**: We applied different sampling techniques to create smaller subsets of the dataset.
4. **Model Training**: The classifiers were trained on the training set (80% of each sample) and evaluated on the testing set (20% of each sample).
5. **Model Evaluation**: The models' performance was evaluated using accuracy as the metric, and results were recorded for each classifier across the different sampling methods.

## Results

The results of the classifiers on different samples were saved in the file `sampling_model_results_formatted.csv`, which contains:
- **Rows**: The models applied (e.g., Logistic Regression, Gradient Boosting, etc.)
- **Columns**: The different sampling techniques (e.g., Systematic Sample 500, Cluster Sample 1000, etc.)
- **Values**: Accuracy scores for each combination of model and sampling technique.

The best sampling technique for each model was also identified and printed.

## Files Included

- `Creditcard_data.csv`: The credit card fraud detection dataset (should be placed in the same directory).
- `sampling_model_results_formatted.csv`: Contains the evaluation results for each classifier and sampling technique combination.
- `README.md`: This file.

## Discussion

The analysis aims to identify the most effective sampling technique for each classifier. This is important in real-world scenarios, as balancing datasets is crucial for ensuring that machine learning models do not suffer from bias toward the majority class. Different classifiers and sampling techniques may work better for different datasets or problem domains.

By exploring multiple classifiers and sampling techniques, we can better understand which combinations provide optimal results for fraud detection. This approach can be extended to other domains where imbalanced datasets are common.

## Running the Code

1. Make sure you have the required libraries installed. You can install them via `pip`:
   ```bash
   pip install pandas scikit-learn imbalanced-learn numpy
