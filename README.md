# Performance Analysis of TabPFN and Classical Machine Learning Algorithms on Tabular Datasets

### BAD702 – Statistical Machine Learning for Data Science

A comprehensive experimental study comparing **TabPFN (Tabular Prior-Data Fitted Network)** with traditional machine learning algorithms across multiple tabular datasets. The project evaluates predictive performance, statistical significance, and computational efficiency across classification and regression tasks.

---

## Overview

Tabular data is widely used in healthcare, finance, business analytics, and scientific research. Classical machine learning algorithms such as Logistic Regression, Random Forest, Support Vector Machines, and Gradient Boosting have traditionally been used to solve tabular prediction problems.

Recently, **TabPFN**, a transformer-based model specifically designed for tabular learning, has emerged as an alternative capable of achieving competitive predictive performance with comparatively limited preprocessing and hyperparameter tuning.

This project conducts a systematic comparison between TabPFN and classical machine learning algorithms to investigate:

- Whether TabPFN consistently outperforms traditional machine learning models.
- How model performance changes across classification and regression tasks.
- Whether observed performance differences are statistically significant.
- How training and prediction times differ between model families.
- Which classical algorithms remain competitive against TabPFN.

The study uses three datasets with different characteristics and evaluates the models under controlled experimental conditions.

## Objectives

1. Compare TabPFN against selected classical machine learning algorithms.
2. Evaluate models across binary classification and regression tasks.
3. Perform exploratory data analysis and data-quality assessment.
4. Apply appropriate preprocessing techniques for different model requirements.
5. Evaluate predictive performance using task-specific metrics.
6. Perform statistical significance testing with Holm correction.
7. Analyse training time, prediction time, and total computational cost.
8. Identify the strengths and limitations of TabPFN compared with classical approaches.

## Datasets

The project uses three publicly available tabular datasets.

| Dataset | Task | Samples | Features |
|---|---|---:|---:|
| Breast Cancer Wisconsin Diagnostic | Binary Classification | 569 | 30 |
| Adult Income | Binary Classification | 32,561 | 14 |
| California Housing | Regression | 20,640 | 8 |

### 1. Breast Cancer Wisconsin Diagnostic

- **Task:** Binary classification
- **Target:** Benign or Malignant
- **Samples:** 569
- **Features:** 30 numerical attributes

Contains measurements extracted from digitised images of breast-mass cell nuclei. The objective is to classify tumours based on their morphological characteristics.

### 2. Adult Income Dataset

- **Task:** Binary classification
- **Target:** Income <=50K or >50K
- **Samples:** 32,561
- **Features:** 14
- **Feature types:** Numerical and categorical

Contains demographic and employment-related attributes, including age, education, occupation, marital status, and working hours.

This dataset is particularly useful for evaluating how models handle mixed numerical and categorical features.

### 3. California Housing Dataset

- **Task:** Regression
- **Target:** Median House Value
- **Samples:** 20,640
- **Features:** 8 numerical attributes

Contains California housing and geographic information. The objective is to predict median house values based on the available housing characteristics.

## Machine Learning Algorithms

### TabPFN

TabPFN (Tabular Prior-Data Fitted Network) is a transformer-based machine learning model designed specifically for tabular datasets.

Unlike conventional machine learning algorithms that learn dataset-specific parameters through iterative training, TabPFN uses in-context learning based on its prior training on synthetic tabular tasks.

### Classical Machine Learning Algorithms

The following algorithms are implemented for comparison:

| Algorithm | Application |
|---|---|
| Logistic Regression | Classification |
| K-Nearest Neighbors (KNN) | Classification |
| Decision Tree | Classification |
| Random Forest | Classification |
| Support Vector Machine (SVM) | Classification |
| Linear Regression | Regression |
| Gradient Boosting | Regression |

## Methodology

The project follows a structured machine learning experimentation pipeline.

### 1. Data Collection

The datasets are loaded using Scikit-learn and the UCI Adult Income dataset.

### 2. Data Quality Analysis

The following checks are performed:

- Missing-value detection.
- Duplicate-record analysis.
- Data-type identification.
- Numerical and categorical feature identification.
- Class distribution analysis.

### 3. Exploratory Data Analysis

EDA is conducted using:

- Descriptive statistics.
- Mean, median, variance, and standard deviation.
- Quartiles and interquartile range.
- Histograms.
- Boxplots.
- Correlation matrices.
- Class distribution visualisations.

### 4. Data Preprocessing

Different preprocessing strategies are applied based on model requirements.

**Classical Machine Learning:**
- Missing-value imputation.
- Numerical feature standardisation.
- Categorical feature encoding using One-Hot Encoding.
- ColumnTransformer-based preprocessing pipelines.

**TabPFN:**
- Preserves the original tabular feature structure.
- Avoids the classical scaling and one-hot encoding pipeline.
- Handles the input according to TabPFN's tabular learning requirements.

### 5. Train-Test Splitting

The experiments use an 80:20 train-test split with `random_state=42`.

- Stratified splitting is used for classification tasks.
- Standard random splitting is used for regression.
- The same underlying training and testing observations are used for the compared models.

**Experimental limitation:** Due to the CPU-only execution environment, the Adult Income and California Housing TabPFN experiments use controlled subsets of 2,000 training and 1,000 testing observations. These are not full-dataset TabPFN benchmarks.

### 6. Model Training and Prediction

TabPFN and the selected classical algorithms are fitted and evaluated on the prepared datasets.

Both training and prediction times are recorded to understand the computational behaviour of each algorithm.

### 7. Model Evaluation

The models are evaluated using classification and regression metrics.

### 8. Statistical Analysis

Paired statistical tests are performed to investigate whether observed performance differences are statistically significant.

Holm correction is applied to control for multiple comparisons.

### 9. Computational Performance Analysis

Training time, prediction time, and total execution time are compared across algorithms.

---

## Evaluation Metrics

### Classification Metrics

| Metric | Purpose |
|---|---|
| Accuracy | Measures overall correct predictions |
| Precision | Measures the reliability of positive predictions |
| Recall | Measures the proportion of actual positives identified |
| F1 Score | Balances precision and recall |
| ROC-AUC | Evaluates discrimination across classification thresholds |
| Confusion Matrix | Shows correct and incorrect classifications |

**Primary metric:** F1 Score.

F1 is selected as the primary classification metric because it balances precision and recall, particularly important for the imbalanced Adult Income dataset.

### Regression Metrics

| Metric | Purpose |
|---|---|
| R² Score | Measures explained variance |
| MAE | Measures average absolute prediction error |
| RMSE | Measures root mean squared prediction error |

**Primary metric:** R² Score.

### Statistical Tests

- McNemar Exact Test for paired classification predictions.
- Wilcoxon Signed-Rank Test for paired regression errors.
- Holm-Bonferroni correction for multiple comparisons.

---

## Experimental Results

The following results are obtained from the executed notebook.

### Best Model Performance

| Dataset | TabPFN Score | Best Classical Model | Classical Score |
|---|---:|---|---:|
| Breast Cancer | F1: 0.9861 | Logistic Regression | F1: 0.9861 |
| Adult Income | F1: 0.6712 | Random Forest | F1: 0.6590 |
| California Housing | R²: 0.8561 | Gradient Boosting | R²: 0.7575 |

### Key Findings

**1. Breast Cancer Classification**

TabPFN achieved an F1 score of 0.9861, matching Logistic Regression and SVM.

The statistical analysis did not establish a significant advantage for TabPFN after Holm correction.

**2. Adult Income Classification**

TabPFN achieved an F1 score of 0.6712, compared with 0.6590 for Random Forest.

Although TabPFN achieved the highest numerical F1 score, the difference against Random Forest was not statistically significant after correction.

**3. California Housing Regression**

TabPFN achieved an R² score of 0.8561, compared with 0.7575 for Gradient Boosting.

This difference of 0.0985 was statistically significant after Holm correction.

### Statistical Significance

Out of 14 pairwise comparisons, 6 remained statistically significant after Holm correction.

- Breast Cancer: 0 of 5 comparisons were significant.
- Adult Income: 2 of 5 comparisons were significant.
- California Housing: 4 of 4 comparisons were significant.

The strongest statistically supported advantage for TabPFN was observed in the California Housing regression experiment.

### Computational Performance

The experiments showed substantial differences in computational cost.

| Dataset | TabPFN Total Time | Classical Comparison |
|---|---:|---|
| Breast Cancer | 92.48 seconds | Logistic Regression: 0.0641 seconds |
| Adult Income | 134.03 seconds | Random Forest: 0.73 seconds |
| California Housing | 147.83 seconds | Gradient Boosting: 1.31 seconds |

These measurements were obtained in a CPU-only environment.

TabPFN exhibited a substantially higher computational cost in these experiments, with much of its computation occurring during prediction rather than conventional model fitting.

GPU execution may change these runtime comparisons.

---

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Manipulation | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Tabular Deep Learning | TabPFN |
| Data Visualisation | Matplotlib, Seaborn |
| Statistical Analysis | SciPy, Statsmodels |
| Development Environment | Jupyter Notebook / Google Colab |

## Project Structure

```text
TabPFN-Performance-Analysis/
│
├── Performance_Analysis_of_TabPFN_and_Classical_Machine_Learning_Algorithms_on_Tabular_Datasets.ipynb
│
├── README.md
│
└── requirements.txt
```

The Jupyter Notebook contains the complete experimental workflow, including dataset loading, exploratory analysis, preprocessing, model implementation, statistical testing, computational analysis, and conclusions.

## Installation and Execution

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
```

### 2. Navigate to the Project Directory

```bash
cd YOUR-REPOSITORY
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy statsmodels tabpfn jupyter
```

### 5. Launch Jupyter Notebook

```bash
jupyter notebook
```

### 6. Run the Notebook

Open the notebook:

`Performance_Analysis_of_TabPFN_and_Classical_Machine_Learning_Algorithms_on_Tabular_Datasets.ipynb`

Execute the cells sequentially to reproduce the analysis.

**Note:** The Adult Income dataset requires an internet connection for downloading. TabPFN execution may require additional computational resources and compatible package versions. GPU availability can affect execution time.

---

## Limitations

The study acknowledges the following limitations:

1. Only three datasets are included.
2. Adult Income and California Housing TabPFN experiments use controlled subsets rather than the full datasets.
3. All reported computational timings were measured in a CPU-only environment.
4. A single train-test split is used rather than repeated cross-validation.
5. Classical model hyperparameters are largely based on default configurations.
6. Statistical conclusions are specific to the evaluated test sets and cannot automatically be generalised to all tabular datasets.

## Future Enhancements

- Evaluate additional tabular datasets.
- Perform experiments using GPU acceleration.
- Implement repeated stratified cross-validation.
- Perform systematic hyperparameter optimisation.
- Compare TabPFN with XGBoost, LightGBM, and CatBoost.
- Investigate probability calibration.
- Explore additional techniques for handling class imbalance.
- Evaluate performance on larger datasets.

## Conclusion

This project demonstrates that no single machine learning algorithm is consistently superior across all tabular learning problems.

TabPFN achieved the highest headline metric on all three datasets, but its advantage was statistically supported only on California Housing regression.

Classical algorithms remained highly competitive, with Logistic Regression matching TabPFN on Breast Cancer and Random Forest approaching its performance on Adult Income.

The experiments also revealed substantial computational differences, with classical algorithms completing considerably faster in the CPU-only environment.

Ultimately, selecting an appropriate model requires considering predictive performance, statistical evidence, computational efficiency, dataset characteristics, and practical deployment requirements.

**The central finding is that TabPFN is a powerful alternative to classical machine learning, not a universal replacement for it.**

---

## Author

**SHASHANK N**

B.Tech – Computer Science and Engineering (Data Science)

BAD702 – Statistical Machine Learning for Data Science

---

## Keywords

`TabPFN` `Machine Learning` `Classical Machine Learning` `Tabular Data` `Transformer Models` `Classification` `Regression` `Random Forest` `Logistic Regression` `Support Vector Machine` `Gradient Boosting` `Statistical Analysis` `Performance Evaluation` `Python` `Data Science`
