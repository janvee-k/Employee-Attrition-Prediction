#  **Employee Attrition Prediction**

This project uses machine learning to analyze employee data and predict attrition using the **IBM HR Analytics Employee Attrition & Performance** dataset from Kaggle.

---

##  Overview

The goal is to:
- Understand employee attrition trends
- Perform exploratory data analysis (EDA)
- Build multiple ML models to predict attrition
- Compare their performance
- Predict attrition for new employee input

---

##  Dataset

-  Total Records: 1470
-  Features: 35
-  Target Variable: `Attrition` (Yes/No)
-  Source: [Kaggle - IBM HR Analytics Dataset](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)

---

##  Tech Stack

- **Python**
- **Jupyter Notebook**
- **Libraries**:  
  `NumPy`, `Pandas`, `Matplotlib`, `Seaborn`, `Scikit-learn`

---

##  Data Preprocessing & Feature Engineering

- Label Encoding of categorical columns
- Handling correlated features
- Dropping low-impact columns
- Splitting data into train/test sets

---

##  Machine Learning Models Used

| Model               | Accuracy |
|--------------------|----------|
| Logistic Regression| 86.39%   |
| Random Forest      | 87.41%   |
| Naive Bayes        | 85.37%   |
| K-Means Clustering | 51.02%   |

---

##  Exploratory Data Analysis

- Countplots & Pie charts for Attrition distribution
- Attrition vs Department
- Job Satisfaction & Monthly Income analysis
- Correlation Heatmap
- Pairplots to study relationships

---

##  Manual Prediction Example

- Accepts new employee input as a dictionary
- Encodes it using the same encoders
- Predicts using trained Random Forest model

```python
Prediction: Yes
Probability of Attrition: 0.56
Accuracy on Test Data: 0.87
```

---

## Results & Conclusion

- Among the models tested, **Random Forest** achieved the highest accuracy of **87.41%**, making it the best performing model for predicting employee attrition in this dataset.  
- Logistic Regression and Naive Bayes also performed well with accuracies above 85%, showing that simple models can still provide good predictive power.  
- K-Means Clustering, being an unsupervised method, showed much lower accuracy and is less suitable for this classification task.  
- This project demonstrates that machine learning can effectively predict employee attrition, which can help HR teams identify at-risk employees and take proactive measures to improve retention and reduce turnover costs.  
- The manual prediction feature allows users to input new employee data and get real-time attrition risk predictions, making the model practical for business applications.

---

##  File Info

- `ML_EmployeeAttritionPrediction.ipynb`: Main notebook with all code, analysis, models, and predictions.

---

##  How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/janvee-k/Employee-Attrition-Prediction.git
   cd Employee-Attrition-Prediction
   ```

## To Install dependencies:
pip install -r requirements.txt

## Open the notebook:
jupyter notebook ML_EmployeeAttritionPredictionPROJECT.ipynb

## License
This project is licensed under the MIT License.

## Project Structure
Employee-Attrition-Prediction/
├── ML_EmployeeAttritionPrediction.ipynb
├── README.md
└── requirements.txt
