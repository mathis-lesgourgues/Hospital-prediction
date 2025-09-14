# Hospital Patient Mortality Prediction

This project explores machine learning models to **predict whether a patient will die in the hospital or not** based on clinical and demographic data.  

The goal is to compare different models, evaluate their performance, and improve predictive accuracy with robust preprocessing and feature engineering.  

---

## Models Tested
So far, I have implemented and benchmarked several traditional machine learning models:
- **Logistic Regression**
- **Random Forest**
- **XGBoost**
- **K-Nearest Neighbors (KNN)**

Performance was primarily evaluated using the **F1-score** to balance precision and recall. 

---

## Preprocessing
- Handling of missing values  
- Encoding of categorical features  
- Standardization of numerical features  

Upcoming improvements:
- New imputation strategies (e.g. KNN imputer, advanced statistical methods)  
- Feature grouping and domain-driven preprocessing  

---

## Results
- Models like **Random Forest** and **XGBoost** achieved the best F1-scores.  
- KNN was more sensitive to scaling and feature selection.  
- Logistic Regression provided a strong baseline.  

(Plots and detailed performance tables can be found in the notebooks / scripts.)

---



## Repository Structure
- `preprocess.py` → data cleaning and feature encoding  
- `models.py` → model training and evaluation scripts  
- `notebooks/` → exploratory analysis and experiments  

---

## 🔮 Next Steps
This project is ongoing. Next iterations will include:
- **Advanced imputation techniques** to better handle missing data.  
- **Neural networks** (MLP, LSTM) to compare deep learning performance with traditional models.  


