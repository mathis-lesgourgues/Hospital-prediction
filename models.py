# ==============
# MODELS.PY
# ==============    
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
# ==============    
# Metrics
# ==============
def summarize(metric_list, name, k):
    mean = np.mean(metric_list)
    std = np.std(metric_list, ddof=1)  # unbiased std (sqrt of variance)
    print(f"{name}: {mean:.3f} ± {std:.3f} (SD over {k} runs)")

def display_metrics(y_test, y_pred):

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("Accuracy:", round(accuracy,2))
    print("Precision:", round(precision,2))
    print("Recall:", round(recall,2))
    print("F1 Score:", round(f1,2))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    plt.show()

# ==============
# Logistic regression
# ==============

def logistic_regression(X_train, X_test, y_train):
    
    LR = LogisticRegression(max_iter=1000).fit(X_train, y_train)

    y_pred = LR.predict(X_test)

    return LR, y_pred

# ==============
# Logistic regression k times
# ==============

# We want to run these regressions k times to check if the results are consistent

def display_metrics_logistic_regression_k_times(X, y, k):
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []

    for _ in range(k):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model, y_pred = logistic_regression(X_train, X_test, y_train)
        
        accuracies.append(accuracy_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred))
        recalls.append(recall_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred))
    
    summarize(accuracies, "Accuracy", k)
    summarize(precisions, "Precision", k)
    summarize(recalls, "Recall", k)
    summarize(f1_scores, "F1 Score", k)

# ==============
# xgboost k times
# ==============        

def display_metrics_xgboost_k_times(X, y, k, n_estimators, learning_rate, max_depth, subsample, colsample_bytree):
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []

    for _ in range(k):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


        xgb_clf = XGBClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,           # similar to max_features effect,
                subsample=subsample,         
                colsample_bytree=colsample_bytree, 
                eval_metric='logloss').fit(X_train, y_train)



        y_pred = xgb_clf.predict(X_test)
        
        accuracies.append(accuracy_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred))
        recalls.append(recall_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred))
    
    summarize(accuracies, "Accuracy", k)
    summarize(precisions, "Precision", k)
    summarize(recalls, "Recall", k)
    summarize(f1_scores, "F1 Score", k)


# ==============
# KNN 
# ==============
def display_metrics_knn_k_times(X1_cleaned, y, k, n_neighbors):
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
   
    for _ in range(5):
        X_train, X_test, y_train, y_test = train_test_split(X1_cleaned, y, test_size=0.2)

        knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance')

        knn.fit(X_train, y_train)   

        y_pred = knn.predict(X_test)     
        accuracies.append(accuracy_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred))
        recalls.append(recall_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred))
        
    summarize(accuracies, "Accuracy", k)
    summarize(precisions, "Precision", k)
    summarize(recalls, "Recall", k)
    summarize(f1_scores, "F1 Score", k)