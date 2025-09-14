import pandas as pd 
from sklearn.preprocessing import StandardScaler
import warnings



# ==========
# preprocessing the df
# ==========

def preprocess_data(X):

    columns_to_drop = [
        "aps", "sps", "surv2m", "surv6m", "prg2m", "prg6m",
        "dnr", "dnrday", "slos", "death", "d.time", "adls", "adlp",
        "dzclass", "scoma", "hday", "totcst", "charges", "totmcst"
    ]

    X.drop(columns=columns_to_drop, inplace=True)

    X.fillna({
        "bun": 6.51,
        "pafi": 333.3,
        "alb": 3.5,
        "bili": 1.01,
        "wblc": 9,
        "urine": 2502,
        "crea": 1.01
    }, inplace=True)


    X.drop(columns=["glucose", "income"], inplace=True)

    return X


def fillna_and_encode(X, categorical_features):

    warnings.filterwarnings("ignore", category=UserWarning)

    X1_categorical = X[categorical_features].copy()
    
    # Replace the missing values 
    for col in X1_categorical.columns:
        most_frequent = X1_categorical[col].value_counts().idxmax()
        # Fill NaNs with the most frequent value
        X1_categorical[col] = X1_categorical[col].fillna(most_frequent)


    # One-hot encode the categorical variables
    X1_categorical_encoded = pd.get_dummies(X1_categorical, columns=categorical_features, drop_first=True)



    X1_numerical = X.drop(columns=categorical_features)

    # Fill remaining NaNs in numerical features with the mean of the column
    X1_numerical.fillna(X1_numerical.mean(numeric_only=True), inplace=True)

    # Scaling numerical features
    sc = StandardScaler()
    X1_numerical = pd.DataFrame(
        sc.fit_transform(X1_numerical),
        columns=X1_numerical.columns,
        index=X1_numerical.index 
    )

    # Merge numerical and encoded categorical features
    X1_cleaned = pd.concat([X1_numerical, X1_categorical_encoded], axis=1)

    return X1_cleaned