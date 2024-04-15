# -*- coding: utf-8 -*-
"""
Created on Tue Apr  12 11:36:21 2024

@author: Ali
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (
    StratifiedShuffleSplit,
    GridSearchCV,
    RandomizedSearchCV,
)
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFE
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_curve,
    auc,
    f1_score,
    confusion_matrix,
)
import pickle


# Create a model builer class
class ModelBuilder:
    def __init__(self):
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {
            "Logistic Regression": LogisticRegression(),
            "Decision Tree": DecisionTreeClassifier(),
            "SVM": SVC(probability=True),
            "Random Forest": RandomForestClassifier(),
            "Neural Network": MLPClassifier(),
        }

    def train_models(self):
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(self.X_train, self.y_train)

    def evaluate_models(self):
        for name, model in self.models.items():
            print(f"Evaluating {name}...")
            y_pred = model.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            print(f"Accuracy Score for {name}: {accuracy:.4f}")
            # precision_score(y_test, y_pred_lr, pos_label='fatal', zero_division=0)
            precision = precision_score(
                self.y_test, y_pred, pos_label="non-fatal", zero_division=0
            )
            print(f"Precision for {name}: {precision:.4f}")
            recall = recall_score(self.y_test, y_pred, pos_label="non-fatal")
            print(f"Recall for {name}: {recall:.4f}")
            # F1 Score
            f1 = f1_score(self.y_test, y_pred, pos_label="non-fatal")
            print(f"F1 Score for {name}: {f1:.4f}")
            # Confusion Matrix
            confusion = confusion_matrix(self.y_test, y_pred)
            print(f"Confusion Matrix for {name}: ")
            print(confusion)

    def plot_models(self):
        plt.figure(figsize=(10, 8))
        for name, model in self.models.items():
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba, pos_label="non-fatal")
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")

        plt.title("Receiver Operating Characteristic (ROC) Curve")
        plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.show()

    def fine_tune_models_grid_search(self, params):
        for name, model in self.models.items():
            print(f"Fine-tuning {name} using GridSearchCV...")
            grid_search = GridSearchCV(model, params[name], cv=5, n_jobs=-1)
            grid_search.fit(self.X_train, self.y_train)
            self.models[name] = grid_search.best_estimator_
            print(f"Best parameters for {name}: {grid_search.best_params_}")
            print(f"Best accuracy for {name}: {grid_search.best_score_:.4f}")

    def fine_tune_models_randomized_search(self, params, n_iter=100):
        for name, model in self.models.items():
            print(f"Fine-tuning {name} using RandomizedSearchCV...")
            random_search = RandomizedSearchCV(
                model, params[name], n_iter=n_iter, cv=5, n_jobs=-1
            )
            random_search.fit(self.X_train, self.y_train)
            self.models[name] = random_search.best_estimator_
            print(f"Best parameters for {name}: {random_search.best_params_}")
            print(f"Best accuracy for {name}: {random_search.best_score_:.4f}")


if __name__ == "__main__":
    # Read the dataset
    grp2 = pd.read_csv(r"KSI.csv")


# "C:\Users\Blessing\OneDrive - Centennial College\WINTER 2024 YEAR 3,2024\COMP 247\KSI.csv"
# Calculate the threshold
# threshold = 0.8 * len(grp2)

# Drop columns with missing values exceeding the threshold
# grp2_cleaned = grp2.dropna(axis=1, thresh=threshold)

# Drop irrelevant columns (unique identifiers)
unique_identifier_columns = ["INDEX_", "ACCNUM", "YEAR", "DATE", "FATAL_NO", "ObjectId"]
grp2_cleaned = grp2.drop(columns=unique_identifier_columns)
print(grp2_cleaned.columns)
# grp2_cleaned = grp2_cleaned.drop(columns=unique_identifier_columns)
# Change 'ACCLASS' column to binary: 'non-fatal' or 'fatal'
grp2_cleaned["ACCLASS"] = grp2_cleaned["ACCLASS"].apply(
    lambda x: (
        "non-fatal" if x in ["Property Damage Only", "Non-Fatal Injury"] else "fatal"
    )
)


# # print(grp2_cleaned)


# # Heatmap to visualize correlation between columns
# correlation_matrix = grp2_cleaned.corr()
# plt.figure(figsize=(10, 8))
# sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
# plt.title("Correlation Heatmap of DataFrame Columns")
# plt.show()
# Remove duplicates
# grp2_cleaned = grp2_cleaned.drop_duplicates()
# print(grp2_cleaned.columns)
# Drop irrelevant columns (e.g., latitude and longitude)
# grp2_cleaned = grp2_cleaned.drop(columns=["X", "Y"])
# Extract features from date column
# grp2_cleaned["DATE"] = pd.to_datetime(grp2_cleaned["DATE"])
# grp2_cleaned["Weekday"] = grp2_cleaned["DATE"].dt.weekday
# grp2_cleaned["Day"] = grp2_cleaned["DATE"].dt.day
# grp2_cleaned["Month"] = grp2_cleaned["DATE"].dt.month
# grp2_cleaned = grp2_cleaned.drop(columns=["DATE"])
# Remove other irrelevant columns or columns with too many missing values
irrelevant_columns = [
    "X",
    "Y",
    "OFFSET",
    "ACCLOC",
    "INITDIR",
    "STREET1",
    "STREET2",
    "WARDNUM",
    "HOOD_158",
    "NEIGHBOURHOOD_158",
    "HOOD_140",
    "NEIGHBOURHOOD_140",
    "DIVISION",
    "INJURY",
    "DRIVACT",
    "DRIVCOND",
    "PEDTYPE",
    "PEDACT",
    "PEDCOND",
    "CYCLISTYPE",
    "CYCACT",
    "CYCCOND",
    "DISTRICT",
    "INVAGE",
    "IMPACTYPE",
    "INVTYPE",
    "VEHTYPE",
    "MANOEUVER",
]
grp2_cleaned = grp2_cleaned.drop(columns=irrelevant_columns)

print(grp2_cleaned.columns)


# Columns to convert to one-hot encoding
columns_to_encode = [
    "PEDESTRIAN",
    "CYCLIST",
    "AUTOMOBILE",
    "MOTORCYCLE",
    "TRUCK",
    "TRSN_CITY_VEH",
    "EMERG_VEH",
    "SPEEDING",
    "AG_DRIV",
    "REDLIGHT",
    "ALCOHOL",
    "DISABILITY",
]

# Perform one-hot encoding
grp2_cleaned_encoded = pd.get_dummies(
    grp2_cleaned[columns_to_encode], dummy_na=False, drop_first=False
)

# Replace 'yes' with 1 and blanks with 0
grp2_cleaned_encoded.replace({"yes": 1, "": 0}, inplace=True)

# Concatenate the one-hot encoded columns with the original DataFrame
grp2_cleaned_f = pd.concat(
    [grp2_cleaned.drop(columns_to_encode, axis=1), grp2_cleaned_encoded], axis=1
)

# Print the DataFrame to verify changes
print(grp2_cleaned_f.columns)


# # Plot number of unique accidents by year, month, and day
# plt.figure(figsize=(12, 10))
# plt.subplot(3, 1, 1)
# sns.countplot(x="YEAR", data=grp2_cleaned)
# plt.title("Accidents by Year")

# plt.subplot(3, 1, 2)
# sns.countplot(x="Month", data=grp2_cleaned)
# plt.title("Accidents by Month")

# plt.subplot(3, 1, 3)
# sns.countplot(x="Weekday", data=grp2_cleaned)
# plt.title("Accidents by Weekday")

# plt.tight_layout()
# plt.show()

# # Check relation between features and target
# plt.figure(figsize=(12, 10))
# plt.subplot(3, 2, 1)
# sns.countplot(x="YEAR", hue="ACCLASS", data=grp2_cleaned)
# plt.title("Accidents by Year")

# plt.subplot(3, 2, 2)
# sns.countplot(x="Month", hue="ACCLASS", data=grp2_cleaned)
# plt.title("Accidents by Month")

# plt.subplot(3, 2, 3)
# sns.countplot(x="Weekday", hue="ACCLASS", data=grp2_cleaned)
# plt.title("Accidents by Weekday")

# plt.tight_layout()
# plt.show()

# # Scatter plot of fatal and non-fatal accidents
# plt.figure(figsize=(8, 6))
# sns.scatterplot(x="LONGITUDE", y="LATITUDE", hue="ACCLASS", data=grp2_cleaned)
# plt.title("Fatal and Non-Fatal Accidents")
# plt.xlabel("Longitude")
# plt.ylabel("Latitude")
# plt.show()


# Data Modeling
# grp2_cleaned = grp2_cleaned.drop(columns=["YEAR"])

# grp2_cleaned_encoded = grp2_cleaned_encoded.drop_duplicates()
# Separate features and target columns
X = grp2_cleaned_f.drop(columns=["ACCLASS"])
y = grp2_cleaned_f["ACCLASS"]

# Stratified Shuffle Splitting
stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in stratified_split.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# Handling missing data and managing categorical data
numeric_features = [
    "TIME",
    "LATITUDE",
    "LONGITUDE",
    # "Weekday", "Day", "Month"
]

categorical_features = [
    "ROAD_CLASS",
    "LOCCOORD",
    "TRAFFCTL",
    "VISIBILITY",
    "LIGHT",
    "RDSFCOND",
    # "INVAGE",
    # "IMPACTYPE",
    # 'INVTYPE',
    # "VEHTYPE",
    # 'MANOEUVER',
    "PEDESTRIAN_Yes",
    "CYCLIST_Yes",
    "AUTOMOBILE_Yes",
    "MOTORCYCLE_Yes",
    "TRUCK_Yes",
    "TRSN_CITY_VEH_Yes",
    "EMERG_VEH_Yes",
    "SPEEDING_Yes",
    "AG_DRIV_Yes",
    "REDLIGHT_Yes",
    "ALCOHOL_Yes",
    "DISABILITY_Yes",
]
numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)
# Managing imbalanced classes
smote = SMOTE(random_state=42)

# Define the pipeline
pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("sampling", smote)])

# Define the model for feature selection
model = RandomForestClassifier()

# Feature selection using RFE
rfe = RFE(estimator=model, n_features_to_select=17)

# Fit RFE within the pipeline
pipeline.fit(X_train, y_train)

# Transform X_train using the preprocessor pipeline
X_train_preprocessed = preprocessor.transform(X_train)

# Get the column names after transformation
transformed_columns = preprocessor.named_transformers_["cat"][
    "onehot"
].get_feature_names_out(categorical_features)

# Combine the names of numeric features and transformed categorical features
all_features = numeric_features + list(transformed_columns)

# # Create a DataFrame with the transformed features
# X_train_encoded = pd.DataFrame(X_train_preprocessed, columns=all_features)

# # Print the DataFrame with encoded categorical columns
# print(X_train_encoded.head())
# print("Shape of X_train_encoded:", X_train_encoded.shape)

# Transform X_test using the preprocessor pipeline
X_test_preprocessed = preprocessor.transform(X_test)

# Model Building
# Use the ModelBuilder Class on the data
modelBuilder = ModelBuilder()
modelBuilder.X_train = X_train_preprocessed
modelBuilder.X_test = X_test_preprocessed
modelBuilder.y_train = y_train
modelBuilder.y_test = y_test
# Define hyperparameters for each model
params = {
    "Logistic Regression": {"C": [0.1, 1, 10, 100]},
    "Decision Tree": {
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
    },
    "SVM": {
        "C": [1, 10],
        "gamma": [0.1, 0.01],
        "kernel": ["rbf"],
    },
    "Random Forest": {
        "n_estimators": [50, 100],
        "max_depth": [None, 10],
        "min_samples_split": [2, 5],
    },
    "Neural Network": {
        "hidden_layer_sizes": [(50,), (100,)],
        "alpha": [0.0001, 0.001],
    },
}

# Use the ModelBuilder Class on the data
modelBuilder = ModelBuilder()
modelBuilder.X_train = X_train_preprocessed
modelBuilder.X_test = X_test_preprocessed
modelBuilder.y_train = y_train
modelBuilder.y_test = y_test
# Train models
modelBuilder.train_models()

# Evaluate models
modelBuilder.evaluate_models()

# # Fine-tune models using GridSearchCV
modelBuilder.fine_tune_models_grid_search(params)

# # Fine-tune models using RandomizedSearchCV
modelBuilder.fine_tune_models_randomized_search(params)

modelBuilder.plot_models()

#for loop to dump each model into its own pipeline
for name, model in modelBuilder.models.items():
     pipeline=Pipeline(steps=[("preprocessor", preprocessor),("model",model)])
     pipeline.fit(X_train,y_train)
     with open(f"{name}_pipeline.pkl",'wb') as file:
         pickle.dump(pipeline,file)
