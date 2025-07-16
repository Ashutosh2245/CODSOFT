import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(path):
    df = pd.read_csv(path)
    print("Loaded data:", df.shape)
    return df

def eda(df):
    print(df.head(), "\n")
    print(df.info(), "\n")
    print(df.describe(), "\n")
    print("Missing values:\n", df.isnull().sum(), "\n")
    sns.countplot(x='Survived', data=df)
    plt.title('Survived Distribution')
    plt.show()

def preprocess(df):
    drop_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    df = df.drop(drop_cols, axis=1)

    num_features = ['Age', 'Fare']
    cat_features = ['Pclass', 'Sex', 'Embarked', 'SibSp', 'Parch']

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_features),
        ('cat', cat_pipeline, cat_features)
    ])

    X = df.drop('Survived', axis=1)
    ";"
    y = df['Survived']

    X_processed = preprocessor.fit_transform(X)
    print("Feature matrix shape:", X_processed.shape)
    return X_processed, y, preprocessor

def split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)
    print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    rf = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5]
    }
    grid = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy')
    grid.fit(X_train, y_train)
    print("Best params:", grid.best_params_)
    return grid.best_estimator_

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

def show_feature_importance(model, preprocessor):
    num_feats = preprocessor.named_transformers_['num'].named_steps['imputer'].feature_names_in_
    cat_feats = preprocessor.named_transformers_['cat'].named_steps['encoder'].get_feature_names_out()
    feature_names = np.concatenate([num_feats, cat_feats])

    importances = model.feature_importances_
    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(10)
    sns.barplot(x=feat_imp, y=feat_imp.index)
    plt.title('Top 10 Feature Importances')
    plt.show()

def main():
    df = load_data('titanic.csv')
    eda(df)
    X, y, preprocessor = preprocess(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_model(X_train, y_train)
    evaluate(model, X_test, y_test)
    show_feature_importance(model, preprocessor)

if __name__ == '__main__':
    main()
