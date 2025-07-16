import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

plt.rcParams['font.family'] = 'Segoe UI Emoji'


def load_data(path):
    df = pd.read_csv(path)
    print(f"✅ Data loaded: shape={df.shape}")
    print(f"🗂 Columns: {df.columns.tolist()}")
    return df


def eda(df, revenue_col, product_col=None):
    print("\n📊 First 5 rows:\n", df.head())
    print("\n🔍 Info:\n")
    df.info()
    print("\n📈 Describe:\n", df.describe())

    print("\n🧼 Missing values:\n", df.isnull().sum())

    plt.figure(figsize=(10, 5))
    sns.histplot(df[revenue_col], bins=30, kde=True, color='teal')
    plt.title(f'💰 {revenue_col} Distribution')
    plt.show()

    if product_col and product_col in df.columns:
        top_products = df[product_col].value_counts().nlargest(5)
        sns.barplot(x=top_products.values, y=top_products.index, palette='cool')
        plt.title('🏆 Top 5 Products')
        plt.xlabel("Number of Sales")
        plt.show()


def preprocess(df, revenue_col, date_col=None):
    df = df.copy()

    median_rev = df[revenue_col].median()
    df['HighSales'] = (df[revenue_col] >= median_rev).astype(int)

    drop_cols = [revenue_col]
    if date_col and date_col in df.columns:
        drop_cols.append(date_col)
    df = df.drop(columns=drop_cols)

    X = df.drop(columns='HighSales')
    y = df['HighSales']

    num_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_features = X.select_dtypes(include=['object']).columns.tolist()

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy='median')),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy='most_frequent')),
        ("encoder", OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_features),
        ("cat", cat_pipeline, cat_features)
    ])

    print(f"✅ Identified {len(num_features)} numeric & {len(cat_features)} categorical features.")
    return X, y, preprocessor


def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_model(X_train, y_train, preprocessor):
    rf = RandomForestClassifier(random_state=42)

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", rf)
    ])

    param_grid = {
        'classifier__n_estimators': [100, 150],
        'classifier__max_depth': [5, 10, None],
        'classifier__min_samples_split': [2, 4]
    }

    grid = GridSearchCV(pipe, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)

    print("✅ Best Parameters:", grid.best_params_)
    return grid.best_estimator_


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(f"🎯 Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\n🧾 Classification Report:\n", classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples')
    plt.title('📉 Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()


def show_feature_importance(model):
    preprocessor = model.named_steps['preprocessor']
    rf = model.named_steps['classifier']

    num_features = preprocessor.transformers_[0][2]
    cat_features = preprocessor.transformers_[1][2]

    dummy_dict = {col: [0] for col in num_features}
    dummy_dict.update({col: ['missing'] for col in cat_features})
    dummy_df = pd.DataFrame(dummy_dict)

    transformed = preprocessor.transform(dummy_df)

    cat_pipeline = preprocessor.named_transformers_['cat']
    encoder = cat_pipeline.named_steps['encoder']

    try:
        cat_feature_names = encoder.get_feature_names_out(cat_features)
    except Exception:
        n_cat_features = transformed.shape[1] - len(num_features)
        cat_feature_names = [f"{col}_encoded_{i}" for col in cat_features for i in range(n_cat_features // len(cat_features))]

    all_features = np.concatenate([num_features, cat_feature_names])

    if len(all_features) > len(rf.feature_importances_):
        all_features = all_features[:len(rf.feature_importances_)]

    feat_imp = pd.Series(rf.feature_importances_, index=all_features).sort_values(ascending=False).head(10)

    # Fixed here: replaced palette with a single color to avoid FutureWarning
    sns.barplot(x=feat_imp, y=feat_imp.index, color='mediumslateblue')
    plt.title("🔥 Top 10 Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.show()


def main():
    df = load_data("advertising.csv")

    possible_revenue_cols = ['Revenue', 'revenue', 'Sales', 'sales']
    revenue_col = next((c for c in possible_revenue_cols if c in df.columns), None)
    if not revenue_col:
        raise ValueError("🚨 Could not find a revenue/sales column in the data.")

    product_col = 'Product' if 'Product' in df.columns else None
    date_col = 'Date' if 'Date' in df.columns else None

    eda(df, revenue_col, product_col)
    X, y, preprocessor = preprocess(df, revenue_col, date_col)

    X_train, X_test, y_train, y_test = split_data(X, y)

    model = train_model(X_train, y_train, preprocessor)
    model.fit(X_train, y_train)

    evaluate(model, X_test, y_test)
    show_feature_importance(model)


if __name__ == "__main__":
    main()
