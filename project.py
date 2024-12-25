import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, text_features):
        self.text_features = text_features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for text_col in self.text_features:
            X[text_col] = X[text_col].fillna('').astype(str)
        # Add new features
        X['title_length'] = X['title'].apply(len)
        X['description_length'] = X['description'].apply(len)
        X['requirements_length'] = X['requirements'].apply(len)
        X['avg_word_length'] = X['description'].apply(lambda x: np.mean([len(w) for w in x.split()]) if x else 0)
        X['unique_word_count'] = X['description'].apply(lambda x: len(set(x.split())) if x else 0)
        X['word_density'] = X['description_length'] / (X['title_length'] + 1)
        suspicious_keywords = ['urgent', 'no experience', 'free', 'easy money']
        X['suspicious_terms'] = X['description'].apply(
            lambda desc: sum(kw in desc.lower() for kw in suspicious_keywords))
        X['logo_telecommute_ratio'] = X['has_company_logo'] / (X['telecommuting'] + 1)
        X['location'] = X['location'].fillna('unknown').astype(str)
        X.fillna(0, inplace=True)
        return X

class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names):
        self.feature_names = feature_names
        self.vectorizers = {name: TfidfVectorizer(ngram_range=(1, 2), max_features=5000, stop_words='english') for name in feature_names}

    def fit(self, X, y=None):
        for name in self.feature_names:
            self.vectorizers[name].fit(X[name].fillna(''))
        return self

    def transform(self, X):
        transformed_columns = [self.vectorizers[name].transform(X[name].fillna('')).toarray() for name in self.feature_names]
        return np.hstack(transformed_columns)

class my_model():
    def __init__(self):
        self.text_features = ["title", "description", "requirements"]
        self.feature_engineer = FeatureEngineer(self.text_features)
        self.text_processor = TextPreprocessor(self.text_features)
        self.numerical_features = ["telecommuting", "has_company_logo", "has_questions"]
        self.categorical_features = ["location"]
        self.numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        self.categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', self.numerical_transformer, self.numerical_features),
                ('cat', self.categorical_transformer, self.categorical_features)
            ]
        )
        self.feature_selector = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42))
        self.model = LogisticRegressionCV(Cs=[0.001, 0.01, 0.1, 1, 10, 100, 1000], cv=5, class_weight='balanced', max_iter=5000)

    def fit(self, X, y):
        X = self.feature_engineer.transform(X)

        # Combine text, numerical, and categorical features using FeatureUnion
        combined_pipeline = FeatureUnion([
            ('text', self.text_processor),
            ('num_cat', self.preprocessor)
        ])
        combined_features = combined_pipeline.fit_transform(X)

        # Feature selection
        selected_features = self.feature_selector.fit_transform(combined_features, y)

        # Fit the Logistic Regression model
        self.model.fit(selected_features, y)

    def predict(self, X):
        X = self.feature_engineer.transform(X)

        # Combine text, numerical, and categorical features using FeatureUnion
        combined_pipeline = FeatureUnion([
            ('text', self.text_processor),
            ('num_cat', self.preprocessor)
        ])
        combined_features = combined_pipeline.transform(X)

        # Feature selection
        selected_features = self.feature_selector.transform(combined_features)

        # Predict using the trained model
        return self.model.predict(selected_features)