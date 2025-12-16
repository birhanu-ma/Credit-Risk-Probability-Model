import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from category_encoders import WOEEncoder
from sklearn import set_config

set_config(transform_output="pandas")

# ----------------------------
# 1. Custom Transformers
# ----------------------------

class AggregateFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.stats = X.groupby('CustomerId')['Amount'].agg([
            ('Total_Transaction_Amount', 'sum'),
            ('Average_Transaction_Amount', 'mean'),
            ('Transaction_Count', 'count'),
            ('Std_Dev_Transaction_Amount', 'std')
        ]).fillna(0)
        return self

    def transform(self, X):
        X = X.copy()
        X = X.merge(self.stats, on='CustomerId', how='left')
        cols = ['Total_Transaction_Amount', 'Average_Transaction_Amount',
                'Transaction_Count', 'Std_Dev_Transaction_Amount']
        X[cols] = X[cols].fillna(0)
        return X

class DateTimeExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        if 'TransactionStartTime' in X.columns:
            dt = pd.to_datetime(X['TransactionStartTime'])
            X['Transaction_Hour'] = dt.dt.hour
            X['Transaction_Day'] = dt.dt.day
            X['Transaction_Month'] = dt.dt.month
            X['Transaction_Year'] = dt.dt.year
            X = X.drop(columns=['TransactionStartTime'])
        return X

class DropIDColumns(BaseEstimator, TransformerMixin):
    """Drop irrelevant IDs but KEEP CustomerId."""
    def fit(self, X, y=None): return self
    def transform(self, X):
        ids_to_drop = [
            "TransactionId", "BatchId", "AccountId",
            "SubscriptionId", "ProviderId", "ProductId"
        ]
        return X.drop(columns=ids_to_drop, errors='ignore')

# ----------------------------
# 2. Feature Engineering Class
# ----------------------------

class FeatureEngineering:
    def __init__(self):
        self.categorical_cols = [
            'CurrencyCode', 'CountryCode', 'ProductCategory',
            'ChannelId', 'PricingStrategy'
        ]
    
    def build_pipeline(self):
        """
        Pipeline that keeps CustomerId untouched.
        Only numeric columns are scaled, categorical columns are WoE encoded.
        """
        class ColumnSelector(BaseEstimator, TransformerMixin):
            """Select columns by name."""
            def __init__(self, columns):
                self.columns = columns
            def fit(self, X, y=None): return self
            def transform(self, X):
                return X[self.columns].copy()
        
        return Pipeline([
            ('aggregates', AggregateFeatures()),
            ('datetime', DateTimeExtractor()),
            ('drop_ids', DropIDColumns()),
            ('imputer', SimpleImputer(strategy='most_frequent')),  # fills missing values
            ('woe', WOEEncoder(cols=self.categorical_cols)),
            ('scaler', StandardScalerWrapper(exclude_cols=['CustomerId']))
        ])

# ----------------------------
# 3. StandardScalerWrapper
# ----------------------------

class StandardScalerWrapper(BaseEstimator, TransformerMixin):
    """
    Custom scaler that excludes specific columns (like CustomerId) from scaling.
    """
    def __init__(self, exclude_cols=None):
        self.exclude_cols = exclude_cols if exclude_cols else []
        self.scaler = StandardScaler()
    
    def fit(self, X, y=None):
        self.cols_to_scale = [c for c in X.columns if c not in self.exclude_cols]
        self.scaler.fit(X[self.cols_to_scale])
        return self
    
    def transform(self, X):
        X_scaled = X.copy()
        X_scaled[self.cols_to_scale] = self.scaler.transform(X[self.cols_to_scale])
        return X_scaled

# ----------------------------
# 4. Information Value Helper
# ----------------------------

def calculate_iv(df, feature, target):
    data = df[[feature, target]].copy()
    data['target'] = data[target]
    grouped = data.groupby(feature)['target'].agg(['count', 'sum'])
    grouped.columns = ['Total', 'Bad']
    grouped['Good'] = grouped['Total'] - grouped['Bad']
    grouped['Dist_Good'] = grouped['Good'] / grouped['Good'].sum()
    grouped['Dist_Bad'] = grouped['Bad'] / grouped['Bad'].sum()
    eps = 1e-10
    grouped['WoE'] = np.log((grouped['Dist_Good'] + eps) / (grouped['Dist_Bad'] + eps))
    grouped['IV'] = (grouped['Dist_Good'] - grouped['Dist_Bad']) * grouped['WoE']
    return grouped['IV'].replace([np.inf, -np.inf], 0).sum()
