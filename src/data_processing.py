import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import set_config
from category_encoders import WOEEncoder

# Force scikit-learn to keep data as a Pandas DataFrame
set_config(transform_output="pandas")

# ---------------------------------------------------------
# 1. Custom Transformers
# ---------------------------------------------------------

class AggregateFeatures(BaseEstimator, TransformerMixin):
    """Requirement: Create Total, Average, Count, and Std per Customer."""
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
    """Requirement: Extract Hour, Day, Month, and Year."""
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
    """Removes unique IDs that cannot be converted to numbers."""
    def fit(self, X, y=None): return self
    def transform(self, X):
        # We drop all columns that are unique IDs or contain non-numeric strings 
        # that we aren't encoding with WoE.
        ids_to_drop = [
            "TransactionId", "BatchId", "AccountId", 
            "SubscriptionId", "CustomerId", "ProviderId", "ProductId"
        ]
        return X.drop(columns=ids_to_drop, errors='ignore')

# ---------------------------------------------------------
# 2. Main Feature Engineering Class
# ---------------------------------------------------------

class FeatureEngineering:
    def __init__(self):
        # These are the categorical columns we WANT to keep and transform via WoE
        self.categorical_cols = [
            'CurrencyCode', 'CountryCode', 'ProductCategory', 
            'ChannelId', 'PricingStrategy'
        ]

    def build_pipeline(self):
        """
        The robust pipeline that handles all Task 3 requirements.
        """
        return Pipeline([
            # 1. Create Aggregate Features (Adds numeric columns)
            ('aggregates', AggregateFeatures()),
            
            # 2. Extract Time Features (Adds numeric columns)
            ('datetime', DateTimeExtractor()),
            
            # 3. DROP IDs (This removes the string IDs causing your error)
            ('drop_ids', DropIDColumns()),
            
            # 4. Handle Missing Values
            ('imputer', SimpleImputer(strategy='most_frequent')), 
            
            # 5. WoE Encoding (Turns your categorical strings into numeric floats)
            ('woe', WOEEncoder(cols=self.categorical_cols)),
            
            # 6. Standardization (Now only sees numbers, so it won't crash!)
            ('scaler', StandardScaler()) 
        ])

# ---------------------------------------------------------
# 3. Information Value (IV) Helper
# ---------------------------------------------------------
def calculate_iv(df, feature, target):
    """Requirement: Implement logic to calculate IV."""
    data = df[[feature, target]].copy()
    data['target'] = data[target]
    
    # Simple IV calculation logic
    grouped = data.groupby(feature)['target'].agg(['count', 'sum'])
    grouped.columns = ['Total', 'Bad']
    grouped['Good'] = grouped['Total'] - grouped['Bad']
    
    grouped['Dist_Good'] = grouped['Good'] / grouped['Good'].sum()
    grouped['Dist_Bad'] = grouped['Bad'] / grouped['Bad'].sum()
    
    # Use small epsilon to avoid log(0)
    eps = 1e-10
    grouped['WoE'] = np.log((grouped['Dist_Good'] + eps) / (grouped['Dist_Bad'] + eps))
    grouped['IV'] = (grouped['Dist_Good'] - grouped['Dist_Bad']) * grouped['WoE']
    
    return grouped['IV'].replace([np.inf, -np.inf], 0).sum()