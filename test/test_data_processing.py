import pandas as pd
import pytest
from src.data_processing import FeatureEngineering
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from category_encoders import WOEEncoder

def test_feature_engineering_pipeline():
    data = pd.DataFrame({
        'CustomerId': ['C1', 'C1', 'C2'],
        'Amount': [100, 200, 300],
        'TransactionStartTime': ['2025-12-16 10:00:00', '2025-12-16 12:00:00', '2025-12-16 15:00:00'],
        'CurrencyCode': [1, 1, 2],
        'CountryCode': [254, 254, 1],
        'ProductCategory': [3, 3, 1],
        'ChannelId': [2, 2, 1],
        'PricingStrategy': [1, 1, 2]
    })

    fe = FeatureEngineering()
    pipeline = fe.build_pipeline()
    processed = pipeline.fit_transform(data, y=None)
    
    # Ensure required features are added after transformations
    assert 'Total_Transaction_Amount' in processed.columns
    assert 'Average_Transaction_Amount' in processed.columns
    assert 'Transaction_Hour' in processed.columns
    assert 'Transaction_Month' in processed.columns
    
    # Ensure no missing values after imputation
    assert processed.isnull().sum().sum() == 0

def test_imputation():
    # Testing missing data imputation
    data_with_missing = pd.DataFrame({
        'CustomerId': ['C1', 'C1', 'C2'],
        'Amount': [100, None, 300],
        'CurrencyCode': [1, 1, 2]
    })
    fe = FeatureEngineering()
    pipeline = fe.build_pipeline()
    processed = pipeline.fit_transform(data_with_missing)
    
    # Check if missing 'Amount' was imputed
    assert processed['Amount'].isnull().sum() == 0

def test_woe_encoding():
    data = pd.DataFrame({
        'CustomerId': ['C1', 'C2', 'C3'],
        'CurrencyCode': [1, 1, 2],
        'CountryCode': [254, 254, 1],
    })
    
    fe = FeatureEngineering()
    pipeline = fe.build_pipeline()
    processed = pipeline.fit_transform(data)
    
    # Ensure WOE encoding has been applied
    assert 'CurrencyCode' not in processed.columns
    assert 'CountryCode' not in processed.columns
    assert 'CurrencyCode_WOE' in processed.columns
