import pandas as pd
import pytest
from src.train import CreditRiskModelTrainer

def test_train_model():
    # Sample data for testing
    df = pd.DataFrame({
        'CustomerId': ['C1', 'C2', 'C3', 'C4'],
        'Amount': [100, 200, 150, 120],
        'is_high_risk': [0, 1, 0, 1],
        'CurrencyCode': [1, 2, 1, 2],
        'CountryCode': [254, 1, 254, 1],
        'ProductCategory': [3, 1, 2, 3],
        'ChannelId': [2, 1, 2, 1],
        'PricingStrategy': [1, 2, 1, 2]
    })

    trainer = CreditRiskModelTrainer(df)
    model, metrics = trainer.train_model('logistic_regression')
    
    # Verify that the metrics are returned correctly
    assert 'roc_auc' in metrics
    assert 'accuracy' in metrics
    assert 'f1_score' in metrics
    assert 'precision' in metrics

def test_best_model_selection():
    df = pd.DataFrame({
        'CustomerId': ['C1', 'C2', 'C3', 'C4'],
        'Amount': [100, 200, 150, 120],
        'is_high_risk': [0, 1, 0, 1],
        'CurrencyCode': [1, 2, 1, 2],
        'CountryCode': [254, 1, 254, 1],
        'ProductCategory': [3, 1, 2, 3],
        'ChannelId': [2, 1, 2, 1],
        'PricingStrategy': [1, 2, 1, 2]
    })

    trainer = CreditRiskModelTrainer(df)
    trainer.train_model('logistic_regression')
    best_model = trainer.get_best_model()
    
    # Ensure that the best model is returned
    assert best_model is not None
