import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn

class CreditRiskModelTrainer:
    """
    Credit Risk Model Trainer for Task 5
    - Handles train/test split
    - Trains multiple models with optional hyperparameter tuning
    - Logs metrics and models to MLflow
    - Automatically tracks and returns the best model by ROC-AUC
    """

    def __init__(self, df, target_col='is_high_risk', test_size=0.2, random_state=42, drop_cols=['CustomerId']):
        self.df = df.copy()
        self.target_col = target_col
        self.random_state = random_state
        self.test_size = test_size

        # Drop unnecessary columns like CustomerId
        self.df.drop(columns=drop_cols, inplace=True, errors='ignore')

        # Train/test split
        self.X_train, self.X_test, self.y_train, self.y_test = self._train_test_split()

        # Scale numerical features
        self._scale_features()

        # Store trained models
        self.models = {}
        self.best_model_name = None

    def _train_test_split(self):
        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]
        return train_test_split(
            X, y, test_size=self.test_size, stratify=y, random_state=self.random_state
        )

    def _scale_features(self):
        numeric_cols = self.X_train.select_dtypes(include=np.number).columns
        self.scaler = StandardScaler()
        self.X_train[numeric_cols] = self.scaler.fit_transform(self.X_train[numeric_cols])
        self.X_test[numeric_cols] = self.scaler.transform(self.X_test[numeric_cols])

    def train_model(self, model_name='logistic_regression', params=None, search='grid', cv=5, n_iter=10):
        """Train a model with optional hyperparameter tuning and log to MLflow."""
        # Initialize model
        if model_name.lower() == 'logistic_regression':
            model = LogisticRegression(random_state=self.random_state, max_iter=1000)
        elif model_name.lower() == 'random_forest':
            model = RandomForestClassifier(random_state=self.random_state)
        elif model_name.lower() == 'gradient_boosting':
            model = GradientBoostingClassifier(random_state=self.random_state)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        # Hyperparameter tuning
        if params:
            if search == 'grid':
                searcher = GridSearchCV(model, param_grid=params, cv=cv, scoring='roc_auc', n_jobs=-1)
            elif search == 'random':
                searcher = RandomizedSearchCV(
                    model, param_distributions=params, n_iter=n_iter,
                    cv=cv, scoring='roc_auc', random_state=self.random_state, n_jobs=-1
                )
            searcher.fit(self.X_train, self.y_train)
            best_model = searcher.best_estimator_
            best_params = searcher.best_params_
        else:
            model.fit(self.X_train, self.y_train)
            best_model = model
            best_params = model.get_params()

        # Evaluate model
        y_pred = best_model.predict(self.X_test)
        y_prob = best_model.predict_proba(self.X_test)[:, 1]

        metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred),
            'recall': recall_score(self.y_test, y_pred),
            'f1_score': f1_score(self.y_test, y_pred),
            'roc_auc': roc_auc_score(self.y_test, y_prob)
        }

        # MLflow tracking
        mlflow.set_experiment("Credit_Risk_Model")
        with mlflow.start_run(run_name=model_name):
            mlflow.log_params(best_params)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(best_model, artifact_path=f"{model_name}_model")

        # Store model and metrics
        self.models[model_name] = {'model': best_model, 'metrics': metrics, 'params': best_params}

        # Update best model by ROC-AUC
        if self.best_model_name is None or metrics['roc_auc'] > self.models[self.best_model_name]['metrics']['roc_auc']:
            self.best_model_name = model_name

        print(f"{model_name} trained. ROC-AUC: {metrics['roc_auc']:.4f}")
        return best_model, metrics

    def evaluate_all_models(self):
        """Return a DataFrame of metrics for all trained models."""
        return pd.DataFrame({name: data['metrics'] for name, data in self.models.items()}).T

    def get_best_model(self):
        """Return the best model based on ROC-AUC."""
        if self.best_model_name:
            return self.models[self.best_model_name]['model']
        return None

    def predict(self, df_new):
        """Make predictions on new data."""
        df_new = df_new.copy()
        df_new.drop(columns=['CustomerId'], inplace=True, errors='ignore')
        numeric_cols = df_new.select_dtypes(include=np.number).columns
        df_new[numeric_cols] = self.scaler.transform(df_new[numeric_cols])
        model = self.get_best_model()
        if model:
            y_prob = model.predict_proba(df_new)[:, 1]
            y_pred = model.predict(df_new)
            return pd.DataFrame({'prediction': y_pred, 'risk_probability': y_prob})
        else:
            raise ValueError("No trained model found. Train a model first.")

