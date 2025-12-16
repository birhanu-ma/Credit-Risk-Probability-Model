import os
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

class CreditRiskModelTrainer:
    def __init__(self, df, target_col='is_high_risk', test_size=0.2, random_state=42, drop_cols=['CustomerId'],
                 mlflow_tracking_uri=None):
        self.df = df.copy()
        self.target_col = target_col
        self.random_state = random_state
        self.test_size = test_size
        self.df.drop(columns=drop_cols, inplace=True, errors='ignore')

        # Train/test split
        self.X_train, self.X_test, self.y_train, self.y_test = self._train_test_split()
        self._scale_features()

        # Store models
        self.models = {}
        self.best_model_name = None

        # Setup MLflow
        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment("Credit_Risk_Model")
        self.client = MlflowClient()

        # Define Project Root and the 'models' container folder
        # Path goes from 'notebook' up to 'Credit-Risk-Probability-Model'
        self.project_root = Path(os.getcwd()).resolve().parent 
        self.models_dir = self.project_root / "models"
        
        # Create the 'models' directory in root if it doesn't exist
        if not self.models_dir.exists():
            self.models_dir.mkdir(parents=True)

    def _train_test_split(self):
        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]
        return train_test_split(X, y, test_size=self.test_size, stratify=y, random_state=self.random_state)

    def _scale_features(self):
        numeric_cols = self.X_train.select_dtypes(include=np.number).columns
        self.scaler = StandardScaler()
        self.X_train[numeric_cols] = self.scaler.fit_transform(self.X_train[numeric_cols])
        self.X_test[numeric_cols] = self.scaler.transform(self.X_test[numeric_cols])

    def train_model(self, model_name='logistic_regression', params=None, search='grid', cv=5, n_iter=10):
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
            else:
                searcher = RandomizedSearchCV(model, param_distributions=params, n_iter=n_iter,
                                              cv=cv, scoring='roc_auc', random_state=self.random_state, n_jobs=-1)
            searcher.fit(self.X_train, self.y_train)
            best_model = searcher.best_estimator_
            best_params = searcher.best_params_
        else:
            model.fit(self.X_train, self.y_train)
            best_model = model
            best_params = model.get_params()

        # Evaluate
        y_pred = best_model.predict(self.X_test)
        y_prob = best_model.predict_proba(self.X_test)[:, 1]

        metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred),
            'recall': recall_score(self.y_test, y_pred),
            'f1_score': f1_score(self.y_test, y_pred),
            'roc_auc': roc_auc_score(self.y_test, y_prob)
        }

        # 1. Log to MLflow (Keeps the history in notebook/mlruns)
        registry_name = f"Credit_Risk_{model_name.title()}"
        with mlflow.start_run(run_name=model_name):
            mlflow.log_params(best_params)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(sk_model=best_model, artifact_path="model", registered_model_name=registry_name)

        # 2. SAVE TO ROOT models/ folder
        # Path will be: Credit-Risk-Probability-Model/models/random_forest
        specific_model_path = self.models_dir / model_name
        
        if specific_model_path.exists():
            shutil.rmtree(specific_model_path)
        
        mlflow.sklearn.save_model(sk_model=best_model, path=str(specific_model_path))

        # Store locally
        self.models[model_name] = {'model': best_model, 'metrics': metrics, 'registry_name': registry_name}

        if self.best_model_name is None or metrics['roc_auc'] > self.models[self.best_model_name]['metrics']['roc_auc']:
            self.best_model_name = model_name

        print(f"{model_name} trained. Saved to: {specific_model_path}")
        return best_model, metrics

    def evaluate_all_models(self):
        return pd.DataFrame({name: data['metrics'] for name, data in self.models.items()}).T

    def get_best_model(self):
        if self.best_model_name:
            return self.models[self.best_model_name]['model']
        return None

    def promote_best_model_to_production(self):
        best_registry_name = self.models[self.best_model_name]['registry_name']
        versions = self.client.get_latest_versions(best_registry_name)
        best_version = max([int(v.version) for v in versions])
        self.client.transition_model_version_stage(
            name=best_registry_name, version=best_version, stage="Production", archive_existing_versions=True
        )
        print(f"Model {best_registry_name} version {best_version} is now Production.")
