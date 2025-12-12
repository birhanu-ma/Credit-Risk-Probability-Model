import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class DataProfiling:
    """Performs data profiling operations on the input dataframe."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
    
    def overview(self):
        print("====== DATA OVERVIEW ======")
        print(f"Shape: {self.df.shape}")
        print("\nData Types:\n", self.df.dtypes)
        print("\nFirst 5 Rows:\n", self.df.head())
    
    def summary_statistics(self):
        print("\n====== SUMMARY STATISTICS ======")
    
        # Identify meaningful numeric columns
        numeric_cols = ['Amount', 'Value']
        numeric_cols = [col for col in numeric_cols if col in self.df.columns]
        
        # Numeric summary
        if numeric_cols:
            print("\n--- Numeric Features ---")
            display(self.df[numeric_cols].describe().transpose())
        
        # Identify categorical columns (exclude IDs)
        id_cols = ['TransactionId', 'BatchId', 'AccountId', 
                   'SubscriptionId', 'CustomerId', 'ProductId']
        cat_cols = [col for col in self.df.select_dtypes(exclude=np.number).columns if col not in id_cols]
        
        if cat_cols:
            print("\n--- Categorical Features ---")
            for col in cat_cols:
                print(f"\nFeature: {col}")
                print(f"Unique values: {self.df[col].nunique()}")
                print(f"Top 5 most frequent values:\n{self.df[col].value_counts().head()}")

    
    def missing_values(self):
        print("\n====== MISSING VALUES ======")
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        mv = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})
        display(mv)
    
    def duplicate_rows(self):
        print("\n====== DUPLICATE ROWS ======")
        dup_count = self.df.duplicated().sum()
        print(f"Duplicate Rows: {dup_count}")
    
    def column_uniques(self):
        print("\n====== UNIQUE VALUES PER COLUMN ======")
        uniques = self.df.nunique().sort_values()
        display(uniques)
    
    def run_all(self):
        """Run all profiling tasks."""
        self.overview()
        self.summary_statistics()
        self.missing_values()
        self.duplicate_rows()
        self.column_uniques()


