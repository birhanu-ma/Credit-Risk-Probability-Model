import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys, os

class EDAAnalysis:
    """Performs visual EDA quickly without blocking."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy() # Work on a copy to prevent modifying the original DataFrame accidentally
        self.num_cols = df.select_dtypes(include=np.number).columns
        self.cat_cols = df.select_dtypes(exclude=np.number).columns

    def safe_show(self):
        """Non-blocking show that does NOT freeze the script."""
        plt.show(block=False)
        plt.pause(0.001)   # tiny pause to render
        plt.close()

    def distribution_numerical(self):
        print("\n====== DISTRIBUTION OF NUMERICAL FEATURES ======")
        for col in self.num_cols:
            plt.figure(figsize=(8, 4))
            sns.histplot(self.df[col], kde=True, bins=30)
            plt.title(f"Distribution: {col}")
            self.safe_show()

    def distribution_categorical(self):
        print("\n====== DISTRIBUTION OF CATEGORICAL FEATURES ======")
        for col in self.cat_cols:
            unique_count = self.df[col].nunique()
            
            # Skip columns with huge cardinality
            if unique_count > 30:
                print(f"Skipping '{col}' (too many categories: {unique_count})")
                continue
            
            plt.figure(figsize=(10, 4))
            self.df[col].value_counts().plot(kind='bar')
            plt.title(f"Categorical Count: {col}")
            plt.ylabel("Count")
            self.safe_show()
    

    def correlation_heatmap(self):
        print("\n====== CORRELATION HEATMAP ======")

        # Convert numeric-like columns correctly
        df_numeric = self.df[self.num_cols].apply(pd.to_numeric, errors='coerce')
    
        # Drop columns that became all-NaN (not truly numeric)
        df_numeric = df_numeric.dropna(axis=1, how='all')
    
        if df_numeric.shape[1] < 2:
            print("⛔ Not enough valid numeric columns to compute correlation.")
            return
    
        corr = df_numeric.corr()
    
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Heatmap")
        self.safe_show()


    def boxplot_outliers(self):
        print("\n====== OUTLIER DETECTION (BOX PLOTS - Seaborn Vertical) ======")
        for col in self.num_cols:
            plt.figure(figsize=(6, 4)) # Adjust figure size
            
            # Pass the column to the 'y' parameter for a vertical boxplot
            sns.boxplot(y=self.df[col])
            
            # Set the title
            plt.title(f"Boxplot: {col}")
            
            self.safe_show()

    def calculate_iqr_fences(self, col):
        """
        Calculates the lower and upper fences for a given column (1.5 * IQR rule).
        """
        Q1 = self.df[col].quantile(0.25)
        Q3 = self.df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_fence = Q1 - 1.5 * IQR
        upper_fence = Q3 + 1.5 * IQR
        
        print(f"--- Fences for {col} ---")
        print(f"Q1 (25th Pct): {Q1:,.2f}")
        print(f"Q3 (75th Pct): {Q3:,.2f}")
        print(f"IQR: {IQR:,.2f}")
        print(f"Lower Fence (Outlier boundary): {lower_fence:,.2f}")
        print(f"Upper Fence (Outlier boundary): {upper_fence:,.2f}")
        
        # Return the mask of identified outliers
        outlier_mask = (self.df[col] < lower_fence) | (self.df[col] > upper_fence)
        return outlier_mask

    def run_all(self):
        print("====== Starting EDA Analysis ======")
        self.distribution_numerical()
        self.distribution_categorical()
        self.correlation_heatmap()
        self.boxplot_outliers()

        # FIXED: Iterate over numerical columns to avoid TypeError
        print("\n====== IQR OUTLIER FENCE CALCULATION ======")
        for col in self.num_cols:
            self.calculate_iqr_fences(col) 

        print("\n✔ All plots displayed successfully (no freezing).")

    # BONUS METHOD: Example of how to remove the identified outliers
    def remove_outliers(self, col):
        """Removes rows identified as outliers in the specified column using IQR."""
        
        # Get the outlier mask using the existing function
        outlier_mask = self.calculate_iqr_fences(col)
        
        total_rows = len(self.df)
        outlier_count = outlier_mask.sum()
        outlier_percent = (outlier_count / total_rows) * 100
        
        # Drop the outliers and update the DataFrame
        self.df = self.df[~outlier_mask].copy()
        
        print(f"Outliers removed for {col}: {outlier_count} ({outlier_percent:.2f}%)")
        print(f"New DataFrame size: {len(self.df)} rows.")

    def save_processed_data(self, file_path: str):
        """
        Saves the current state of the DataFrame (self.df) to a CSV file,
        creating necessary directories if they don't exist.
        """
        
        # 1. Determine the directory path
        directory = os.path.dirname(file_path)
        
        # 2. Create the directory if it doesn't exist
        if directory and not os.path.exists(directory):
            print(f"Creating directory: {directory}")
            # Use os.makedirs with exist_ok=True to handle existing paths and create nested paths
            os.makedirs(directory, exist_ok=True)
        
        # 3. Save the DataFrame to CSV
        self.df.to_csv(file_path, index=False)
        print(f"\n✅ Processed data successfully saved to: {file_path}")
        print(f"Final DataFrame size: {len(self.df)} rows.")