import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class ProxyTargetEngineering:
    def __init__(self, df):
        """
        Initialize with the transaction dataframe.
        """
        self.df = df.copy()
        self.rfm = None
        self.scaler = StandardScaler()
        self.kmeans = None
        self.high_risk_cluster_id = None

    def calculate_rfm(self):
        """
        Calculate Recency, Frequency, and Monetary metrics for each customer.
        """
        print("Calculating RFM metrics...")
        self.df['TransactionStartTime'] = pd.to_datetime(self.df['TransactionStartTime'])

        # Snapshot date for recency calculation
        snapshot_date = self.df['TransactionStartTime'].max() + pd.Timedelta(days=1)

        # Aggregate by CustomerId
        self.rfm = self.df.groupby('CustomerId').agg({
            'TransactionStartTime': lambda x: (snapshot_date - x.max()).days,
            'TransactionId': 'count',
            'Amount': lambda x: x[x > 0].sum()  # only positive transactions
        }).rename(columns={
            'TransactionStartTime': 'Recency',
            'TransactionId': 'Frequency',
            'Amount': 'Monetary'
        })

        # Optional: Composite RFM score (for visualization / sanity check)
        self.rfm['RFM_Score'] = (
            self.rfm['Recency'].rank(ascending=False) + 
            self.rfm['Frequency'].rank(ascending=True) + 
            self.rfm['Monetary'].rank(ascending=True)
        )
        return self.rfm

    def cluster_customers(self, n_clusters=3, random_state=42):
        """
        Segment customers using K-Means clustering on Recency, Frequency, Monetary.
        """
        if self.rfm is None:
            self.calculate_rfm()

        print(f"Clustering customers into {n_clusters} clusters...")
        rfm_features = self.rfm[['Recency', 'Frequency', 'Monetary']]

        # Standardize features
        rfm_scaled = self.scaler.fit_transform(rfm_features)

        # Fit KMeans
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.rfm['Cluster'] = self.kmeans.fit_predict(rfm_scaled)

        return self.rfm

    def assign_risk_label(self):
        """
        Assign 'is_high_risk' = 1 to the cluster with lowest engagement
        (low Frequency and Monetary), 0 to others.
        """
        if self.rfm is None or 'Cluster' not in self.rfm.columns:
            self.cluster_customers()

        # Identify high-risk cluster (lowest Frequency + Monetary)
        cluster_stats = self.rfm.groupby('Cluster')[['Frequency','Monetary']].mean()
        cluster_stats['Score'] = cluster_stats['Frequency'] + cluster_stats['Monetary']
        self.high_risk_cluster_id = cluster_stats['Score'].idxmin()

        # Binary target
        self.rfm['is_high_risk'] = (self.rfm['Cluster'] == self.high_risk_cluster_id).astype(int)

        # Human-readable label
        self.rfm['Cluster_Label'] = self.rfm['Cluster'].apply(
            lambda x: 'High Risk' if x == self.high_risk_cluster_id else 'Low Risk'
        )

        print(f"Cluster {self.high_risk_cluster_id} identified as HIGH RISK.")

        return self.rfm

    def get_labeled_data(self):
        """
        Merge the high-risk label back into the original transactions.
        """
        if self.rfm is None or 'is_high_risk' not in self.rfm.columns:
            self.assign_risk_label()

        labeled_df = self.df.merge(
            self.rfm[['is_high_risk','Cluster_Label']],
            on='CustomerId',
            how='left'
        )
        return labeled_df

    def plot_3d_clusters(self):
        """
        3D visualization of Recency, Frequency, Monetary with high-risk highlighted.
        """
        if self.rfm is None or 'Cluster' not in self.rfm.columns:
            self.cluster_customers()
    
        fig = plt.figure(figsize=(12,8))
        ax = fig.add_subplot(111, projection='3d')
    
        # High-risk and Low-risk
        high_risk = self.rfm['is_high_risk'] == 1
        low_risk = ~high_risk
    
        # Plot low-risk customers
        ax.scatter(
            self.rfm.loc[low_risk, 'Recency'], 
            self.rfm.loc[low_risk, 'Frequency'], 
            self.rfm.loc[low_risk, 'Monetary'],
            c='blue', label='Low Risk', alpha=0.6
        )
    
        # Plot high-risk customers
        ax.scatter(
            self.rfm.loc[high_risk, 'Recency'], 
            self.rfm.loc[high_risk, 'Frequency'], 
            self.rfm.loc[high_risk, 'Monetary'],
            c='red', label='High Risk', alpha=0.6
        )
    
        ax.set_xlabel('Recency')
        ax.set_ylabel('Frequency')
        ax.set_zlabel('Monetary')
        ax.set_title('Customer Segments (3D RFM)')
        ax.legend()
    
        # Add extra margin on the right
        plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1)
        plt.show()
    