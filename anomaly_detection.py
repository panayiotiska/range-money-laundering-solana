import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import networkx as nx
from collections import defaultdict
import os
import json
from datetime import datetime

def create_token_features(df):
    """
    Create features for each token based on its transaction patterns
    """
    print("Creating token features...")
    
    # Group by token
    token_groups = df.groupby('AddressTo')
    
    # Initialize features dictionary
    features = []
    
    for token, group in token_groups:
        # Basic transaction counts
        total_txs = len(group)
        buy_txs = len(group[group['EdgeType'] == 'buy'])
        sell_txs = len(group[group['EdgeType'] == 'sell'])
        transfer_txs = len(group[group['EdgeType'] == 'transfer'])
        
        # Create transaction graph
        G = nx.DiGraph()
        for _, tx in group.iterrows():
            G.add_edge(tx['AddressFrom'], tx['AddressTo'], 
                      edge_type=tx['EdgeType'],
                      cardinality=tx['Cardinality'])
        
        # Network features
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        avg_degree = sum(dict(G.degree()).values()) / num_nodes if num_nodes > 0 else 0
        
        # Transaction type ratios
        buy_ratio = buy_txs / total_txs if total_txs > 0 else 0
        sell_ratio = sell_txs / total_txs if total_txs > 0 else 0
        transfer_ratio = transfer_txs / total_txs if total_txs > 0 else 0
        
        # Cardinality features
        avg_cardinality = group['Cardinality'].mean()
        max_cardinality = group['Cardinality'].max()
        std_cardinality = group['Cardinality'].std()
        
        # Unique addresses
        unique_senders = group['AddressFrom'].nunique()
        unique_receivers = group['AddressTo'].nunique()
        
        # Transaction patterns
        avg_txs_per_address = total_txs / (unique_senders + unique_receivers) if (unique_senders + unique_receivers) > 0 else 0
        
        # Create feature dictionary
        token_features = {
            'token': token,
            'total_transactions': total_txs,
            'buy_transactions': buy_txs,
            'sell_transactions': sell_txs,
            'transfer_transactions': transfer_txs,
            'buy_ratio': buy_ratio,
            'sell_ratio': sell_ratio,
            'transfer_ratio': transfer_ratio,
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'avg_degree': avg_degree,
            'avg_cardinality': avg_cardinality,
            'max_cardinality': max_cardinality,
            'std_cardinality': std_cardinality,
            'unique_senders': unique_senders,
            'unique_receivers': unique_receivers,
            'avg_txs_per_address': avg_txs_per_address
        }
        
        features.append(token_features)
    
    return pd.DataFrame(features)

def analyze_features(features_df):
    """
    Print analysis of the created features
    """
    print("\nFeature Analysis:")
    print(f"Total tokens analyzed: {len(features_df)}")
    print("\nTransaction Statistics:")
    print(features_df[['total_transactions', 'buy_transactions', 'sell_transactions', 'transfer_transactions']].describe())
    
    print("\nRatio Statistics:")
    print(features_df[['buy_ratio', 'sell_ratio', 'transfer_ratio']].describe())
    
    print("\nNetwork Statistics:")
    print(features_df[['num_nodes', 'num_edges', 'avg_degree']].describe())
    
    print("\nCardinality Statistics:")
    print(features_df[['avg_cardinality', 'max_cardinality', 'std_cardinality']].describe())
    
    # Save feature statistics to file
    stats = {
        'summary': {
            'total_tokens': len(features_df),
            'timestamp': datetime.now().isoformat()
        },
        'transaction_stats': features_df[['total_transactions', 'buy_transactions', 'sell_transactions', 'transfer_transactions']].describe().to_dict(),
        'ratio_stats': features_df[['buy_ratio', 'sell_ratio', 'transfer_ratio']].describe().to_dict(),
        'network_stats': features_df[['num_nodes', 'num_edges', 'avg_degree']].describe().to_dict(),
        'cardinality_stats': features_df[['avg_cardinality', 'max_cardinality', 'std_cardinality']].describe().to_dict()
    }
    
    with open('data/feature_statistics.json', 'w') as f:
        json.dump(stats, f, indent=2)

def detect_anomalies(features_df, contamination=0.01):
    """
    Apply Isolation Forest to detect anomalies
    """
    print("\nDetecting anomalies...")
    
    # Select features for anomaly detection
    feature_columns = [
        'total_transactions', 'buy_transactions', 'sell_transactions', 'transfer_transactions',
        'buy_ratio', 'sell_ratio', 'transfer_ratio',
        'num_nodes', 'num_edges', 'avg_degree',
        'avg_cardinality', 'max_cardinality', 'std_cardinality',
        'unique_senders', 'unique_receivers', 'avg_txs_per_address'
    ]
    
    # Initialize and fit Isolation Forest
    iso_forest = IsolationForest(
        n_estimators=100,
        max_samples='auto',
        contamination=contamination,
        random_state=42
    )
    
    # Fit and predict
    features_df['anomaly_score'] = iso_forest.fit_predict(features_df[feature_columns])
    features_df['anomaly_score'] = features_df['anomaly_score'].map({1: 0, -1: 1})  # Convert to binary (0: normal, 1: anomaly)
    
    # Calculate anomaly scores
    features_df['isolation_score'] = -iso_forest.score_samples(features_df[feature_columns])
    
    # Print results
    n_anomalies = features_df['anomaly_score'].sum()
    print(f"\nFound {n_anomalies} anomalous tokens ({n_anomalies/len(features_df)*100:.2f}%)")
    
    # Get top anomalies
    top_anomalies = features_df[features_df['anomaly_score'] == 1].sort_values('isolation_score', ascending=False)
    print("\nTop 10 most anomalous tokens:")
    print(top_anomalies[['token', 'total_transactions', 'buy_ratio', 'sell_ratio', 'isolation_score']].head(10))
    
    # Save anomalous tokens to CSV
    top_anomalies.to_csv('data/anomalous_tokens.csv', index=False)
    print("\nAnomalous tokens saved to data/anomalous_tokens.csv")
    
    return features_df

def main():
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Load the data
    print("Loading data...")
    df = pd.read_parquet('data/solana_llm.graph.parquet')
    
    # Create features
    features_df = create_token_features(df)
    
    # Save features
    features_df.to_parquet('data/token_features.parquet')
    print("\nFeatures saved to data/token_features.parquet")
    
    # Analyze features
    analyze_features(features_df)
    
    # Detect anomalies
    features_df = detect_anomalies(features_df)
    
    print("\nAnalysis complete! Check data/anomalous_tokens.csv for the results.")

if __name__ == "__main__":
    main() 