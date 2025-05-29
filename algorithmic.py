import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import os

# Create visuals directory if it doesn't exist
os.makedirs('visuals', exist_ok=True)

# Read the data
print("Loading data...")
df = pd.read_parquet('data/solana_llm.graph.parquet')

# Drop the Amount column if it exists
if 'Amount' in df.columns:
    df = df.drop('Amount', axis=1)

def identify_suspicious_patterns():
    print("Identifying token creators...")
    # Create a dictionary of token creators and their tokens
    creator_tokens = defaultdict(set)
    for _, row in df[df['EdgeType'] == 'create_token'].iterrows():
        creator_tokens[row['AddressFrom']].add(row['AddressTo'])
    
    print(f"Found {len(creator_tokens)} token creators")
    print(f"Total tokens created: {sum(len(tokens) for tokens in creator_tokens.values())}")
    
    # Create a set of all created tokens for faster lookup
    all_created_tokens = set()
    for tokens in creator_tokens.values():
        all_created_tokens.update(tokens)
    
    print(f"Unique tokens created: {len(all_created_tokens)}")
    
    print("\nAnalyzing trading patterns...")
    suspicious_cases = []
    
    # Pre-filter transactions to only include buy and sell
    trade_df = df[df['EdgeType'].isin(['buy', 'sell'])]
    print(f"Total buy/sell transactions: {len(trade_df)}")
    
    # Create a dictionary to store token transactions
    token_txs = defaultdict(lambda: {'buy': 0, 'sell': 0, 'creator_involved': 0, 'creator_buys': 0, 'creator_sells': 0})
    
    # Process all trades in a single pass
    for _, row in trade_df.iterrows():
        # For buy/sell transactions, the token is always in AddressTo
        token = row['AddressTo']
        if token in all_created_tokens:
            token_txs[token][row['EdgeType']] += 1
            # Check if creator is involved
            for creator, tokens in creator_tokens.items():
                if token in tokens:
                    if row['AddressFrom'] == creator or row['AddressTo'] == creator:
                        token_txs[token]['creator_involved'] += 1
                        if row['EdgeType'] == 'buy':
                            token_txs[token]['creator_buys'] += 1
                        else:
                            token_txs[token]['creator_sells'] += 1
    
    print(f"\nFound {len(token_txs)} tokens with trading activity")
    
    if len(token_txs) > 0:
        # Debug: Show distribution of trade counts
        trade_counts = [stats['buy'] + stats['sell'] for stats in token_txs.values()]
        print("\nTrade count distribution:")
        print(f"Min trades: {min(trade_counts)}")
        print(f"Max trades: {max(trade_counts)}")
        print(f"Mean trades: {sum(trade_counts)/len(trade_counts):.2f}")
        
        # Debug: Show distribution of creator involvement
        creator_involvement = [stats['creator_involved'] for stats in token_txs.values()]
        print("\nCreator involvement distribution:")
        print(f"Min involvement: {min(creator_involvement)}")
        print(f"Max involvement: {max(creator_involvement)}")
        print(f"Mean involvement: {sum(creator_involvement)/len(creator_involvement):.2f}")
        
        # Debug: Show distribution of creator ratios
        creator_ratios = [stats['creator_involved'] / (stats['buy'] + stats['sell']) 
                         for stats in token_txs.values() 
                         if stats['buy'] + stats['sell'] > 0]
        print("\nCreator ratio distribution:")
        print(f"Min ratio: {min(creator_ratios):.2%}")
        print(f"Max ratio: {max(creator_ratios):.2%}")
        print(f"Mean ratio: {sum(creator_ratios)/len(creator_ratios):.2%}")
    
    # Identify suspicious cases with adjusted criteria
    for token, stats in token_txs.items():
        total_trades = stats['buy'] + stats['sell']
        creator_ratio = stats['creator_involved'] / total_trades if total_trades > 0 else 0
        
        # Debug: Print criteria for each token with significant trading
        if total_trades >= 5:  # First filter
            print(f"\nToken: {token[:8]}...")
            print(f"Total trades: {total_trades}")
            print(f"Creator ratio: {creator_ratio:.2%}")
            print(f"Creator buys: {stats['creator_buys']}")
            print(f"Creator sells: {stats['creator_sells']}")
            print(f"Creator involved: {stats['creator_involved']}")
        
        # New criteria:
        # 1. At least 5 trades (reduced from 10)
        # 2. Creator is involved in at least 20% of trades
        # 3. Creator has both buy and sell transactions
        # 4. Creator's involvement is significant (at least 2 trades)
        if (total_trades >= 5 and  # At least 5 trades
            creator_ratio >= 0.2 and  # Creator involved in at least 20% of trades
            stats['creator_buys'] > 0 and stats['creator_sells'] > 0 and  # Both buy and sell exist
            stats['creator_involved'] >= 2):  # Creator involved in at least 2 trades
            
            # Find the creator of this token
            creator = next(creator for creator, tokens in creator_tokens.items() if token in tokens)
            
            suspicious_cases.append({
                'creator': creator,
                'token': token,
                'total_trades': total_trades,
                'buy_trades': stats['buy'],
                'sell_trades': stats['sell'],
                'creator_involved': stats['creator_involved'],
                'creator_buys': stats['creator_buys'],
                'creator_sells': stats['creator_sells'],
                'creator_ratio': creator_ratio
            })
    
    return suspicious_cases

def visualize_suspicious_case(creator, token, case_number, stats):
    # Get all transactions for this token (only buy and sell)
    token_txs = df[
        ((df['AddressTo'] == token) | (df['AddressFrom'] == token)) &
        (df['EdgeType'].isin(['buy', 'sell']))
    ].copy()
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes and edges
    for _, tx in token_txs.iterrows():
        G.add_edge(tx['AddressFrom'], tx['AddressTo'], 
                  edge_type=tx['EdgeType'],
                  cardinality=tx['Cardinality'])
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Use spring layout for better visualization
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Draw edges
    edge_colors = ['red' if G[u][v]['edge_type'] == 'sell' else 'green' 
                  for u, v in G.edges()]
    
    # Draw the network
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, alpha=0.6)
    nx.draw_networkx_nodes(G, pos, 
                          node_color='lightblue',
                          node_size=500)
    
    # Highlight the creator node
    if creator in G.nodes():
        nx.draw_networkx_nodes(G, pos,
                             nodelist=[creator],
                             node_color='red',
                             node_size=1000)
    
    # Add labels only for important nodes
    labels = {node: node[:8] + '...' if node != creator else 'CREATOR'
             for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    
    # Add title with statistics
    plt.title(f'Suspicious Case #{case_number}\n'
             f'Token: {token[:8]}...\n'
             f'Total Trades: {stats["total_trades"]} | '
             f'Creator Involvement: {stats["creator_involved"]} ({stats["creator_ratio"]:.1%})\n'
             f'Creator Buys: {stats["creator_buys"]} | '
             f'Creator Sells: {stats["creator_sells"]}')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'visuals/suspicious_case_{case_number}.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print("Starting analysis...")
    suspicious_cases = identify_suspicious_patterns()
    
    print(f"\nFound {len(suspicious_cases)} suspicious cases")
    
    # Sort cases by creator involvement ratio
    suspicious_cases.sort(key=lambda x: x['creator_ratio'], reverse=True)
    
    # Visualize top 10 most suspicious cases
    for i, case in enumerate(suspicious_cases[:10]):
        print(f"\nVisualizing case {i+1}:")
        print(f"Creator: {case['creator'][:8]}...")
        print(f"Token: {case['token'][:8]}...")
        print(f"Total trades: {case['total_trades']}")
        print(f"Buy trades: {case['buy_trades']}")
        print(f"Sell trades: {case['sell_trades']}")
        print(f"Creator involvement: {case['creator_involved']} ({case['creator_ratio']:.1%})")
        print(f"Creator buys: {case['creator_buys']}")
        print(f"Creator sells: {case['creator_sells']}")
        
        visualize_suspicious_case(case['creator'], case['token'], i+1, case)
    
    print("\nVisualizations have been saved to the 'visuals' directory")

if __name__ == "__main__":
    main() 