import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import os
from matplotlib.patches import Patch
import seaborn as sns

# Set the style for all plots
plt.style.use('dark_background')
sns.set_theme(style="darkgrid")

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
    # Get all transactions for this token (buy, sell, and transfer)
    token_txs = df[
        ((df['AddressTo'] == token) | (df['AddressFrom'] == token)) &
        (df['EdgeType'].isin(['buy', 'sell', 'transfer']))
    ].copy()
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes and edges
    for _, tx in token_txs.iterrows():
        G.add_edge(tx['AddressFrom'], tx['AddressTo'], 
                  edge_type=tx['EdgeType'],
                  cardinality=tx['Cardinality'])
    
    # Set up the figure with a dark background
    plt.figure(figsize=(15, 10), facecolor='#1a1a1a')
    ax = plt.gca()
    ax.set_facecolor('#1a1a1a')
    
    # Use spring layout with adjusted parameters for better visualization
    pos = nx.spring_layout(G, k=2, iterations=100, seed=42)
    
    # Separate edges by type
    buy_edges = [(u, v) for u, v, d in G.edges(data=True) if d['edge_type'] == 'buy']
    sell_edges = [(u, v) for u, v, d in G.edges(data=True) if d['edge_type'] == 'sell']
    transfer_edges = [(u, v) for u, v, d in G.edges(data=True) if d['edge_type'] == 'transfer']
    
    # Draw buy edges in green with arrows
    nx.draw_networkx_edges(G, pos, edgelist=buy_edges, 
                          edge_color='#00ff00', alpha=0.7,
                          arrows=True, arrowsize=20, width=2,
                          connectionstyle='arc3,rad=0.1')
    
    # Draw sell edges in red with arrows
    nx.draw_networkx_edges(G, pos, edgelist=sell_edges,
                          edge_color='#ff0000', alpha=0.7,
                          arrows=True, arrowsize=20, width=2,
                          connectionstyle='arc3,rad=-0.1')
    
    # Draw transfer edges in orange with arrows
    nx.draw_networkx_edges(G, pos, edgelist=transfer_edges,
                          edge_color='#ffa500', alpha=0.7,
                          arrows=True, arrowsize=20, width=2,
                          connectionstyle='arc3,rad=0.2')
    
    # Draw regular nodes with a modern style
    nx.draw_networkx_nodes(G, pos,
                          node_color='#4a90e2',
                          node_size=800,
                          alpha=0.8,
                          edgecolors='white',
                          linewidths=2)
    
    # Highlight the creator node
    if creator in G.nodes():
        nx.draw_networkx_nodes(G, pos,
                             nodelist=[creator],
                             node_color='#ff6b6b',
                             node_size=1200,
                             alpha=0.9,
                             edgecolors='white',
                             linewidths=3)
    
    # Add labels with better formatting
    labels = {node: node[:8] + '...' if node != creator else 'CREATOR'
             for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, 
                           font_size=10,
                           font_color='white',
                           font_weight='bold')
    
    # Create custom legend
    legend_elements = [
        Patch(facecolor='#4a90e2', edgecolor='white', label='Trader'),
        Patch(facecolor='#ff6b6b', edgecolor='white', label='Token Creator'),
        plt.Line2D([0], [0], color='#00ff00', lw=2, label='Buy Transaction'),
        plt.Line2D([0], [0], color='#ff0000', lw=2, label='Sell Transaction'),
        plt.Line2D([0], [0], color='#ffa500', lw=2, label='Transfer')
    ]
    
    # Add legend with custom styling
    legend = plt.legend(handles=legend_elements, 
                       loc='upper right',
                       facecolor='#1a1a1a',
                       edgecolor='none',
                       labelcolor='white',
                       fontsize=10)
    
    # Add title with statistics in a modern style
    title_text = (
        f'Suspicious Trading Pattern Analysis\n'
        f'Case #{case_number}\n\n'
        f'Token: {token[:8]}...\n'
        f'Total Trades: {stats["total_trades"]} | '
        f'Creator Involvement: {stats["creator_involved"]} ({stats["creator_ratio"]:.1%})\n'
        f'Creator Buys: {stats["creator_buys"]} | '
        f'Creator Sells: {stats["creator_sells"]}'
    )
    plt.title(title_text, color='white', pad=20, fontsize=12)
    
    # Add explanatory text
    explanation = (
        "This visualization shows trading patterns for a potentially suspicious token.\n"
        "Green arrows indicate buy transactions, red arrows indicate sell transactions,\n"
        "and orange arrows show transfers. The red node represents the token creator,\n"
        "who is actively involved in trading. Suspicious patterns include high creator\n"
        "involvement and both buy/sell activity."
    )
    plt.figtext(0.02, 0.02, explanation, color='white', fontsize=10, wrap=True)
    
    plt.axis('off')
    plt.tight_layout()
    
    # Save with high DPI and transparent background
    plt.savefig(f'visuals/suspicious_case_{case_number}.png', 
                dpi=300, 
                bbox_inches='tight',
                facecolor='#1a1a1a',
                edgecolor='none')
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