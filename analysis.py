import pandas as pd

df = pd.read_parquet('data/solana_llm.graph.parquet')

# Drop the Amount column (inconsistent)
df = df.drop('Amount', axis=1)

print("First 5 rows:")
print(df.head())
print("\nLast 5 rows:")
print(df.tail())

# Display basic information about the DataFrame
print("\nDataFrame Info:")
df.info()

# Display DataFrame shape
print("\nDataFrame Shape:")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

# Display basic statistics
print("\nBasic Statistics:")
print(df.describe())

# Display column names
print("\nColumn Names:")
print(df.columns.tolist())

# Display number of null values
print("\nNull Values Count:")
print(df.isnull().sum())

# Get number of unique AddressFrom
unique_from_addresses = df['AddressFrom'].nunique()
print(f"\nNumber of unique 'AddressFrom' addresses: {unique_from_addresses}")

# Get number of unique AddressTo
unique_to_addresses = df['AddressTo'].nunique()
print(f"Number of unique 'AddressTo' addresses: {unique_to_addresses}")

# Analyze EdgeType distribution
print("\nEdgeType Distribution:")
print(df['EdgeType'].value_counts())
print("\nEdgeType Distribution (percentage):")
print(df['EdgeType'].value_counts(normalize=True) * 100)

# Analyze AddressTo patterns
print("\nAddressTo length distribution:")
print(df['AddressTo'].str.len().value_counts().sort_index().head(10))
print("\nMost common AddressTo values:")
print(df['AddressTo'].value_counts().head(10))

# Analyze AddressFrom patterns
print("\nAddressFrom length distribution:")
print(df['AddressFrom'].str.len().value_counts().sort_index().head(10))
print("\nMost common AddressFrom values:")
print(df['AddressFrom'].value_counts().head(10))

# Analyze Cardinality patterns
print("\nCardinality distribution:")
print(df['Cardinality'].value_counts().sort_index().head(10))
print("\nTransactions with Cardinality > 1:")
print(len(df[df['Cardinality'] > 1]))

# Analyze relationships between EdgeType and Cardinality
print("\nAverage Cardinality by EdgeType:")
print(df.groupby('EdgeType')['Cardinality'].mean())

# Check for any patterns in AddressTo that might indicate special addresses
print("\nAddressTo starting with common prefixes:")
print(df['AddressTo'].str[:3].value_counts().head(10))

# Check for any patterns in AddressFrom that might indicate special addresses
print("\nAddressFrom starting with common prefixes:")
print(df['AddressFrom'].str[:3].value_counts().head(10))

# Filter rows where AddressTo string length > 15
long_addresses = df[df['AddressTo'].str.len() > 15]
print(f"\nNumber of transactions with AddressTo longer than 15 characters: {len(long_addresses)}")
print("\nSample of these transactions:")
print(long_addresses.head())

print("Columns:")
print(df.columns)