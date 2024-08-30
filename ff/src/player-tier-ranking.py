import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import jenks_natural_breaks as jnb
import argparse
import os

def load_csv_to_dataframe(file_path):
    """Load CSV file into a pandas DataFrame."""
    return pd.read_csv(file_path)

def extract_position(df):
    """Extract the general position from the 'POS' column."""
    df['Position'] = df['POS'].str.extract('([A-Z]+)')
    return df

def kmeans_tiers(data, num_tiers=5):
    """Create tiers using K-means clustering."""
    values = data.values.reshape(-1, 1)
    kmeans = KMeans(n_clusters=num_tiers, random_state=42)
    tiers = kmeans.fit_predict(values)
    tier_centers = kmeans.cluster_centers_.flatten()
    tier_order = np.argsort(tier_centers)[::-1]
    tier_map = {old: new for new, old in enumerate(tier_order, 1)}
    return pd.Series(tiers).map(tier_map)

def jenks_tiers(data, num_tiers=5):
    """Create tiers using Jenks Natural Breaks."""
    breaks = jnb(data, nb_class=num_tiers)
    return pd.cut(data, bins=breaks, labels=range(1, num_tiers+1), include_lowest=True)

def quantile_tiers(data, num_tiers=5):
    """Create tiers using quantiles."""
    return pd.qcut(data, q=num_tiers, labels=range(1, num_tiers+1))

def std_dev_tiers(data, std_dev_range=0.5):
    """Create tiers based on standard deviations from the mean."""
    mean = data.mean()
    std = data.std()
    return pd.cut(data, 
                  bins=[-np.inf, mean-2*std_dev_range*std, mean-std_dev_range*std, 
                        mean+std_dev_range*std, mean+2*std_dev_range*std, np.inf],
                  labels=[5, 4, 3, 2, 1])

def equal_interval_tiers(data, num_tiers=5):
    """Create tiers using equal intervals."""
    return pd.cut(data, bins=num_tiers, labels=range(1, num_tiers+1))

def create_tiers(df, column_name, method='kmeans', by_position=False, **kwargs):
    """Create tiers based on specified method and parameters."""
    tier_methods = {
        'kmeans': kmeans_tiers,
        'jenks': jenks_tiers,
        'quantile': quantile_tiers,
        'std_dev': std_dev_tiers,
        'equal_interval': equal_interval_tiers
    }
    
    if method not in tier_methods:
        raise ValueError("Invalid tiering method specified.")
    
    tier_func = tier_methods[method]
    
    if by_position:
        df = extract_position(df)
        df['Tier'] = df.groupby('Position')[column_name].transform(lambda x: tier_func(x, **kwargs))
    else:
        df['Tier'] = tier_func(df[column_name], **kwargs)
    
    return df

def save_csv(df, output_file):
    """Save DataFrame to CSV file."""
    df.to_csv(output_file, index=False)
    print(f"Saved rankings to {output_file}")

def save_position_csv(df, output_dir, method):
    """Save individual CSV files for each position."""
    os.makedirs(output_dir, exist_ok=True)
    for position in df['Position'].unique():
        position_df = df[df['Position'] == position].sort_values('Tier')
        output_file = os.path.join(output_dir, f'{position}_{method}_tiers.csv')
        position_df.to_csv(output_file, index=False)
        print(f"Saved {position} rankings to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Create tiers for fantasy football rankings.")
    parser.add_argument("input_file", help="Path to the input CSV file")
    parser.add_argument("--column", default="VBD", help="Column name to use for tiering (default: VBD)")
    parser.add_argument("--method", default="kmeans", choices=['kmeans', 'jenks', 'quantile', 'std_dev', 'equal_interval'],
                        help="Tiering method to use (default: kmeans)")
    parser.add_argument("--num_tiers", type=int, default=5, help="Number of tiers to create (default: 5)")
    parser.add_argument("--by_position", action="store_true", help="Create tiers by position")
    parser.add_argument("--output", default="rankings_tiers.csv", help="Output file name (default: rankings_tiers.csv)")
    parser.add_argument("--output_dir", default="position_tiers", help="Directory to save position-specific CSV files")
    
    args = parser.parse_args()
    
    df = load_csv_to_dataframe(args.input_file)
    
    df_tiered = create_tiers(df, args.column, method=args.method, by_position=args.by_position, num_tiers=args.num_tiers)
    
    # Sort the DataFrame by the specified column
    df_tiered = df_tiered.sort_values(args.column, ascending=False)
    
    if args.by_position:
        # Save position-specific CSV files
        save_position_csv(df_tiered, args.output_dir, args.method)
        
        # Save overall rankings
        overall_output = os.path.join(args.output_dir, f'overall_{args.method}_tiers.csv')
        save_csv(df_tiered, overall_output)
    else:
        # Save single CSV file with all rankings
        save_csv(df_tiered, args.output)
    
    # Display a sample of the results
    print(f"\nSample of {args.method} tiers:")
    if args.by_position:
        print(df_tiered[['PLAYER', 'Position', args.column, 'Tier']].groupby('Position').head(3))
    else:
        print(df_tiered[['PLAYER', args.column, 'Tier']].head(10))

if __name__ == "__main__":
    main()
