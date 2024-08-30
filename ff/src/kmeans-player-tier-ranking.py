import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
#from scipy.stats import jenks_natural_breaks as jnb

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

def create_position_tiers(df, column_name, method='kmeans', **kwargs):
    """Create tiers for each position based on specified method and parameters."""
    df = extract_position(df)
    
    tier_methods = {
        'kmeans': kmeans_tiers,
        #'jenks': jenks_tiers,
        'quantile': quantile_tiers,
        'std_dev': std_dev_tiers,
        'equal_interval': equal_interval_tiers
    }
    
    if method not in tier_methods:
        raise ValueError("Invalid tiering method specified.")
    
    tier_func = tier_methods[method]
    
    # Group by position and apply tiering method
    df['PositionTier'] = df.groupby('Position')[column_name].transform(lambda x: tier_func(x, **kwargs))
    
    return df

def main():
    file_path = 'fantasypros_vbd_ppr_rankings.csv'
    df = load_csv_to_dataframe(file_path)
    
    # Create position-based tiers using different methods
    methods = ['kmeans', 'jenks', 'quantile', 'std_dev', 'equal_interval']
    
    for method in methods:
        df_tiered = create_position_tiers(df.copy(), 'VBD', method=method, num_tiers=8)
        
        # Sort the DataFrame by Position and VBD
        df_tiered = df_tiered.sort_values(['Position', 'VBD'], ascending=[True, False])
        
        # Save results
        output_file = f'rankings_{method}_position_tiers.csv'
        df_tiered.to_csv(output_file, index=False)
        print(f"Rankings with {method} position-based tiers saved to {output_file}")
        
        # Display a sample of the results
        print(f"\nSample of {method} position-based tiers:")
        print(df_tiered[['PLAYER', 'Position', 'VBD', 'PositionTier']].groupby('Position').head(3))
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    main()
