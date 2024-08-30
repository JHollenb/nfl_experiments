import pandas as pd
import argparse

def clean_player_names(player):
    # Split the player string to separate the name and the team
    name_parts = player.split(' ')
    team = name_parts[-1].strip('()')  # Extract the team from the last part
    player_name = ' '.join(name_parts[:-1])  # Join the rest as the player name
    
    # Replace special characters and make lowercase
    cleaned_name = player_name.replace(',', ' ').replace('.', ' ').replace('-', ' ').lower()
    
    return cleaned_name, team

def process_csv(input_file, output_file):
    # Load the CSV file
    df = pd.read_csv(input_file)

    # Apply the cleaning function to the 'PLAYER' column
    df[['cleaned_names', 'TEAM']] = df['PLAYER'].apply(lambda x: pd.Series(clean_player_names(x)))

    # Save the cleaned DataFrame to the specified output file
    df.to_csv(output_file, index=False)
    print(f"Cleaned data saved to {output_file}")

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Clean player names in a fantasy football CSV file.")
    parser.add_argument('input', help="Input CSV file path")
    parser.add_argument('output', help="Output CSV file path")

    # Parse the arguments
    args = parser.parse_args()

    # Process the CSV with the provided input and output file paths
    process_csv(args.input, args.output)

if __name__ == "__main__":
    main()

