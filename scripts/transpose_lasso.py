import os
import pandas as pd
from pathlib import Path


def transpose_lasso_csvs(directory):
    """
    Transpose all CSV files ending with '_lasso.csv' in the given directory
    and save them with the same filename.

    Args:
        directory (str): Path to the directory containing the CSV files
    """
    # Convert to Path object for easier manipulation
    dir_path = Path(directory)

    # Check if directory exists
    if not dir_path.exists():
        print(f"Directory '{directory}' does not exist.")
        return

    # Find all files ending with '_lasso.csv'
    lasso_files = list(dir_path.glob("*_lasso.csv"))

    if not lasso_files:
        print(f"No files ending with '_lasso.csv' found in '{directory}'")
        return

    print(f"Found {len(lasso_files)} lasso CSV files to process:")

    for file_path in lasso_files:
        try:
            print(f"Processing: {file_path.name}")

            # Read the CSV file
            df = pd.read_csv(file_path, header=None)

            # Transpose the dataframe
            df_transposed = df.T

            # Save the transposed dataframe with the same filename
            df_transposed.to_csv(file_path, index=False, header=False)

            print(f"Successfully transposed and saved: {file_path.name}")

        except Exception as e:
            print(f"**********Error processing {file_path.name}: {str(e)}************")

    print("Processing complete!")


# Example usage
if __name__ == "__main__":
    # Replace with your directory path
    directory_path = "/scratch/gpfs/tk6637/princeton/247-plotting/data/encoding/podcast/tk-podcast-777-gemma-scope-2b-pt-res-canonical-lag2k-25-all/tk-200ms-777-lay13-con32-reglasso-sig_coeffs"  # Current directory

    # You can also specify a different directory:
    # directory_path = "/path/to/your/directory"

    transpose_lasso_csvs(directory_path)