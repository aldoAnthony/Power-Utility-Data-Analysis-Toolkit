import numpy as np
import pandas as pd

# Electricification by province (EBP) data
ebp_path = 'python\data\electrification_by_province.csv'
ebp_df = pd.read_csv(ebp_path)


# Preprocessing: Remove commas and convert to integer
for col, row in ebp_df.iloc[:,1:].items():
    ebp_df[col] = ebp_df[col].str.replace(',','').astype(int)


""" ### Important Variables ### """
# gauteng ebp data as a list
gauteng = ebp_df['Gauteng'].astype(float).to_list()

# Function to calculate the five-number summary of a dataset
def five_num_summary(data):
    """
    Calculate the five-number summary (minimum, maximum, median, 1st quartile, and 3rd quartile) of a dataset.

    Args:
    data (list or array-like): The dataset for which to calculate the summary.

    Returns:
    dict: A dictionary containing the five-number summary statistics.
    """
    # Calculate the maximum, median, minimum, 1st quartile, and 3rd quartile
    _max = np.max(data)
    median = np.median(data)
    _min = np.min(data)
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)

    # Create a dictionary with the summary statistics
    summary_stats = {
        "Maximum": _max,
        "Median": median,
        "Minimum": _min,
        "1st Quartile": q1,
        "3rd Quartile": q3
    }

    # Round all the values in the dictionary to 2 decimal places
    rounded_summary_stats = {key: round(value, 2) for key, value in summary_stats.items()}

    return rounded_summary_stats

# Input:
five_num_summary(gauteng)

# Output:
output_data = five_num_summary(gauteng)
print(output_data)
