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


# Function to calculate summary statistics of a dataset
def dictionary_of_metrics(data):
    """
    Calculate summary statistics (mean, median, variance, standard deviation, minimum, maximum) of a dataset.

    Args:
    data (list or array-like): The dataset for which to calculate the summary statistics.

    Returns:
    dict: A dictionary containing the summary statistics.
    """
    # Calculate mean, median, variance, standard deviation, minimum, and maximum
    mean = np.mean(data)
    median = np.median(data)
    variance = np.var(data, ddof=1)  # Set ddof=1 for sample variance
    std_dev = np.std(data, ddof=1)   # Set ddof=1 for sample standard deviation
    minimum = np.min(data)
    maximum = np.max(data)

    # Create a dictionary with the summary statistics
    summary_stats = {
        "Mean": mean,
        "Median": median,
        "Variance": variance,
        "Standard Deviation": std_dev,
        "Minimum": minimum,
        "Maximum": maximum
    }

    # Round all the values in the dictionary to 2 decimal places
    rounded_summary_stats = {key: round(value, 2) for key, value in summary_stats.items()}

    return rounded_summary_stats

# Input:
dictionary_of_metrics(gauteng)

# Output:
output_data = dictionary_of_metrics(gauteng)
print(output_data)
