import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load crime.csv into a pandas DataFrame
crime_dataframe = pd.read_csv("crime.csv")

# Step 2: Focus on the ViolentCrimesPerPop column.
violent_crime_rate_per_population_statistics = crime_dataframe["ViolentCrimesPerPop"]

# Step 3: Compute the mean and median using numpy
mean_value = np.mean(violent_crime_rate_per_population_statistics)
median_value = np.median(violent_crime_rate_per_population_statistics)

# Step 4: Compute standard deviation, and then get the maximum and minimum values
standard_deviation = np.std(violent_crime_rate_per_population_statistics)
minimum_value = min(violent_crime_rate_per_population_statistics)
maximum_value = max(violent_crime_rate_per_population_statistics)

# Step 5: Print the results
print("Violent Crimes Per Population Statistics")
print(f"Mean: {mean_value:.3f}")
print(f"Median: {median_value:.3f}")
print(f"Standard Deviation: {standard_deviation:.3f}")
print(f"Minimum: {minimum_value:.3f}")
print(f"Maximum: {maximum_value:.3f}")

# Written Answers
# Compare the MEAN and MEDIAN:
# Does the distribution look symmetric or skewed? Explain briefly.
# In this dataset, the MEAN (~0.441) is higher than the MEDIAN (~0.390).
# That usually implies the distribution is right-skewed (AKA positively-skewed)
# When MEAN > MEDIAN this usually means that the distribution is driven by outliers.
# While most values remain low, a small number of high scores are pulling the MEAN upward.

# Extreme Values: Which statistic is more affected: MEAN or MEDIAN?
# The MEAN is affected more by extreme values because it takes into account every data point
# The MEDIAN is less affected because it only depends on the middle value(s),
# so a few very large or very small values do not change it much