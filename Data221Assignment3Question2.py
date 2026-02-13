import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load crime.csv into a pandas DataFrame
crime_dataframe = pd.read_csv("crime.csv")

# Step 2: Get the ViolentCrimesPerPop column (remove missing values just in case)
violent_crime_rate_per_population_statistics = crime_dataframe["ViolentCrimesPerPop"].dropna()


#  Plot 1: Histogram
plt.figure()
plt.hist(violent_crime_rate_per_population_statistics, bins=20)
plt.title("Distribution of Violent Crime Rate Across Communities")
plt.xlabel("Violent Crime Rate Per Population (ViolentCrimesPerPop)")
plt.ylabel("Number of Communities")
plt.show()

#  Plot 2: Box Plot
plt.figure()
plt.boxplot(violent_crime_rate_per_population_statistics, vert=False)
plt.title("Violent Crime Rate Across Communities (Median, Spread, Outliers)")
plt.xlabel("Violent Crime Rate Per Population (ViolentCrimesPerPop)")
plt.ylabel("Communities")
plt.show()

# Written Answers
# What does the histogram show about how the data points are spread?
# The histogram shows most communities clustered in the lower-to-middle violent crime rates,
# with the biggest concentration roughly around 0.2 to 0.3. As the values increase toward
# the high end, fewer communities appear, but there is also a clear spike at the high end.
# A substantial number of communities are at (or very near) 1.00,
# which explains why the MEAN (~0.44) is higher than,
# the MEDIAN (~0.39) and demonstrates a slight right-skew.

# What does the box plot show about the MEDIAN?
# The box plot shows the MEDIAN around 0.39, meaning about half of the communities have
# ViolentCrimesPerPop below ~0.39 and half are above it. The middle 50% of values (the box)
# runs from about 0.21 (25th percentile) to about 0.65 (75th percentile),
# showing a fairly wide spread in typical values.

# Does the box plot suggest the presence of outliers?
# The legs extend from about 0.02 up to 1.00, and there are no separate outlier points
# shown beyond the whiskers. This suggests there are no strong outliers.