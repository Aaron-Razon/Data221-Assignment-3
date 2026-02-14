import pandas as pd
from sklearn.model_selection import train_test_split

# Step 1: Load kidney_disease.csv into pandas DataFrame
kidney_disease_dataframe = pd.read_csv("kidney_disease.csv")

# Step 2: Create feature matrix X (for all columns except CKD) by using .drop
feature_matrix = kidney_disease_dataframe.drop(columns = ["classification"])

# Step 3: Create the label vector y using the CKD column
label_vector = kidney_disease_dataframe["classification"]

# Step 4: Split into training (70%) and testing (30%) using a fixed random state (can be any number)
X_train, X_test, y_train, y_test = train_test_split(
    feature_matrix, label_vector, test_size = 0.30, random_state = 42
)

# Step 5: Print sizes to make sure the split worked
print("Training set size:", len(X_train)) # Expected output: 70% of 400 = 280
print("Testing set size:", len(X_test)) # Expected output: 30% of 400 = 120

# Written Answers
# Why should we not train and test a model on the same data?
# If we test on the same data we trained on, we risk doing something known as overfitting.
# The model may perform very well just because it "memorized" the patterns in that data.
# This gives a unrealistic, overly optimistic performance result and
# does not accurately tell us how well the model will work on new data.

# What is the purpose the testing set?
# The testing set provides for an unbiased evaluation of a model's performance.
# We only use it after training to see how well the model can adapt to new data and generalize,
# rather than just memorizing the target set. This yields a more accurate performance estimate.