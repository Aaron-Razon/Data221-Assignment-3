import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Step 1: Load the dataset
kidney_disease_dataframe = pd.read_csv("kidney_disease.csv")

# Clean and convert classification column to binary ( 1 = CKD, 0 = not CKD)
clean_label_text = kidney_disease_dataframe["classification"].astype(str).str.strip().str.lower()
label_vector = clean_label_text.map({"ckd": 1, "notckd":0})

# Build feature matrix (drop label and ID, keep numeric values)
raw_feature_matrix = kidney_disease_dataframe.drop(columns = ["classification", "id"])
numeric_feature_matrix = raw_feature_matrix.apply(pd.to_numeric, errors = "coerce")
# Converts to numeric values. Otherwise, if invalid convert to NaN

# Drop all columns that are entirely missing (non-numeric columns that become all NaN)
numeric_feature_matrix = numeric_feature_matrix.dropna(axis = 1, how = "all")

# Fill missing values with column means
numeric_feature_matrix = numeric_feature_matrix.fillna(numeric_feature_matrix.mean())

# Step 2: Use the same train/test split as before
feature_matrix_train, feature_matrix_test, label_train, label_test = train_test_split(
    numeric_feature_matrix, label_vector, test_size = 0.30, random_state = 42
)
# Scale features
scaler = StandardScaler()
scaled_feature_matrix_train = scaler.fit_transform(feature_matrix_train)
scaled_feature_matrix_test = scaler.transform(feature_matrix_test)

# Step 3: Train KNN models for multiple values of K
k_values_to_test = [1, 3, 5, 7, 9]
knn_accuracy_results = []  # list of (k, accuracy)

for k_value in k_values_to_test:
    knn = KNeighborsClassifier(n_neighbors=k_value)

    # Train on SCALED training data
    knn.fit(scaled_feature_matrix_train, label_train)

    # Predict on SCALED test data
    predicted_test_labels = knn.predict(scaled_feature_matrix_test)

    test_accuracy = accuracy_score(label_test, predicted_test_labels)

    # Append a (k, accuracy) pair
    knn_accuracy_results.append((k_value, test_accuracy))

# Step 4: Print a small table of results
print("K-Value --> Test Accuracy")
for k_value, test_accuracy in knn_accuracy_results:
    print(f"{k_value:>5} --> {test_accuracy:.3f}")
    # The (f"{k_value:>5} is a cool alignment trick for f-strings that I got from:
    # https://www.geeksforgeeks.org/python/string-alignment-in-python-f-string/
    # It is just there to make my output look a little nicer, it doesn't really serve a functional purpose.

# Step 5: Identify which K gives the highest test accuracy
best_k_value, best_accuracy = knn_accuracy_results[0]
for k_value, accuracy in knn_accuracy_results:
    if accuracy > best_accuracy:
        best_k_value = k_value
        best_accuracy = accuracy

print(f"\nBest K-Value --> {best_k_value} (accuracy = {best_accuracy:.3f})")

# Written Answers
# Changing K changes how many neighbors vote on the prediction. Small K-Values focus on smaller, local patterns,
# while larger K-Values make predictions based on a wider set of values and tends to be smoother.

# Very small K-Values (ex. k = 1) can overfit because the model becomes too sensitive to noise/outliers.

# Very large K-Values (ex. k = 9) can underfit because the model becomes too simple and
# fails to learn underlying patterns in the data.

# A good K is usually a balance where the model generalizes well to new data,
# without being vulnerable to outliers or being oversimplistic.

