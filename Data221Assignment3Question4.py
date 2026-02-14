import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Step 1: Load the dataset
kidney_disease_dataframe = pd.read_csv("kidney_disease.csv")

# In this file the label column is named "classification" (values like "ckd", "notckd", sometimes "ckd\t")
label_text = kidney_disease_dataframe["classification"].astype(str).str.strip().str.lower()
# For the line above, this basically coverts everything into strings, gets rid of whitespace, and makes it all lowercase

# Map to binary labels: 1 = kidney disease, 0 = healthy
label_vector = label_text.map({"ckd": 1, "notckd": 0})

# Step 2: Create feature matrix using numeric medical features only

# Drop the label column and ID column
raw_feature_matrix = kidney_disease_dataframe.drop(columns = ["classification", "id"])

# Convert to numeric. Non-numeric features become NaN
numeric_feature_matrix = raw_feature_matrix.apply(pd.to_numeric, errors = "coerce")

# Drop columns that become entirely NaN (these are non-numeric columns in the CSV)
numeric_feature_matrix = numeric_feature_matrix.dropna(axis = 1, how = "all")

# Fill missing values with the column mean
numeric_feature_matrix = numeric_feature_matrix.fillna(numeric_feature_matrix.mean())

# Step 3: Train/test split (70%/30%). Should match Question 3.
X_train, X_test, y_train, y_test = train_test_split(
    numeric_feature_matrix, label_vector, test_size = 0.30, random_state = 42
)

# Step 4: Scale features (important for KNN distances)
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

# Step 5: Train KNN (k = 5) and make predictions. Use X_train, y_train to train the model
knn_classifier = KNeighborsClassifier(n_neighbors = 5)

knn_classifier.fit(X_train_scaled, y_train)

predicted_labels = knn_classifier.predict(X_test_scaled) # Feed X_test into the model to make predictions

# Step 6: Display the confusion matrix and print the metrics. Use y_test to see the real values.
confusion_matrix_result = confusion_matrix(y_test, predicted_labels)

accuracy_value = accuracy_score(y_test, predicted_labels)

precision_score_value = precision_score(y_test, predicted_labels)

recall_value = recall_score(y_test, predicted_labels)

f1_value = f1_score(y_test, predicted_labels)

print("Confusion Matrix (rows = actual, columns = predicted):")
print(confusion_matrix_result)
print()

print(f"Accuracy:  {accuracy_value:.3f}")
print(f"Precision: {precision_score_value:.3f}")
print(f"Recall:    {recall_value:.3f}")
print(f"F1-score:  {f1_value:.3f}")

# The confusion matrix comes in the form:

# My confusion matrix essentially reads:
# Out of 44 actually healthy patients, the model got 44 right and made 0 false alarms .
# out of 76 patients with Chronic Kidney Disease,
# the model correctly caught 72 but missed 4 (falsely said they were healthy)

# Written Answers
# True Positive (TP): The model predicts CKD (1) and the patient actually has kidney disease.
# True Negative (TN): The model predicts healthy (0) and the patient is actually healthy.
# False Positive (FP): The model predicts CKD (1) but the patient is actually healthy (false alarm).
# False Negative (FN): The model predicts healthy (0) but the patient actually has kidney disease (missed case).

# Accuracy alone may not be enough because a model can look good overall while still making dangerous mistakes,
# especially false negatives (missing kidney disease cases). This can happen when the classes are imbalanced,
# meaning there are not equal numbers of healthy vs CKD patients. For example, if a dataset has 900 healthy and
# 100 CKD patients, a model that always predicts “healthy” would still achieve 90% accuracy while missing every CKD case.

# If missing a kidney disease case is very serious, Recall is the most important metric because it measures how many
# actual CKD cases the model successfully catches. High recall is prioritized when the cost of a false negative is much
# higher than the cost of a false positive.
