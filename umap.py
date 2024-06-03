import pandas as pd
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import umap #pip install umap-learn
import matplotlib.pyplot as plt

# Define the range for random numbers
num_range = range(1, 10)  # Example range from 1 to 9

# Generate random rows
num_rows = 100  # specify the number of rows you want
row1 = [random.choice(num_range) for _ in range(num_rows)]
row2 = [random.choice(num_range) for _ in range(num_rows)]
row3 = [random.choice(num_range) for _ in range(num_rows)]
row4 = [random.choice(num_range) for _ in range(num_rows)]
row5 = [random.choice(num_range) for _ in range(num_rows)]

animal_list = ["['hest']", "['ku']", "['gris']", "['sau']"]
parsed_a = [ast.literal_eval(item)[0] for item in animal_list]
random_animals = [[random.choice(parsed_a)] for _ in range(num_rows)]
print(random_animals)

# Initialize the LabelEncoder
label_encoder = LabelEncoder()
# Create the DataFrame
df = pd.DataFrame()
df['farm_animals'] = pd.DataFrame(random_animals, columns=['farm_animals'])
print(df.farm_animals)
# Fit and transform the 'farm_animals' column
df['farm_animals_encoded'] = label_encoder.fit_transform(df['farm_animals'])


# Create the DataFrame
df1 = pd.DataFrame({
    'row1': row1,
    'row2': row2,
    'row3': row3,
    'row4': row4,
    'row5': row5
})


# Assume the target labels are the sum of the rows modulo 2 (just for demonstration)
df1['target'] = df['farm_animals_encoded']
print(df1)

# Separate features and target
y = df1['target']
X = df1.drop(columns=['target'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply UMAP for dimensionality reduction
umap_reducer = umap.UMAP(n_components=2, random_state=42)
X_train_umap = umap_reducer.fit_transform(X_train)
X_test_umap = umap_reducer.transform(X_test)


# Initialize the K-Nearest Neighbors classifier
knn_classifier = KNeighborsClassifier(n_neighbors=3)

# Initialize lists to store accuracy values
train_accuracies = []
test_accuracies = []

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    # Train the classifier
    knn_classifier.fit(X_train_umap, y_train)
    
    # Make predictions on the training set
    y_train_pred = knn_classifier.predict(X_train_umap)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_accuracies.append(train_accuracy)
    
    # Make predictions on the test set
    y_test_pred = knn_classifier.predict(X_test_umap)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_accuracies.append(test_accuracy)
    
    print(f'Epoch {epoch + 1}/{num_epochs}, Train Accuracy: {train_accuracy:.2f}, Test Accuracy: {test_accuracy:.2f}')

# Plotting the converging accuracy
plt.plot(range(1, num_epochs + 1), train_accuracies, label='Train Accuracy')
plt.plot(range(1, num_epochs + 1), test_accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Testing Accuracy')
plt.legend()
plt.show()
