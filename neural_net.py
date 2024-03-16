df = pd.read_excel('XXXX.xlsx')
el_df_clean = df.drop(['XXX', 'YYY'], axis = 1)
el_df_clean.head()

##################################################################################################################
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

# Load the data
data = el_df_clean

# Preprocessing
# Drop rows with missing values
data.dropna(inplace=True)

# Encode labels 
label_encoder = LabelEncoder()
label = data['name']
print(label)
data['name'] = label_encoder.fit_transform(data['name'])

# Split features and labels
X = data.drop(['name', 'Y'], axis=1)  
y = data['name']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the neural network architecture
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(len(label_encoder.classes_), activation='softmax'))

# Compile the model
optimizer = Adam(lr=0.001)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train_scaled, y_train, epochs=30, batch_size=32, validation_split=0.2, verbose=1)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')

# Predict on the test data
y_pred_prob = model.predict(X_test_scaled)
y_pred = label.iloc[y_pred_prob.argmax(axis=1)].values




def summarize_diagnostics(history):
    '''Displaying results as plots'''
    plt.plot(history.history['loss'], color='blue', label='train_loss')
    plt.plot(history.history['val_loss'], color='orange', label='val_loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.plot(history.history['accuracy'], color='blue', label='train_accuracy')
    plt.plot(history.history['val_accuracy'], color='orange', label='val_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

# Plot examples of the output
summarize_diagnostics(history)


#############################################################################################################

label = dict(zip(range(len(label_encoder.classes_)), label_encoder.classes_))

# Predict on the test data
y_pred_prob = model.predict(X_test_scaled)
y_pred = label_encoder.inverse_transform(y_pred_prob.argmax(axis=1))  # Inverse transform predicted labels


for i in range(len(y_pred)):
    actual_name = label[y_test.iloc[i]]
    print(f'Example {i+1}: Actual: {actual_name}, Predicted: {y_pred[i]}')
