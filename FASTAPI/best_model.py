import mlflow
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import pandas as pd

# Load preprocessed data
train_df = pd.read_csv("/Users/saraths17/sem8/BDL PROJECT/apache pipeline/processed data/train_preprocessed_data.csv")
test_df = pd.read_csv("/Users/saraths17/sem8/BDL PROJECT/apache pipeline/processed data/test_preprocessed_data.csv")

# Separate features and labels
x_train = train_df.drop(columns=['label']).values
y_train = train_df['label'].values
x_test = test_df.drop(columns=['label']).values
y_test = test_df['label'].values

# Define best parameters
best_params = {'hidden_units': 256, 'dropout': 0.1, 'learning_rate': 0.002}

# Define and compile the model with the best parameters
best_model = Sequential([
    Dense(best_params['hidden_units'], activation='relu', input_shape=(x_train.shape[1],)),
    Dropout(best_params['dropout']),
    Dense(10, activation='softmax')
])
best_model.compile(optimizer=Adam(learning_rate=best_params['learning_rate']), 
                   loss='sparse_categorical_crossentropy', 
                   metrics=['accuracy'])

# Train the model
history = best_model.fit(x_train, y_train, batch_size=64, epochs=10, validation_split=0.2)

# Evaluate the model on test data
test_loss, test_accuracy = best_model.evaluate(x_test, y_test)

# Save the trained model
save_model(best_model, "Fashionmnist_best_model.h5")
