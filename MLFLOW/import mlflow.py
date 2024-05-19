import mlflow
import mlflow.keras
from hyperopt import hp, tpe, fmin, Trials
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout  
from tensorflow.keras.optimizers import Adam  
from sklearn.metrics import accuracy_score
import pandas as pd
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK ,space_eval
# Set MLflow experiment name
experiment_name = "Fashion MNIST Experiment"
mlflow.set_experiment(experiment_name)

# Set MLflow tracking URI (optional)
tracking_uri = "http://0.0.0.0:5050"
mlflow.set_tracking_uri(tracking_uri)

# Enable autologging
mlflow.keras.autolog()

# Define search space for hyperparameters
space = {
    'hidden_units': hp.choice('hidden_units', [64, 128, 256]),
    'dropout': hp.uniform('dropout', 0.0, 0.5),
    'learning_rate': hp.uniform('learning_rate', 0.0001, 0.01)
}

# Define objective function
# Define objective function
def objective(params, train_df, test_df):
    # Create model
    model = Sequential([
        Dense(params['hidden_units'], activation='relu', input_shape=(784,)),
        Dropout(params['dropout']),
        Dense(10, activation='softmax')
    ])
    
    # Compile model
    model.compile(optimizer=Adam(learning_rate=params['learning_rate']), 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    
    # Train model
    history = model.fit(train_df.iloc[:, :-1], train_df['label'], batch_size=64, epochs=10, validation_split=0.2, verbose=0)
    
    # Log metrics for this run
    with mlflow.start_run(nested=True):  # Start a nested run for each hyperopt run
        mlflow.log_params(params)
        train_loss = history.history['loss'][-1]
        train_accuracy = history.history['accuracy'][-1]
        val_loss = history.history['val_loss'][-1]
        val_accuracy = history.history['val_accuracy'][-1]
        mlflow.log_metrics({'train_loss': train_loss, 'train_accuracy': train_accuracy,
                            'val_loss': val_loss, 'val_accuracy': val_accuracy})
    
    return {'loss': -val_accuracy, 'status': STATUS_OK}

# Run hyperparameter optimization


# Load train and test datasets
train_df = pd.read_csv("/Users/saraths17/sem8/BDL PROJECT/apache pipeline/processed data/train_preprocessed_data.csv")
test_df = pd.read_csv("/Users/saraths17/sem8/BDL PROJECT/apache pipeline/processed data/test_preprocessed_data.csv")

# Run hyperparameter optimization
trials = Trials()
best = fmin(lambda params: objective(params, train_df, test_df), space, algo=tpe.suggest, max_evals=10, trials=trials)

# Get best parameters
best_params = space_eval(space, best)
print("Best Parameters:", best_params)

# Log best parameters
with mlflow.start_run():
    mlflow.log_params(best_params)
