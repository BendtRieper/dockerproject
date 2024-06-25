import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(csv_path: str, data_sampling_percentage=0.5, client_id=1, total_clients=2):
    """Load and preprocess dataset from a CSV file.

    Args:
        csv_path (str): Path to the CSV file.
        data_sampling_percentage (float): Percentage of the dataset to use for training.
        client_id (int): Unique ID for the client.
        total_clients (int): Total number of clients.

    Returns:
        Tuple: (train_data, test_data, input_shape, output_shape)
    """
    # Load the data
    df = pd.read_csv(csv_path, encoding='latin1', low_memory=False)

    # Clean up column names and drop unnecessary columns
    df.columns = df.columns.str.strip()
    df = df.drop(columns=['Flow ID', 'Source IP', 'Source Port', 'Destination IP', 'Destination Port', 'Protocol', 'Timestamp'])

    # Clean up label column
    df['Label'] = df['Label'].str.strip()
    labels_to_remove_prefix = ['XSS', 'Brute Force', 'Sql Injection']
    for label in labels_to_remove_prefix:
        df['Label'] = df['Label'].str.replace(f'Web Attack \x96 {label}', label)
    df = df.dropna(subset=['Label'])
    df = df[df['Label'] != 'Label']

    # Encode labels
    unique_labels_before_encoding = df['Label'].unique()
    label_mapping = {label: idx for idx, label in enumerate(unique_labels_before_encoding)}
    df['Label'] = df['Label'].map(label_mapping)

    # Convert all columns to numeric and handle missing values
    numeric_columns = df.columns.drop('Label')
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

    # Replace infinite values with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Drop rows with NaN values
    df = df.dropna()

    # Normalize features
    scaler = StandardScaler()
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

    # Split the data
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    # Convert DataFrame to numpy arrays
    x_train, y_train = train_df.drop(columns=['Label']).values, train_df['Label'].values
    x_val, y_val = val_df.drop(columns=['Label']).values, val_df['Label'].values
    x_test, y_test = test_df.drop(columns=['Label']).values, test_df['Label'].values

    # Apply data sampling
    num_samples = int(data_sampling_percentage * len(x_train))
    indices = np.random.choice(len(x_train), num_samples, replace=False)
    x_train, y_train = x_train[indices], y_train[indices]

    # Split data among clients
    client_data_size = len(x_train) // total_clients
    client_start_idx = client_data_size * (client_id - 1)
    client_end_idx = client_start_idx + client_data_size

    x_train_client = x_train[client_start_idx:client_end_idx]
    y_train_client = y_train[client_start_idx:client_end_idx]

    input_shape = x_train_client.shape[1:]  # Shape of one data point
    output_shape = len(np.unique(y_train_client))  # Number of classes

    return (x_train_client, y_train_client), (x_val, y_val), (x_test, y_test), input_shape, output_shape
