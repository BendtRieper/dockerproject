import os
import argparse
import flwr as fl
import tensorflow as tf
import logging
from helpers.load_data import load_data
from model.model import Model

logging.basicConfig(level=logging.INFO)  # Configure logging
logger = logging.getLogger(__name__)  # Create logger for the module

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Parse command line arguments
parser = argparse.ArgumentParser(description="Flower client")

parser.add_argument(
    "--server_address", type=str, default="server:8080", help="Address of the server"
)
parser.add_argument(
    "--batch_size", type=int, default=32, help="Batch size for training"
)
parser.add_argument(
    "--learning_rate", type=float, default=0.1, help="Learning rate for the optimizer"
)
parser.add_argument("--client_id", type=int, default=1, help="Unique ID for the client")
parser.add_argument(
    "--total_clients", type=int, default=2, help="Total number of clients"
)
parser.add_argument(
    "--data_percentage", type=float, default=0.5, help="Portion of client data to use"
)

args = parser.parse_args()

# Create an instance of the model and pass the learning rate as an argument
model = Model(learning_rate=args.learning_rate)


class Client(fl.client.NumPyClient):
    def __init__(self, args):
        self.args = args

        # Load and prepare data
        logger.info("Preparing data...")
        (x_train, y_train), (x_val, y_val), (x_test, y_test), input_shape, output_shape = load_data(
            csv_path="/app/combinedTraffic_clean.csv",
            data_sampling_percentage=self.args.data_percentage,
            client_id=self.args.client_id,
            total_clients=self.args.total_clients,
        )

        # Initialize the model
        self.model = Model(learning_rate=self.args.learning_rate)
        self.model.compile(input_shape=input_shape, output_shape=output_shape)

        # Store datasets as attributes
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test

    def get_parameters(self, config):
        return self.model.get_model().get_weights()

    def fit(self, parameters, config):
        self.model.get_model().set_weights(parameters)
        self.model.get_model().fit(self.x_train, self.y_train, epochs=1, verbose=0)
        parameters_prime = self.model.get_model().get_weights()

        # Calculate validation accuracy after training
        val_loss, val_accuracy = self.model.get_model().evaluate(self.x_val, self.y_val, verbose=0)

        return parameters_prime, len(self.x_train), {"accuracy": float(val_accuracy)}

    def evaluate(self, parameters, config):
        self.model.get_model().set_weights(parameters)
        test_loss, test_accuracy = self.model.get_model().evaluate(self.x_test, self.y_test, verbose=0)
        return float(test_loss), len(self.x_test), {"accuracy": float(test_accuracy)}


# Function to Start the Client
def start_fl_client():
    try:
        client = Client(args)
        fl.client.start_numpy_client(server_address=args.server_address, client=client)
    except Exception as e:
        logger.error("Error starting FL client: %s", e)
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    # Call the function to start the client
    start_fl_client()
