"""Perceptron model for Assignment 1: Starter code.
You can change this code while keeping the function giving headers.
You can add any functions that will help you. The given function headers are used for testing the code, so changing them will fail testing.
"""

import argparse
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Set

from features import make_featurize
from tqdm import tqdm
from utils import DataPoint, DataType, accuracy, load_data, save_results

# For debugging purposes
import pdb

@dataclass(frozen=True)
class DataPointWithFeatures(DataPoint):
    features: Dict[str, float]


def featurize_data(data: List[DataPoint], feature_types: Set[str]) -> List[DataPointWithFeatures]:
    """Add features to each datapoint based on feature types.
    
    This function uses the provided `make_featurize` to create a featurization
    function (a combination of feature extractors) and then applies it to the text
    of every datapoint.
    """
    featurize_fn = make_featurize(feature_types)
    data_with_features = []
    for dp in data:
        features = featurize_fn(dp.text)
        # Always-on bias term
        features["BIAS"] = 1.0

        data_with_features.append(
            DataPointWithFeatures(id=dp.id, text=dp.text, label=dp.label, features=features)
        )
    # Uncomment below to easily access the (original) text and corresponding features in the debugger    
    #pdb.set_trace()    
    return data_with_features


class PerceptronModel:
    """Perceptron model for classification."""

    def __init__(self):
        # Using defaultdict(float) ensures missing keys default to 0.0.
        self.weights: Dict[str, float] = defaultdict(float)
        # This set will contain the labels encountered during training.
        self.labels: Set[str] = set()

    def _get_weight_key(self, feature: str, label: str) -> str:
        """Internal function to create a unique key for a (feature, label) pair.
        
        This key is used in self.weights for parameter storage.
        """
        return feature + "#" + str(label)

    def score(self, datapoint: DataPointWithFeatures, label: str) -> float:
        """Compute the score of a class given the input.

        Inputs:
            datapoint (Datapoint): a single datapoint with features populated
            label (str): label

        Returns:
            The output score.
        """

        # Here we initialize w and perform the dot product in an efficient manner through levaraging dictionaries
        score_val = 0.0
        for feature, value in datapoint.features.items():
            score_val += self.weights.get(self._get_weight_key(feature, label), 0.0) * value # Note: used .get(key_combination, 0) to solve unittest test_score error
        return score_val


    def predict(self, datapoint: DataPointWithFeatures) -> str:
        """Predicts a label for an input.

        Inputs:
            datapoint: Input data point.

        Returns:
            The predicted class.
        """
        # !Trick: Lamba function instead of for loop here is more clean and (memory) efficient! 
        return max(self.labels, key=lambda label: self.score(datapoint, label))

    def update_parameters(
        self, datapoint: DataPointWithFeatures, prediction: str, lr: float
    ) -> None:
        """Update the model weights of the model using the perceptron update rule.

        Inputs:
            datapoint: The input example, including its label.
            prediction: The predicted label.
            lr: Learning rate.
        """
        if prediction != datapoint.label:
            correct_label = datapoint.label
            for feature, value in datapoint.features.items():
                if value != 0: # To only update relevant features
                    # Increase weight for correct label
                    self.weights[self._get_weight_key(feature, correct_label)] += lr * value
                    # Decrease weight for incorrect prediction (to leverage the both ways updating as discussed in lecture)
                    self.weights[self._get_weight_key(feature, prediction)] -= lr * value



    def train(
        self,
        training_data: List[DataPointWithFeatures],
        val_data: List[DataPointWithFeatures],
        num_epochs: int,
        lr: float,
    ) -> None:
        """Perceptron model training. Updates self.weights and self.labels
        We greedily learn about new labels.

        Inputs:
            training_data: Suggested type is (list of tuple), where each item can be
                a training example represented as an (input, label) pair or (input, id, label) tuple.
            val_data: Validation data.
            num_epochs: Number of training epochs.
            lr: Learning rate.
        """
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            for datapoint in tqdm(training_data, desc=f"Training Epoch {epoch+1}"):
                # Greedily add new labels as they are encountered through 'if' statement 
                if datapoint.label not in self.labels:
                    self.labels.add(datapoint.label)
                prediction = self.predict(datapoint)
                if prediction != datapoint.label:
                    self.update_parameters(datapoint, prediction, lr)
            # Evaluate on validation data after each epoch.
            if val_data:
                val_acc = self.evaluate(val_data, save_path=None)
                print(f"Validation accuracy after epoch {epoch+1}: {100 * val_acc:.2f}%")
            

    def save_weights(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(json.dumps(self.weights, indent=2, sort_keys=True))
        print(f"Model weights saved to {path}")

    def evaluate(
        self,
        data: List[DataPointWithFeatures],
        save_path: str = None,
    ) -> float:
        """Evaluate the model on the provided dataset.
        
        The function predicts labels for every datapoint, calculates accuracy if
        true labels are available, and optionally saves the predictions.
        """
        predictions = []
        for datapoint in data:
            pred = self.predict(datapoint)
            predictions.append(pred)
        # If all datapoints have a true label, compute accuracy.
        if all(datapoint.label is not None for datapoint in data):
            true_labels = [datapoint.label for datapoint in data]
            acc = accuracy(predictions, true_labels)
        else:
            acc = 0.0  # For unlabeled test data, accuracy is not defined.
        if save_path is not None:
            save_results(data, predictions, save_path)
        return acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perceptron model")
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        default="sst2",
        help="Data source, one of ('sst2', 'newsgroups')",
    )
    parser.add_argument(
        "-f",
        "--features",
        type=str,
        default="bow",
        help="Feature type, e.g., bow+len",
    )
    parser.add_argument(
        "-e", "--epochs", type=int, default=3, help="Number of epochs"
    )
    parser.add_argument(
        "-l", "--learning_rate", type=float, default=0.1, help="Learning rate"
    )
    args = parser.parse_args()

    data_type = DataType(args.data)
    feature_types: Set[str] = set(args.features.split("+"))
    num_epochs: int = args.epochs
    lr: float = args.learning_rate

    # Load and featurize the data.
    train_data, val_data, dev_data, test_data = load_data(data_type)
    train_data = featurize_data(train_data, feature_types)
    val_data = featurize_data(val_data, feature_types)
    dev_data = featurize_data(dev_data, feature_types)
    test_data = featurize_data(test_data, feature_types)

    model = PerceptronModel()
    print("Training the model...")
    model.train(train_data, val_data, num_epochs, lr)

    # Evaluate on the development set and save predictions.
    dev_acc = model.evaluate(
        dev_data,
        save_path=None
    )
    print(f"Development accuracy: {100 * dev_acc:.2f}%")

    # Evaluate on the test set and save predictions.
    _ = model.evaluate(
        test_data,
        save_path=os.path.join(
            "results",
            f"perceptron_{args.data}_test_predictions.csv",
        ),
    )

    # Save the final model weights.
    model.save_weights(
        os.path.join(
            "results", f"perceptron_{args.data}_{args.features}_model.json"
        )
    )
