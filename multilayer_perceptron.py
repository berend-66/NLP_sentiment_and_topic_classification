"""Multi-layer perceptron model for Assignment 1: Starter code.

You can change this code while keeping the function giving headers.
You can add any functions that will help you. The given function headers are used for testing the code, so changing them will fail testing.

We adapt shape suffixes style when working with tensors.
See https://medium.com/@NoamShazeer/shape-suffixes-good-coding-style-f836e72e24fd.

Dimension key:
    b: batch size
    l: max sequence length
    c: number of classes
    v: vocabulary size

For example,
    feature_b_l means a tensor of shape (b, l) == (batch_size, max_sequence_length).
    length_1 means a tensor of shape (1) == (1,).
    loss means a tensor of shape (). You can retrieve the loss value with loss.item().
"""


# !!!Note: something going wrong when using the model for the sst2 dataset, something you still need to look into


import argparse
import os
from collections import Counter
from pprint import pprint
from typing import Dict, List, Tuple

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from utils import DataPoint, DataType, accuracy, load_data, save_results


class Tokenizer:
    # The index of the padding embedding.
    # This is used to pad variable length sequences.
    TOK_PADDING_INDEX = 0
    STOP_WORDS = set(pd.read_csv("stopwords.txt", header=None)[0])

    def _pre_process_text(self, text: str) -> List[str]:
        """
        Pre-process the input text:
          - Lowercase the text.
          - Split on whitespace.
          - Strip common punctuation.
          - Remove stop words.
        """
        tokens = text.lower().split()
        tokens = [token.strip(".,!?;:()[]\"'") for token in tokens]
        tokens = [token for token in tokens if token not in self.STOP_WORDS and token != ""]
        # add more here
        return tokens

    def __init__(self, data: List[DataPoint], max_vocab_size: int = None):
        # Create a corpus from all texts
        corpus = " ".join([d.text for d in data])
        token_freq = Counter(self._pre_process_text(corpus))
        # Keep only the most common tokens if max_vocab_size is set
        token_freq = token_freq.most_common(max_vocab_size)
        tokens = [t for t, _ in token_freq]
        # Offset indices by 1 because index 0 is reserved for padding
        self.token2id = {t: (i + 1) for i, t in enumerate(tokens)}
        self.token2id["<PAD>"] = Tokenizer.TOK_PADDING_INDEX
        self.id2token = {i: t for t, i in self.token2id.items()}

    def tokenize(self, text: str) -> List[int]:
        """
        Convert text into a list of token IDs
        Only tokens present in the vocabulary are mapped
        """
        tokens = self._pre_process_text(text)
        return [self.token2id[token] for token in tokens if token in self.token2id]


def get_label_mappings(data: List[DataPoint]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Reads the labels in data and returns the mapping dictionaries."""
    labels = list(set([d.label for d in data if d.label is not None]))
    label2id = {label: index for index, label in enumerate(labels)}
    id2label = {index: label for index, label in enumerate(labels)}
    return label2id, id2label


class BOWDataset(Dataset):
    def __init__(
        self,
        data: List[DataPoint],
        tokenizer: Tokenizer,
        label2id: Dict[str, int],
        max_length: int = 100,
    ):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns a single example as a tuple of torch.Tensors.
        features_l: The tokenized text padded/truncated to (max_length,)
        length: The length of the text (before padding), shape ()
        label: The label of the example, shape ()
        All tensors are of type torch.int64.
        """
        dp: DataPoint = self.data[idx]
        token_ids = self.tokenizer.tokenize(dp.text)
        length = min(len(token_ids), self.max_length)
        # Pad the token sequence with the padding index
        if len(token_ids) < self.max_length:
            padded = token_ids + [self.tokenizer.token2id["<PAD>"]] * (self.max_length - len(token_ids))
        else:
            padded = token_ids[:self.max_length]
        # For training/validation data, dp.label is not None
        if dp.label is not None:
            label = self.label2id[dp.label]
        else:
            label = -1  # Use -1 or any default for unlabeled test data
        return (
            torch.tensor(padded, dtype=torch.int64),
            torch.tensor(length, dtype=torch.int64),
            torch.tensor(label, dtype=torch.int64),
        )



class MultilayerPerceptronModel(nn.Module):
    """Multi-layer perceptron model for classification."""

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        padding_index: int,
        hidden_dims: list = [100, 50],  # List of sizes for hidden layers with this a default 
        activation: str = "relu"     # Activation function: "relu", "sigmoid", or "tanh", can add more later if needed
    ):
        """
        Initializes the model.
        
        Args:
            vocab_size (int): Size of the vocabulary.
            num_classes (int): Number of classes.
            padding_index (int): Index used for padding.
            hidden_dims (list): A list of integers specifying hidden layer sizes.
            activation (str): Activation function to use ("relu", "sigmoid", "tanh").
        """
        super().__init__()
        self.padding_index = padding_index
        embedding_dim = 50  # Embedding dimension can be adjusted as needed
        
        # Embedding layer: maps token IDs to embeddings
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_index)
        
        # Choose the activation function based on the input (can add more later)
        activation = activation.lower()
        if activation == "relu":
            act_fn = nn.ReLU()
        elif activation == "sigmoid":
            act_fn = nn.Sigmoid()
        elif activation == "tanh":
            act_fn = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation function: {activation}")
        
        # Build the MLP layers
        layers = []
        input_dim = embedding_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(act_fn)
            input_dim = hidden_dim
        # Final layer: outputs !logits! for each class --> might need to edit
        layers.append(nn.Linear(input_dim, num_classes))
        
        # Wrap layers in a Sequential container
        self.mlp = nn.Sequential(*layers)

    def forward(
        self, input_features_b_l: torch.Tensor, input_length_b: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            input_features_b_l (torch.Tensor): Tensor of shape (batch_size, max_length)
                containing token IDs.
            input_length_b (torch.Tensor): Tensor of shape (batch_size,) containing the true lengths.
        
        Returns:
            torch.Tensor: Logits of shape (batch_size, num_classes).
        """
        # Get embeddings: shape (batch_size, max_length, embedding_dim)
        embedded = self.embedding(input_features_b_l)
        # Create a mask for non-padding tokens.
        mask = (input_features_b_l != self.padding_index).unsqueeze(-1).float()
        # Sum embeddings along the sequence length.
        summed = torch.sum(embedded * mask, dim=1)
        # Average the embeddings by the true length.
        avg_embedded = summed / input_length_b.unsqueeze(1).float()
        # Pass the averaged embedding through the MLP.
        logits = self.mlp(avg_embedded)
        return logits



class Trainer:
    def __init__(self, model: nn.Module):
        self.model = model

    def predict(self, data: BOWDataset) -> List[int]:
        """
        Predicts labels for all examples in the dataset.
        Returns:
            List[int]: The predicted label IDs.
        """
        self.model.eval()
        all_predictions = []
        dataloader = DataLoader(data, batch_size=32, shuffle=False)
        with torch.no_grad():
            for inputs_b_l, lengths_b, labels_b in dataloader:
                outputs = self.model(inputs_b_l, lengths_b)
                preds = torch.argmax(outputs, dim=1)
                all_predictions.extend(preds.cpu().tolist())
        return all_predictions

    def evaluate(self, data: BOWDataset) -> float:
        """
        Evaluates the model on a dataset.
        Returns:
            float: Accuracy of the model.
        """
        self.model.eval()
        all_predictions = []
        all_labels = []
        dataloader = DataLoader(data, batch_size=32, shuffle=False) # Larger batch sizes (32) in both the eval and predict function as there are no gradients being computed so we can handle larger sizes
        with torch.no_grad():
            for inputs_b_l, lengths_b, labels_b in dataloader:
                outputs = self.model(inputs_b_l, lengths_b)
                preds = torch.argmax(outputs, dim=1)
                all_predictions.extend(preds.cpu().tolist())
                all_labels.extend(labels_b.cpu().tolist())
        correct = sum(1 for p, t in zip(all_predictions, all_labels) if p == t)
        return correct / len(all_labels)

    def train(
        self,
        training_data: BOWDataset,
        val_data: BOWDataset,
        optimizer: torch.optim.Optimizer,
        num_epochs: int,
    ) -> None:
        """
        Trains the MLP.
        Args:
            training_data (BOWDataset): Training dataset.
            val_data (BOWDataset): Validation dataset.
            optimizer (torch.optim.Optimizer): Optimization method.
            num_epochs (int): Number of training epochs.
        """
        torch.manual_seed(0)
        loss_fn = nn.CrossEntropyLoss()
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0.0
            dataloader = DataLoader(training_data, batch_size=4, shuffle=True) # batch_size currently set to 4, experiment with later
            for inputs_b_l, lengths_b, labels_b in tqdm(dataloader):
                optimizer.zero_grad()
                outputs = self.model(inputs_b_l, lengths_b)
                loss = loss_fn(outputs, labels_b)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * inputs_b_l.size(0)
            per_dp_loss = total_loss / len(training_data)
            self.model.eval()
            val_acc = self.evaluate(val_data)
            print(
                f"Epoch: {epoch + 1:<2} | Loss: {per_dp_loss:.2f} | Val accuracy: {100 * val_acc:.2f}%"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MultiLayerPerceptron model")
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        default="sst2",
        help="Data source, one of ('sst2', 'newsgroups')",
    )
    parser.add_argument("-e", "--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument(
        "-l", "--learning_rate", type=float, default=0.001, help="Learning rate"
    )
    # New arguments for activation, hidden layer dimensions, and optimizer 
    parser.add_argument(
        "-a",
        "--activation",
        type=str,
        default="relu",
        help="Activation function to use (relu, sigmoid, tanh)", # For now these are the only three activation functions implemented, but can implement more later if needed/wanted
    )
    parser.add_argument(
        "-hd",
        "--hidden_dims",
        type=str,
        default="100, 50",
        help="Comma-separated list of hidden layer sizes (e.g., '100,50')",
    )
    parser.add_argument(
        '-o',
        "--optimizer",
        type=str,
        default="adam",
        help="Optimizer to use (adam, sgd, adagrad)",
    )
    args = parser.parse_args()

    # Parse the hidden_dims string into a list of integers
    hidden_dims = [int(x) for x in args.hidden_dims.split(",") if x.strip()]

    num_epochs = args.epochs
    lr = args.learning_rate
    data_type = DataType(args.data)

    train_data, val_data, dev_data, test_data = load_data(data_type)

    tokenizer = Tokenizer(train_data, max_vocab_size=20000)
    label2id, id2label = get_label_mappings(train_data)
    print("Id to label mapping:")
    pprint(id2label)

    max_length = 100
    train_ds = BOWDataset(train_data, tokenizer, label2id, max_length)
    val_ds = BOWDataset(val_data, tokenizer, label2id, max_length)
    dev_ds = BOWDataset(dev_data, tokenizer, label2id, max_length)
    test_ds = BOWDataset(test_data, tokenizer, label2id, max_length)

    model = MultilayerPerceptronModel(
        vocab_size=len(tokenizer.token2id),
        num_classes=len(label2id),
        padding_index=Tokenizer.TOK_PADDING_INDEX,
        hidden_dims=hidden_dims,   
        activation=args.activation        
    )

    # Different optimizer choices, instantiated (might be interesting to experiment with leaky relu and ELU later)
    optimizer_name = args.optimizer.lower()
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    elif optimizer_name == "adagrad":
        optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")

    trainer = Trainer(model)

    print("Training the model...")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    trainer.train(train_ds, val_ds, optimizer, num_epochs)

    # Evaluate on dev
    dev_acc = trainer.evaluate(dev_ds)
    print(f"Development accuracy: {100 * dev_acc:.2f}%")

    # Predict on test and save predictions
    test_preds = trainer.predict(test_ds)
    test_preds = [id2label[pred] for pred in test_preds]

    # Set to true if you want to save results
    if False:
        save_results(
            test_data,
            test_preds,
            os.path.join("results", f"mlp_{args.data}_test_predictions.csv"),
        )
