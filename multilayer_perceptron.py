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
import re

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from utils import DataPoint, DataType, accuracy, load_data, save_results

import pdb 

class Tokenizer:
    # The index of the padding embedding.
    # This is used to pad variable length sequences.
    TOK_PADDING_INDEX = 0
    STOP_WORDS = set(pd.read_csv("stopwords.txt", header=None)[0])

    def _pre_process_text(self, text: str) -> List[str]:
        """
        Enhanced pre-processing of input text:
          - Lowercase the text
          - Split on whitespace
          - Strip punctuation except for ! and ?
          - Remove stop words
          - Remove numbers
          - Remove extra whitespace
          - Remove very short words
          - Handle contractions
          - Remove special characters
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^\w\s!?]', ' ', text) # Replace special chars with space while keeping ! and ?
        text = re.sub(r'\d+', '', text)       # Remove numbers
        
        # Handle common contractions
        text = text.replace("'s", "")
        text = text.replace("n't", " not")
        text = text.replace("'ve", " have")
        text = text.replace("'re", " are")
        text = text.replace("'m", " am")
        text = text.replace("'ll", " will")
        text = text.replace("'d", " would")
        
        # Split into tokens
        tokens = text.split()
        
        # Remove stop words and empty strings
        tokens = [token for token in tokens 
                 if token not in self.STOP_WORDS 
                 and token != ""
                 and len(token) > 2]  # Remove very short words
        
        # Remove extra whitespace
        tokens = [token.strip() for token in tokens]
        
        return tokens

    def __init__(self, data: List[DataPoint], max_vocab_size: int = None):
        # Create a corpus from all texts.
        corpus = " ".join([d.text for d in data])
        token_freq = Counter(self._pre_process_text(corpus))
        # Keep only the most common tokens if max_vocab_size is set.
        token_freq = token_freq.most_common(max_vocab_size)
        tokens = [t for t, _ in token_freq]
        # Reserve index 0 for <PAD> and index 1 for <UNK>.
        self.token2id = {t: (i + 2) for i, t in enumerate(tokens)}
        self.token2id["<PAD>"] = Tokenizer.TOK_PADDING_INDEX  # 0
        self.token2id["<UNK>"] = 1  # unknown token
        self.id2token = {i: t for t, i in self.token2id.items()}

    def tokenize(self, text: str) -> List[int]:
        """
        Convert text into a list of token IDs
        Only tokens present in the vocabulary are mapped
        """
        tokens = self._pre_process_text(text)
        # If token list is empty, return the <PAD> token.
        if not tokens:
            return [self.token2id["<PAD>"]] # if the token list is empty, return the <PAD> token !NOT the unknown token
        # For each token, if it's in the vocabulary use its ID, otherwise use <UNK>.
        return [self.token2id.get(token, self.token2id["<UNK>"]) for token in tokens]


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
        max_length: int = 100, # ie. default max length of 100 tokens
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

        # debugging to check if the tokenizer is working correctly (ie. if it is returning a list of token ids instead of an empty list)
        if not token_ids:
            pdb.set_trace()

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
        activation: str = "relu",     # Activation function: "relu", "sigmoid", or "tanh", can add more later if needed
        dropout_rate: float = 0.0 # Default dropout rate of 0.0 which means no dropout
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
            # Dropout layer
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(dropout_rate))
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
        patience: int = 3  # Number of epochs to wait for improvement
    ) -> None:
        """
        Trains the MLP model with early stopping based on validation accuracy.

        Parameters:
            training_data (BOWDataset): The training dataset.
            val_data (BOWDataset): The validation dataset.
            optimizer (torch.optim.Optimizer): The optimizer for training.
            num_epochs (int): Maximum number of epochs.
            patience (int): Number of epochs with no improvement after which training will be stopped.
        """
        torch.manual_seed(0)
        loss_fn = nn.CrossEntropyLoss()
        best_val_acc = 0.0
        epochs_without_improvement = 0
        best_model_state = None

        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0.0
            dataloader = DataLoader(training_data, batch_size=4, shuffle=True)
            for inputs_b_l, lengths_b, labels_b in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
                optimizer.zero_grad()
                outputs = self.model(inputs_b_l, lengths_b)
                loss = loss_fn(outputs, labels_b)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * inputs_b_l.size(0)
            per_dp_loss = total_loss / len(training_data)

            # Evaluate on training and validation sets
            train_acc = self.evaluate(training_data)
            val_acc = self.evaluate(val_data)

            print(f"Epoch: {epoch + 1:<2} | Loss: {per_dp_loss:.2f} | Train Acc: {100 * train_acc:.2f}% | Val Acc: {100 * val_acc:.2f}%")

            # Early stopping check
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_without_improvement = 0
                best_model_state = self.model.state_dict()
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print(f"Validation accuracy has not improved for {patience} epochs. Stopping early.")
                break

        # Restore the best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MultiLayerPerceptron model")
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        default="sst2",
        help="Data source, one of ('sst2', 'newsgroups')",
    )
    parser.add_argument(
        "-e", 
        "--epochs", 
        type=int, 
        default=3, 
        help="Number of epochs"
    )
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
        default="256, 128",
        help="Comma-separated list of hidden layer sizes (e.g., '100,50')",
    )
    parser.add_argument(
        '-o',
        "--optimizer",
        type=str,
        default="adam",
        help="Optimizer to use (adam, sgd, adagrad)",
    )
    # New arguments for vocabulary size and document max length.
    parser.add_argument(
        "--max_vocab_size",
        type=int,
        default=20000,
        help="Maximum vocabulary size for the tokenizer",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=100,
        help="Maximum length (in tokens) for each document",
    )
    parser.add_argument(
        "-dr",
        "--dropout_rate",
        type=float,
        default=0.5,
        help="Dropout rate to use in hidden layers (0.0 to disable dropout, e.g., 0.5 for 50% dropout)"
    )
    parser.add_argument(
        "--error_analysis",
        action="store_true",
        help="If set, print error examples from the dev set"
    )
    parser.add_argument(
        "--save_test_predictions", 
        action="store_true",
        help="If set, save test predictions"
    )
    
    args = parser.parse_args()

    # Parse the hidden_dims string into a list of integers
    hidden_dims = [int(x) for x in args.hidden_dims.split(",") if x.strip()]

    num_epochs = args.epochs
    lr = args.learning_rate
    data_type = DataType(args.data)

    train_data, val_data, dev_data, test_data = load_data(data_type)

    tokenizer = Tokenizer(train_data, max_vocab_size=args.max_vocab_size)
    label2id, id2label = get_label_mappings(train_data)
    print("Id to label mapping:")
    pprint(id2label)

    max_length = args.max_length
    train_ds = BOWDataset(train_data, tokenizer, label2id, max_length)
    val_ds = BOWDataset(val_data, tokenizer, label2id, max_length)
    dev_ds = BOWDataset(dev_data, tokenizer, label2id, max_length)
    test_ds = BOWDataset(test_data, tokenizer, label2id, max_length)

    model = MultilayerPerceptronModel(
        vocab_size=len(tokenizer.token2id),
        num_classes=len(label2id),
        padding_index=Tokenizer.TOK_PADDING_INDEX,
        hidden_dims=hidden_dims,   
        activation=args.activation,
        dropout_rate=args.dropout_rate
    )

    # Different optimizer choices, instantiated (might be interesting to experiment with leaky relu and ELU later)
    # Implemented L2 regularization for the weights of the model with the weight_decay parameter to avoid overfitting 
    optimizer_name = args.optimizer.lower()
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.001)
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=0.001)
    elif optimizer_name == "adagrad":
        optimizer = torch.optim.Adagrad(model.parameters(), lr=lr, weight_decay=0.001)
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

    # Error Analysis Block
    if args.error_analysis:
        print("\nQualitative Error Analysis:")
        errors = []
        model.eval()  # Set model to evaluation mode
        
        # Define label mappings for different datasets
        label_maps = {
            'sst2': {
                0: "negative",
                1: "positive"
            },
            'newsgroups': {
                0: "atheism",
                1: "computer graphics",
                2: "computer os microsoft windows misc",
                3: "computer systems ibm pc hardware",
                4: "computer windows x",
                5: "misc forsale",
                6: "rec autos",
                7: "rec motorcycles",
                8: "rec sport baseball",
                9: "rec sport hockey",
                10: "sci crypt",
                11: "sci electronics",
                12: "sci med",
                13: "sci space",
                14: "soc religion christian",
                15: "talk politics guns",
                16: "talk politics mideast",
                17: "talk politics misc",
                18: "talk religion misc",
                19: "computer systems mac hardware"
            }
        }
        
        # Get the appropriate label map
        label_map = label_maps.get(args.data, {})
        
        # Loop over each development example
        with torch.no_grad():
            for dp in dev_data:
                try:
                    # Prepare input
                    token_ids = tokenizer.tokenize(dp.text)
                    input_tensor = torch.tensor(token_ids).unsqueeze(0)
                    input_length = torch.tensor([len(token_ids)])
                    
                    # Get model prediction
                    logits = model(input_tensor, input_length)
                    pred = logits.argmax(dim=1).item()
                    true_label = dp.label
                    
                    # Only collect actual errors
                    if pred != true_label:
                        errors.append({
                            'text': dp.text,
                            'true_label': true_label,
                            'true_label_name': label_map.get(true_label, str(true_label)),
                            'pred_label': pred,
                            'pred_label_name': label_map.get(pred, str(pred)),
                            'confidence': torch.softmax(logits, dim=1).max().item()
                        })
                except Exception as e:
                    print(f"Error processing example: {str(e)}")
                    continue
        
        # Report results
        total_errors = len(errors)
        print(f"\nFound {total_errors} errors in {len(dev_data)} examples")
        print(f"Error rate: {(total_errors/len(dev_data))*100:.2f}%\n")
        
        # Print detailed error analysis
        for i, error in enumerate(errors[:100], 1):
            print(f"Error {i}/{total_errors}:")
            print(f"Text snippet: {error['text'][:200]}...")
            print(f"True label: {error['true_label_name']}")
            print(f"Predicted label: {error['pred_label_name']}")
            print(f"Confidence: {error['confidence']*100:.2f}%")
            print("-" * 80)

    # Set to true if you want to save results
    if args.save_test_predictions:
        save_results(
            test_data,
            test_preds,
            os.path.join("results", f"mlp_{args.data}_test_predictions.csv"),
        )
