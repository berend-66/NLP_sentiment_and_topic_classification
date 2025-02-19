#!/usr/bin/env python3
import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

# If you're running in Google Colab, these imports and functions will work.
# Otherwise, you may remove or modify them as needed.
try:
    from google.colab import drive
except ImportError:
    drive = None

def mount_drive():
    """Mount Google Drive and set the working directory (for Colab)."""
    if drive is not None:
        drive.mount('/content/drive')
        os.chdir('/content/drive/My Drive/NLP/a1-berend-66')
    else:
        print("Google Colab drive not detected; skipping drive mount.")

# Local module imports (ensure these modules are in your PYTHONPATH)
from utils import DataPoint, DataType, accuracy, load_data, save_results
from multilayer_perceptron import MultilayerPerceptronModel, Trainer, BOWDataset, Tokenizer, get_label_mappings


def benchmark_inference(model, dataset, device, batch_size, num_examples=1000, num_iterations=10):
    """
    Benchmarks the inference speed of the model on the given dataset.
    
    Args:
        model: The neural network model.
        dataset: The dataset to run inference on.
        device: The device (CPU/GPU) to use.
        batch_size: Batch size for inference.
        num_examples: Number of examples to use from the dataset.
        num_iterations: Number of times to run inference for averaging.
    
    Returns:
        A tuple (avg_time, std_time) representing the average and standard deviation 
        of the inference time in milliseconds.
    """
    model.eval()
    model.to(device)
    subset_indices = list(range(min(num_examples, len(dataset))))
    subset = Subset(dataset, subset_indices)
    dataloader = DataLoader(subset, batch_size=batch_size, shuffle=False, pin_memory=True)
    
    times = []
    for _ in range(num_iterations):
        torch.cuda.synchronize(device)
        start_time = time.time()
        with torch.no_grad():
            for inputs, lengths, _ in dataloader:
                inputs = inputs.to(device, non_blocking=True)
                lengths = lengths.to(device, non_blocking=True)
                _ = model(inputs, lengths)
        torch.cuda.synchronize(device)
        end_time = time.time()
        times.append((end_time - start_time) * 1000)  # convert to milliseconds
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    return avg_time, std_time

def main():
    # Check if GPU is available
    print("GPU available:", torch.cuda.is_available())

    # Mount drive if in Colab; ignore if running locally.
    mount_drive()

    # -------------------------------
    # Benchmark on SST2 Dataset
    # -------------------------------
    print("\nBenchmarking on SST2 dataset...")
    data_type = DataType("sst2")
    _, _, dev_data, _ = load_data(data_type)
    tokenizer = Tokenizer(dev_data, max_vocab_size=20000)  # adjust as needed

    label2id, id2label = get_label_mappings(dev_data)
    dev_ds = BOWDataset(dev_data, tokenizer, label2id, max_length=250)

    # Instantiate the model for SST2 dataset
    hidden_dims = [200, 100, 50]
    activation = "relu"
    dropout_rate = 0.5  # adjust as needed
    model = MultilayerPerceptronModel(
        vocab_size=len(tokenizer.token2id),
        num_classes=2,  # change as needed for your dataset
        padding_index=Tokenizer.TOK_PADDING_INDEX,
        hidden_dims=hidden_dims,
        activation=activation,
        dropout_rate=dropout_rate
    )

    # Get the device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Benchmark inference for various batch sizes
    batch_sizes = [1, 2, 4, 8, 16, 32, 64]
    for bs in batch_sizes:
        avg_time, std_time = benchmark_inference(model, dev_ds, device, batch_size=bs)
        print(f"Batch size {bs}: Avg time = {avg_time:.2f} ms, Std = {std_time:.2f} ms")

    # -------------------------------
    # Benchmark on Newsgroups Dataset
    # -------------------------------
    print("\nBenchmarking on Newsgroups dataset...")
    data_type = DataType("newsgroups")
    _, _, dev_data, _ = load_data(data_type)
    tokenizer = Tokenizer(dev_data, max_vocab_size=30000)  # adjust as needed

    max_length = 500  # Maximum document length
    label2id, id2label = get_label_mappings(dev_data)
    dev_ds = BOWDataset(dev_data, tokenizer, label2id, max_length=max_length)

    # Instantiate the model for newsgroups dataset
    hidden_dims = [512, 256, 128]
    activation = "relu"
    dropout_rate = 0.5
    model = MultilayerPerceptronModel(
        vocab_size=len(tokenizer.token2id),
        num_classes=len(label2id),  # number of classes in the dataset
        padding_index=Tokenizer.TOK_PADDING_INDEX,
        hidden_dims=hidden_dims,
        activation=activation,
        dropout_rate=dropout_rate
    )

    # Get the device again (in case it changed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Benchmark inference for various batch sizes
    for bs in batch_sizes:
        avg_time, std_time = benchmark_inference(model, dev_ds, device, batch_size=bs)
        print(f"Batch size {bs}: Avg time = {avg_time:.2f} ms, Std = {std_time:.2f} ms")

if __name__ == "__main__":
    main()
