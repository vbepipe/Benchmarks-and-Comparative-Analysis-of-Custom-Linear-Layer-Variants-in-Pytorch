import math 
from typing import Tuple, Dict, Any, Optional 
import numpy as np 
import torch 
torch.set_num_threads(8) # Set to number of physical cores, not logical
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
from torch.utils.data import DataLoader, TensorDataset, Dataset 
from torchvision import datasets, transforms  # [NEW] Import torchvision for benchmark datasets 
import random
import time
from datetime import datetime
from typing import Tuple, Dict, Any, Optional



##############################
##############################
##############################
import sys
sys.dont_write_bytecode = True

import shutil
import os
for root, dirs, files in os.walk('.'):
    if '__pycache__' in dirs:
        shutil.rmtree(os.path.join(root, '__pycache__'))

import gc
import os
os.environ['TORCHINDUCTOR_FORCE_DISABLE_CACHES'] = '1'


# For custom caching implementations
def clear_all_caches():
    """Clear various Python caches"""
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Force garbage collection
    gc.collect()
    

clear_all_caches()
##############################
##############################
##############################



# ============================================================================ 
# CONFIGURATION CONSTANTS (UPDATED FOR BENCHMARKS) 
# ============================================================================ 

# Replace synthetic dataset with standard benchmarks: 
DATASET_NAMES = ['FashionMNIST', 'MNIST'] #['MNIST', 'FashionMNIST'] # , 'CIFAR10']  # [NEW] List of benchmark datasets to use 

# Model and training configuration (tuned for smaller laptop-friendly runs) 
HIDDEN_DIM: int = 128 #256 # 64 # 243 # 126 # 128   # Reduced hidden size from 256 to 128 for faster training 
LEARNING_RATE: float = 1e-3 #1e-3 # 0.09  # 1e-3 
momentum_ = 0.0 # 0.7 # 0.9 
weight_decay_ = 0.0 #1e-4 #1e-4 
DROPOUT_RATE: float = 0.0 # 0.2 #0.0 # LEARNING_RATE # 0.2 
EPOCHS_PER_DATASET: int = 30 #100 # 30 100 # 10  # Fewer epochs for quick runs 
BATCH_SIZE: int = 128 # 64      # Moderate batch size for memory constraints 

# Initialization configuration 
INIT_RANGE: float = 0.01    # Parameter initialization range [-INIT_RANGE, +INIT_RANGE] 

info = torch.finfo(torch.float32) 

# ============================================================================  
# DATA LOADING UTILITIES  
# ============================================================================  

def scale_255(t: torch.Tensor) -> torch.Tensor:
    return t * 255.0 


# Move this OUTSIDE the get_data_loaders function
class FlattenedDataset(Dataset):
    def __init__(self, base_dataset):
        self.base = base_dataset
    def __len__(self):
        return len(self.base)
    def __getitem__(self, idx):
        x, y = self.base[idx]
        # x shape: [channels, H, W]; flatten to [channels*H*W]
        return x.view(-1), y


def scale_2(t: torch.Tensor) -> torch.Tensor:
    return t + 0.0001 


def get_data_loaders(dataset_name: str, batch_size: int) -> Tuple[DataLoader, DataLoader, int, int]:
    """
    Create DataLoaders for a given benchmark dataset using torchvision.
    
    Args:
        dataset_name: Name of the dataset ('MNIST', 'FashionMNIST', or 'CIFAR10').
        batch_size: Batch size for training and test loaders.
    
    Returns:
        (train_loader, test_loader, input_dim, n_classes)
    """
    # Define transforms: convert PIL images to tensors and normalize.
    if dataset_name in ['FashionMNIST', 'MNIST']: #['MNIST', 'FashionMNIST']:
        # Grayscale 28x28 images -> flatten later in model or via custom dataset
        transform = transforms.Compose([
            #transforms.RandomCrop(28, padding=4),
            transforms.ToTensor(),  # range [0,1], shape [1,28,28]
            #transforms.Lambda(scale_255),
            #transforms.Lambda(scale_2),
            #transforms.Normalize((0.5,),(0.5,)),
        ])
        n_channels = 1
        img_size = 28
    elif dataset_name == 'CIFAR10':
        transform = transforms.Compose([
            #transforms.RandomCrop(32, padding=4),  # Keep this for variety
            transforms.ToTensor(),  # range [0,1], shape [3,32,32]
            #transforms.Lambda(scale_255),
            #transforms.Lambda(scale_2),
            #transforms.Normalize((0.5,),(0.5,)),
        ])
        n_channels = 3
        img_size = 32
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    

    # Load training and test sets (with automatic download)
    if dataset_name == 'MNIST':
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        n_classes = 10
    elif dataset_name == 'FashionMNIST':
        train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset  = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
        n_classes = 10
    elif dataset_name == 'CIFAR10':
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset  = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        n_classes = 10
    

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=8,
        pin_memory=False,
        persistent_workers=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=8,
        pin_memory=False, 
        persistent_workers=True
    )
    
    # Compute input dimension (flattened image size)
    #input_dim = n_channels * img_size * img_size
    input_dim = (n_channels, img_size, img_size)

    return train_loader, test_loader, input_dim, n_classes



# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def uniform_init(shape: Tuple[int, ...]) -> torch.Tensor:
    """
    Initialize a tensor with uniform distribution in the range [-INIT_RANGE, +INIT_RANGE].
    
    This controlled initialization helps ensure stable training across different
    layer types by preventing extreme parameter values.
    
    Args:
        shape: Shape of the tensor to initialize
        
    Returns:
        Initialized tensor with uniform distribution
    """
    tensor = torch.empty(*shape)
    return nn.init.uniform_(tensor, -INIT_RANGE, INIT_RANGE)



def ones_init(shape: Tuple[int, ...]) -> torch.Tensor:
    """
    Initialize a tensor with all values set to 1.0.
    
    Args:
        shape: Shape of the tensor to initialize
        
    Returns:
        Tensor filled with 1.0 values
    """
    return torch.ones(*shape)



def uniform_initV(shape: Tuple[int, ...]) -> torch.Tensor:
    """
    Initialize a tensor with uniform distribution in the range [-INIT_RANGE, +INIT_RANGE].
    
    This controlled initialization helps ensure stable training across different
    layer types by preventing extreme parameter values.
    
    Args:
        shape: Shape of the tensor to initialize
        
    Returns:
        Initialized tensor with uniform distribution
    """
    tensor = torch.empty(*shape)
    return nn.init.uniform_(tensor, 1, 1)


class FourFoldAugmentedDataset(Dataset):
    """
    Custom dataset that creates 4 variants of each training image:
    1. Original image (imagesA)
    2. Horizontally flipped image (imagesB) 
    3. Vertically flipped original image (imagesA + vertical flip)
    4. Vertically flipped horizontally flipped image (imagesB + vertical flip)
    """
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
    
    def __len__(self):
        # Return 4 times the original dataset length
        return 4 * len(self.base_dataset)
    
    def __getitem__(self, idx):
        # Map index to original dataset and variant type
        base_idx = idx // 4        # Which original image
        variant = idx % 4          # Which variant (0,1,2,3)
        
        # Get the base image and label
        img, label = self.base_dataset[base_idx]
        
        # Apply transformations based on variant
        if variant == 0:
            # Original image (imagesA)
            return img, label
        elif variant == 1:
            # Horizontally flipped image (imagesB)
            return transforms.functional.hflip(img), label
        elif variant == 2:
            # Vertically flipped original (imagesA + vertical flip)
            return transforms.functional.vflip(img), label
        elif variant == 3:
            # Both horizontal and vertical flip (imagesB + vertical flip)
            hflipped = transforms.functional.hflip(img)
            return transforms.functional.vflip(hflipped), label




# ============================================================================  
# CUSTOM LINEAR LAYER IMPLEMENTATIONS  
# ============================================================================  



class StandardLinear(nn.Module):
    """
    Standard linear transformation: output = input @ weight.T + bias
    
    This serves as a baseline for comparison with other custom layers.
    Implements the standard affine transformation used in typical neural networks.
    """
    
    def __init__(self, input_dim: int, output_dim: int):
        """
        Initialize the standard linear layer.
        
        Args:
            input_dim: Number of input features
            output_dim: Number of output features
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Initialize weight matrix and bias vector
        self.weight = nn.Parameter(uniform_init((output_dim, input_dim)))
        self.bias = nn.Parameter(uniform_init((output_dim,)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the standard linear layer.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            
        Returns:
            Output tensor of shape [batch_size, output_dim]
        """

        return torch.matmul(x, self.weight.T) + self.bias







# ============================================================================  
# NEURAL NETWORK ARCHITECTURE (UNCHANGED)  
# ============================================================================  

class CustomLayerNetwork(nn.Module):
    """
    A three-layer fully-connected network using a custom transformation as one hidden layer.
    Architecture: Flattened Input -> Input Linear -> ReLU -> Dropout -> Custom Layer -> Dropout -> Output Linear.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, layer_type: str, dropout_rate: float = 0.2):
        super().__init__()
        
        #self.input_layer = nn.Linear(input_dim, hidden_dim)

        channels, height, width = input_dim

        # input_dim = n_channels * img_size * img_size


        # Convolutional feature extractor
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=3, padding=1, stride=1, bias=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1, bias=True)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1, bias=True)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1, bias=True )
        self.pool = nn.MaxPool2d(2, 2)


        # Calculate the flattened size after convolutions and pooling
        # After conv1 + pool: (H/2, W/2)
        # After conv2 + pool: (H/4, W/4)
        # After conv3 + pool: (H/8, W/8)
        # After conv4 + pool: (H/16, W/16)
        conv_output_size = 256 * (height // 16) * (width // 16)


        # Map string to custom layer class (as before)
        custom_layer_map = {

            'standard': StandardLinear,

        }
        if layer_type not in custom_layer_map:
            raise ValueError(f"Unknown layer type: {layer_type}")
        
        # Output layer to number of classes
        self.dropout = nn.Dropout(dropout_rate)
        
        self.custom_layer = custom_layer_map[layer_type](hidden_dim, hidden_dim) 
        self.layerVinput  = custom_layer_map[layer_type](conv_output_size, hidden_dim)
        self.output_layer = custom_layer_map[layer_type](hidden_dim, output_dim) 




    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = F.gelu(self.conv1(x))
        x = self.pool(x)
        x = F.gelu(self.conv2(x))
        x = self.pool(x)
        x = F.gelu(self.conv3(x))
        x = self.pool(x)
        x = F.gelu(self.conv4(x))
        x = self.pool(x)
        
        # Flatten for linear layers
        x = x.view(x.size(0), -1)


        # Initial layer with F.silu
        x = F.gelu(self.layerVinput(x))
        #x = self.dropout(x)

        residual = x.clone()  # Store residual

        x = F.silu(self.custom_layer(x))
        x = F.silu(self.custom_layer(x))
        #x = self.dropout(x)

        x = x + residual
        x = F.gelu(self.custom_layer(x))
        x = F.gelu(self.custom_layer(x))
        #x = self.dropout(x)

        x = x + residual
        x = F.gelu(self.custom_layer(x))
        x = F.gelu(self.custom_layer(x))

        x = x + residual
        x = F.gelu(self.custom_layer(x))
        x = F.gelu(self.custom_layer(x))


        # Output layer (no activation for classification)
        return self.output_layer(x)



# ============================================================================  
# TRAINING AND EVALUATION UTILITIES (UNCHANGED)  
# ============================================================================  


def train_model(model: nn.Module, train_loader: DataLoader, criterion: nn.Module, test_loader,
                optimizer: optim.Optimizer, num_epochs: int = 5, verbose: bool = True) -> None:

    avg_loss_list = []
    loss_threshold = 0.00001 #0.199999  ### 0.02  0.11  0.19
    patience_window = 3  ### number of recent epochs to average
    test_loss_threshold = loss_threshold # 0.1
    model.train()

    for epoch in range(1, num_epochs + 1):
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        avg_loss_list.append(avg_loss)

        test_loss, test_acc = evaluate_model(model, test_loader, criterion)

        if verbose:
            print(f"Epoch {epoch}/{num_epochs}: Loss = {avg_loss:.9f}")
            
            print(f"Test Loss: {test_loss:.9f} | Test Accuracy: {test_acc:.9f}%")
            

        if test_loss < test_loss_threshold:
            print(f"Breaking early as Test Loss  "
                        f"({test_loss:.9f}) is below threshold of {test_loss_threshold}")
            break

        if avg_loss < loss_threshold:
            print(f"Breaking early as average of last avg_loss  "
                        f"({avg_loss:.9f}) is below threshold of {loss_threshold}")
            break

        if len(avg_loss_list) >= patience_window:
            recent_avg = sum(avg_loss_list[-patience_window:]) / patience_window
            if verbose:
                print(f"Average of last {patience_window} losses is " f"({recent_avg:.9f})")
            if recent_avg < loss_threshold or avg_loss < loss_threshold:
                print(f"Breaking early as average of last {patience_window} losses "
                        f"({recent_avg:.9f}) is below threshold of {loss_threshold}")
                break




@torch.no_grad()
def evaluate_model(model: nn.Module, test_loader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for batch_x, batch_y in test_loader:
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == batch_y).sum().item()
        total += batch_y.size(0)
    avg_loss = total_loss / len(test_loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy

def display_results_summary(dataset_name: str, results: Dict[str, Dict[str, float]]) -> None:
    """
    Print a summary table of losses and accuracies for each layer type on the given dataset.
    """
    print(f"\n{'='*60}\nResults on {dataset_name}\n{'='*60}")
    print(f"{'Layer Type':<15}{'Test Loss':<12}{'Accuracy (%)':<15}{'Rank':<5}")
    print('-'*60)
    # Sort by accuracy
    sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    for rank, (layer, metrics) in enumerate(sorted_results, 1):
        print(f"{layer.capitalize():<15}{metrics['loss']:<12.4f}{metrics['accuracy']:<15.2f}#{rank}")
    print('-'*60)
    best_acc_layer = sorted_results[0]
    best_loss_layer = min(results.items(), key=lambda x: x[1]['loss'])
    print(f"🏆 BEST ACCURACY: {best_acc_layer[0].capitalize()} ({best_acc_layer[1]['accuracy']:.2f}%)")
    print(f"📉 LOWEST LOSS: {best_loss_layer[0].capitalize()} ({best_loss_layer[1]['loss']:.4f})\n")

# ============================================================================  
# MAIN EXPERIMENT PIPELINE (UPDATED)  
# ============================================================================  

def run_dataset_experiments():
    """
    Run experiments for each dataset in DATASET_NAMES using all custom layer types.
    """
    layer_types = [ 
                    'standard', 
                ] 
    
    for dataset_name in DATASET_NAMES:
        print(f"\nRunning experiments on dataset: {dataset_name}")
        # Load data and get input/output sizes
        train_loader, test_loader, input_dim, n_classes = get_data_loaders(dataset_name, BATCH_SIZE)
        print(f"Dataset: {dataset_name} | Input dim: {input_dim} | Classes: {n_classes}")
        print(f"Train samples: {len(train_loader.dataset)}, Test samples: {len(test_loader.dataset)}")

        
        results = {} 
        for layer_type in layer_types: 
            print(f"\n-- Training model with {layer_type} layer --") 
            model = CustomLayerNetwork(input_dim, HIDDEN_DIM, n_classes, layer_type, dropout_rate=DROPOUT_RATE) 
            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=weight_decay_, eps=1e-34) 
            #optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=momentum_, weight_decay=weight_decay_) 
            criterion = nn.CrossEntropyLoss()
            train_model(model, train_loader, criterion, test_loader, optimizer, num_epochs=EPOCHS_PER_DATASET, verbose=True) 
            test_loss, test_acc = evaluate_model(model, test_loader, criterion) 
            results[layer_type] = {'loss': test_loss, 'accuracy': test_acc} 
            print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.2f}%") 
        
        display_results_summary(dataset_name, results) 




def main():
    SEED_ = int(random.randint(1, 1000000)) 
    print("Seed number", SEED_)
    print("LEARNING_RATE: ", LEARNING_RATE)
    print("DROPOUT_RATE: ", DROPOUT_RATE)
    print("WEIGHT_DECAY_RATE: ", weight_decay_)
    print("Neurons: ", HIDDEN_DIM)

    torch.manual_seed(SEED_)
    np.random.seed(SEED_)
    run_dataset_experiments()
    print("\nAll experiments completed successfully.")






if __name__ == "__main__": 
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    multiprocessing.freeze_support()  # Add this for Windows

    main()
    