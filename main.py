import clip
from data_loader import get_data_loaders
from clip_few_shot import extract_prototypes, evaluate_on_dataset
from eda import visualise
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the CLIP model
clip_model, transform = clip.load("ViT-B/32", device=device)

# Define data directories
train_dir = "./dataset/sleevetypes/train"
test_dir = "./dataset/sleevetypes/test"
val_dir = "./dataset/sleevetypes/val"

#visualise the data
visualise(train_dir)

#Load the data
train_loader, test_loader, val_loader = get_data_loaders(train_dir, test_dir, val_dir, transform,batch_size=32 )

# Extract prototypes from the training set
prototypes = extract_prototypes(train_loader,device, clip_model, shots=20)

# Evaluate on the validation dataset
correct_val, total_val = evaluate_on_dataset(val_loader, clip_model, prototypes, device)
val_accuracy = 100 * correct_val / total_val
print(f"Validation Accuracy: {val_accuracy:.2f}%")

# Test the classifier
correct_test, total_test = evaluate_on_dataset(test_loader, clip_model, prototypes, device)
test_accuracy = 100 * correct_test / total_test
print(f"Test Accuracy: {test_accuracy:.2f}%")
