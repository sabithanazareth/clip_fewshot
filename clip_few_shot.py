import torch

# Function to extract prototypes for few-shot learning
def extract_prototypes(dataloader, device, model, shots=20):
    # Initialize dictionaries to store class embeddings and counts
    class_embeddings = {}  # Stores embeddings for each class
    counts = {}  # Keeps track of the number of embeddings per class
    
    # Disable gradient calculation for efficiency
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)  # Move images to the specified device
            embeddings = model.encode_image(images)  # Get CLIP embeddings for images
            
            # Loop through embeddings and labels
            for emb, label in zip(embeddings, labels):
                lbl = label.item()  # Get the numeric label
                if lbl not in class_embeddings:
                    class_embeddings[lbl] = []  # Initialize list for each class
                    counts[lbl] = 0
                
                if counts[lbl] < shots:  # Limit the number of embeddings per class
                    class_embeddings[lbl].append(emb)  # Add the embedding to the class
                    counts[lbl] += 1
    
    # Average the embeddings for each class
    for label, embs in class_embeddings.items():
        class_embeddings[label] = torch.stack(embs).mean(dim=0)
    return class_embeddings

# Function to classify an image using prototypes
def classify_image(image, model, prototypes):
    with torch.no_grad():
        image_embedding = model.encode_image(image)  # Get CLIP embedding for the image
        max_similarity = float("-inf")  # Initialize maximum similarity
        predicted_class = -1  # Initialize predicted class
        for label, prototype in prototypes.items():
            similarity = torch.nn.functional.cosine_similarity(image_embedding, prototype.unsqueeze(0))
            # Calculate cosine similarity between image and prototype
            if similarity > max_similarity:
                max_similarity = similarity  # Update maximum similarity
                predicted_class = label  # Update predicted class
    return predicted_class

# Function to evaluate a model on a dataset
def evaluate_on_dataset(loader, model, prototypes, device):
    correct = 0  # Initialize correct predictions counter
    total = 0  # Initialize total predictions counter
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)  # Move images to the specified device
            for image, label in zip(images, labels):
                predicted_label = classify_image(image.unsqueeze(0), model, prototypes)
                # Classify the image using prototypes and model
                if predicted_label == label.item():  # Check if prediction is correct
                    correct += 1  # Increment correct predictions counter
                total += 1  # Increment total predictions counter
    return correct, total
