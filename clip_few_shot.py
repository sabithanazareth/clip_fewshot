import torch

def extract_prototypes(dataloader,device, model, shots=20):
    # Your implementation here, as provided in your code


    class_embeddings = {}
    counts = {}
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            embeddings = model.encode_image(images)
            for emb, label in zip(embeddings, labels):
                lbl = label.item()
                if lbl not in class_embeddings:
                    class_embeddings[lbl] = []
                    counts[lbl] = 0
                
                if counts[lbl] < shots:
                    class_embeddings[lbl].append(emb)
                    counts[lbl] += 1
    
    # Average the embeddings for each class
    for label, embs in class_embeddings.items():
        class_embeddings[label] = torch.stack(embs).mean(dim=0)
    return class_embeddings

def classify_image(image, model, prototypes):
    # Your implementation here, as provided in your code
    with torch.no_grad():
        image_embedding = model.encode_image(image)
        max_similarity = float("-inf")
        predicted_class = -1
        for label, prototype in prototypes.items():
            similarity = torch.nn.functional.cosine_similarity(image_embedding, prototype.unsqueeze(0))
            if similarity > max_similarity:
                max_similarity = similarity
                predicted_class = label
    return predicted_class

def evaluate_on_dataset(loader, model, prototypes, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            for image, label in zip(images, labels):
                predicted_label = classify_image(image.unsqueeze(0), model, prototypes)
                if predicted_label == label.item():
                    correct += 1
                total += 1
    return correct, total
