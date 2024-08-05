import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import requests
from io import BytesIO

# Import the ViT model (Assuming your package is named vit_pytorch)
from vit_pytorch import ViT

def test_vit():
    # Initialize the model
    model = ViT(
        image_size=224,
        patch_size=16,
        num_classes=1000,
        dim=1024,
        depth=6,
        heads=8,
        mlp_dim=2048,
        pool='cls',
        channels=3,
        dim_head=64,
        dropout=0.1,
        emb_dropout=0.1
    )
    
    # Set the model to evaluation mode
    model.eval()

    # Sample image for testing (URL to an image)
    url = 'https://example.com/sample.jpg'
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))

    # Transform the image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    img = transform(img).unsqueeze(0)  # Add batch dimension

    # Forward pass through the model
    with torch.no_grad():
        logits = model(img)
    
    # Print the output logits
    print("Logits:", logits)

if __name__ == "__main__":
    test_vit()
