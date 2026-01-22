import torch
from torchvision import transforms
from PIL import Image
from model import SimpleSceneCNN
import os
import sys

# Standard Samsung Settings Logic
CLASSES = ['animation', 'cinema', 'sports'] 

def get_tv_settings(scene_type):
    settings = {
        'sports': "Mode: SPORTS | Motion Xcelerator: ON | Color Temp: COOL",
        'cinema': "Mode: FILMMAKER | Brightness: DIM | Color Temp: WARM2",
        'animation': "Mode: DYNAMIC | Contrast: HIGH | Color: VIVID"
    }
    return settings.get(scene_type, "Mode: STANDARD")

def run_inference(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = SimpleSceneCNN(num_classes=len(CLASSES)).to(device)
    model.load_state_dict(torch.load('./saved_models/tv_scene_model.pth', map_location=device))
    model.eval()

    # Professional MobileNet Normalization
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img = Image.open(image_path).convert('RGB')
    img_t = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_t)
        _, predicted_idx = torch.max(output, 1)
        scene = CLASSES[predicted_idx.item()]

    print("\n" + "="*45)
    print("      SAMSUNG SMART TV AI PROCESSOR (v2.0)")
    print("="*45)
    print(f"DETECTED CONTENT: {scene.upper()}")
    print(f"HARDWARE ACTION:  {get_tv_settings(scene)}")
    print("="*45 + "\n")

if __name__ == "__main__":
    test_img = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
    run_inference(test_img)