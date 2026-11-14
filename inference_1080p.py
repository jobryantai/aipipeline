import argparse
import os
import sys
from PIL import Image
import torch
import torchvision.transforms as transforms
from tqdm import tqdm

sys.path.append("/mnt/d/aipipeline/animegan2-pytorch")
from model import Generator

def process_image(model, image, device):
    W, H = image.size
    new_W = (W // 32) * 32
    new_H = (H // 32) * 32

    transform = transforms.Compose([
        transforms.Resize((new_H, new_W)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    input_tensor = transform(image).unsqueeze(0).to(device).float()

    with torch.no_grad():
        output_tensor = model(input_tensor)

    output_tensor = output_tensor.squeeze(0).cpu().float()
    output_tensor = torch.clamp(output_tensor * 0.5 + 0.5, 0.0, 1.0)
    
    output_image = transforms.ToPILImage()(output_tensor)

    if output_image.size != (W, H):
        output_image = output_image.resize((W, H), Image.Resampling.BICUBIC)

    return output_image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_path", default="/mnt/d/aipipeline/animegan2-pytorch/weights/hayao_clean.pt")
    parser.add_argument("--threshold", type=float, default=2.5)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[*] Loading model from {args.model_path} on {device}...")
    model = Generator().to(device).eval()
    
    checkpoint = torch.load(args.model_path, map_location=device)
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint.get('model_state', checkpoint)))
    else:
        state_dict = checkpoint

    new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=True)

    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")
    image_files = sorted([f for f in os.listdir(args.input_dir) if f.lower().endswith(valid_extensions)])

    print(f"[*] Processing {len(image_files)} frames...")
    for filename in tqdm(image_files):
        in_path = os.path.join(args.input_dir, filename)
        out_path = os.path.join(args.output_dir, filename)
        
        try:
            image = Image.open(in_path).convert("RGB")
            output_image = process_image(model, image, device)
            output_image.save(out_path)
        except Exception as e:
            print(f"[!] Error processing {filename}: {e}")

if __name__ == "__main__":
    main()
