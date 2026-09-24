import os
import urllib.request
import zipfile

def download_and_extract(url, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    zip_path = os.path.join(target_dir, "temp.zip")
    
    print(f"Downloading weights from {url}...")
    # Add a user-agent to avoid 403 Forbidden errors from some servers
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    
    with urllib.request.urlopen(req) as response, open(zip_path, 'wb') as out_file:
        data = response.read()
        out_file.write(data)
        
    print("Extracting files...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(target_dir)
        
    os.remove(zip_path)
    print(f"Extraction complete! Weights are now in {target_dir}")

if __name__ == "__main__":
    print("1. Downloading DINOv1 (ViT-B/16) weights...")
    b16_url = "https://huggingface.co/YiminJimmy/MM-OVSeg/resolve/main/B16.zip"
    download_and_extract(b16_url, "./checkpoint")
    
    # print("\n2. Downloading DINOv3 (ViT-L/14) weights...")
    # l14_url = "https://huggingface.co/YiminJimmy/MM-OVSeg/resolve/main/L14.zip"
    # download_and_extract(l14_url, "./checkpoint")
    
    print("\nAll model weights have been successfully downloaded and placed in the ./checkpoint directory!")
