import sys
import re

def toggle_backbone(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    # Find which one is currently active
    dino_v1_active = bool(re.search(r'^from \.MMOV import MMOV', content, re.MULTILINE))
    dino_v3_active = bool(re.search(r'^from \.MMOVL14 import MMOV', content, re.MULTILINE))

    if dino_v1_active:
        print("Switching from DINOv1 to DINOv3...")
        content = re.sub(r'^(from \.MMOV import MMOV)', r'#\1', content, flags=re.MULTILINE)
        content = re.sub(r'^#(from \.MMOVL14 import MMOV)', r'\1', content, flags=re.MULTILINE)
    elif dino_v3_active:
        print("Switching from DINOv3 to DINOv1...")
        content = re.sub(r'^(from \.MMOVL14 import MMOV)', r'#\1', content, flags=re.MULTILINE)
        content = re.sub(r'^#(from \.MMOV import MMOV)', r'\1', content, flags=re.MULTILINE)
    else:
        # Default behavior if none are properly formatted, let's just make DINOv3 active
        print("Couldn't detect active backbone. Defaulting to DINOv3...")
        content = re.sub(r'^#?(from \.MMOV import MMOV)', r'#\1', content, flags=re.MULTILINE)
        content = re.sub(r'^#?(from \.MMOVL14 import MMOV)', r'\1', content, flags=re.MULTILINE)

    with open(file_path, 'w') as file:
        file.write(content)
        
    print("Backbone switched successfully.")

if __name__ == "__main__":
    init_file = "mmov_seg/__init__.py"
    toggle_backbone(init_file)
