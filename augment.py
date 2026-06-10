import os
import numpy as np
from PIL import Image
import random

BASE_DIR      = r"D:\Users\Mimouna Zughdani\Desktop\558pr\Training-&-Research"
ORIGINAL_DIR  = os.path.join(BASE_DIR, 'dataset_original')
AUGMENTED_DIR = os.path.join(BASE_DIR, 'dataset_augmented')
TARGET_SIZE   = (224, 224)
SAMPLES_PER_IMAGE = 20

os.makedirs(AUGMENTED_DIR, exist_ok=True)

def augment_image(img_path):
    img = Image.open(img_path).convert('RGB')

    # Random rotation
    angle = random.randint(0, 359)
    img = img.rotate(angle, expand=True)

    # Random zoom
    w, h = img.size
    zoom_type = random.choice(['in', 'out', 'none'])
    if zoom_type == 'in':
        factor = random.uniform(1.1, 1.4)
        new_w, new_h = int(w / factor), int(h / factor)
        left, top = (w - new_w) // 2, (h - new_h) // 2
        img = img.crop((left, top, left + new_w, top + new_h))
    elif zoom_type == 'out':
        factor = random.uniform(0.7, 0.9)
        new_w, new_h = int(w * factor), int(h * factor)
        canvas = Image.new('RGB', (w, h), (0, 0, 0))
        img = img.resize((new_w, new_h), Image.LANCZOS)
        canvas.paste(img, ((w - new_w) // 2, (h - new_h) // 2))
        img = canvas

    # Random brightness
    brightness = random.uniform(0.6, 1.5)
    arr = np.clip(np.array(img).astype(np.float32) * brightness, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)

    # Resize to target size
    img = img.resize(TARGET_SIZE, Image.LANCZOS)
    return img

# Process all classes
for class_name in os.listdir(ORIGINAL_DIR):
    class_path = os.path.join(ORIGINAL_DIR, class_name)
    if not os.path.isdir(class_path):
        continue

    class_output_dir = os.path.join(AUGMENTED_DIR, class_name)
    os.makedirs(class_output_dir, exist_ok=True)

    orig_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if len(orig_files) == 0:
        print(f"⚠️ {class_name}: empty, skipping")
        continue

    print(f"\n📁 {class_name}: {len(orig_files)} original images")
    for idx, filename in enumerate(orig_files):
        src = os.path.join(class_path, filename)
        name, _ = os.path.splitext(filename)
        for i in range(SAMPLES_PER_IMAGE):
            augment_image(src).save(
                os.path.join(class_output_dir, f"aug_{name}_{i:02d}.jpg"), quality=100)
        print(f"   Progress: {idx+1}/{len(orig_files)} images processed", end='\r')

    print(f"✅ {class_name}: {len(orig_files) * SAMPLES_PER_IMAGE} images generated")

print("\n🎉 All classes completed!")
print(f"📊 Total expected: {SAMPLES_PER_IMAGE * 200 * 10:,} images")



