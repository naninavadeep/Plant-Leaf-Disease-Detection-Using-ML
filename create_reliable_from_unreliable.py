import os
import shutil

ORIG_DIR = 'PlantVillage'
UNRELIABLE_DIR = 'PlantVillage_unreliable'
RELIABLE_DIR = 'PlantVillage_reliable'

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

for class_folder in os.listdir(ORIG_DIR):
    orig_class_path = os.path.join(ORIG_DIR, class_folder)
    unreliable_class_path = os.path.join(UNRELIABLE_DIR, class_folder)
    reliable_class_path = os.path.join(RELIABLE_DIR, class_folder)
    if not os.path.isdir(orig_class_path):
        continue
    ensure_dir(reliable_class_path)
    unreliable_files = set(os.listdir(unreliable_class_path)) if os.path.exists(unreliable_class_path) else set()
    for fname in os.listdir(orig_class_path):
        if fname not in unreliable_files:
            src = os.path.join(orig_class_path, fname)
            dst = os.path.join(reliable_class_path, fname)
            print(f"Copying {src} to {dst}")
            shutil.copy2(src, dst) 