import numpy as np
import os

# Check a few files from each class
data_root = "events_np"

for class_name in os.listdir(data_root):
    class_path = os.path.join(data_root, class_name)
    if not os.path.isdir(class_path):
        continue
    
    files = os.listdir(class_path)[:5]  # Check first 5 files
    print(f"\nChecking class: {class_name}")
    
    for npz_file in files:
        npz_path = os.path.join(class_path, npz_file)
        try:
            data = np.load(npz_path)
            t = data['t']
            x = data['x']
            y = data['y']
            p = data['p']
            
            print(f"  {npz_file}: {len(t)} events, "
                  f"x:[{x.min()},{x.max()}], "
                  f"y:[{y.min()},{y.max()}], "
                  f"p:{np.unique(p)}, "
                  f"dtypes: t={t.dtype}, x={x.dtype}, y={y.dtype}, p={p.dtype}")
                  
            # Check for issues
            if len(t) == 0:
                print(f"    WARNING: Empty file!")
            if x.max() >= 128 or y.max() >= 128:
                print(f"    WARNING: Coordinates out of bounds!")
            if not np.all((p == 0) | (p == 1)):
                print(f"    WARNING: Invalid polarity values!")
                
        except Exception as e:
            print(f"  {npz_file}: ERROR - {e}")