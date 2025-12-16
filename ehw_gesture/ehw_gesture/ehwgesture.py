from typing import Callable, Dict, Optional, Tuple
import numpy as np
from spikingjelly import datasets as sjds


class EHWGesture(sjds.NeuromorphicDatasetFolder):
    """
    Custom EHW Gesture dataset compatible with SpikingJelly framework.
    
    Args:
        root: Root directory path of the dataset
        data_type: 'event' for event data, 'frame' for integrated frames
        frames_number: Number of frames to integrate events into (used when data_type='frame')
        split_by: How to split events into frames - 'number' or 'time'
        duration: Duration for time-based splitting (used when split_by='time')
        custom_integrate_function: Custom function for event-to-frame integration
        custom_integrated_frames_dir_name: Custom directory name for cached frames
        transform: Transform to apply to samples
        target_transform: Transform to apply to labels
    
    Expected data structure:
        root/
        ├── class_0/
        │   ├── sample_001.npz
        │   ├── sample_002.npz
        │   └── ...
        ├── class_1/
        │   ├── sample_001.npz
        │   └── ...
        └── ...
    
    Each NPZ file must contain:
        - 't': timestamps (1D numpy array)
        - 'x': x-coordinates (1D numpy array)
        - 'y': y-coordinates (1D numpy array)
        - 'p': polarity (1D numpy array with binary values 0/1)
    """
    
    def __init__(
        self,
        root: str,
        data_type: str = 'event',
        frames_number: int = None,
        split_by: str = None,
        duration: int = None,
        custom_integrate_function: Callable = None,
        custom_integrated_frames_dir_name: str = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        """
        Initialize the EHWGesture dataset.
        """
        super().__init__(
            root,
            None,  # train parameter (not used, handled by custom split)
            data_type,
            frames_number,
            split_by,
            duration,
            custom_integrate_function,
            custom_integrated_frames_dir_name,
            transform,
            target_transform
        )
    
    @staticmethod
    def get_H_W() -> Tuple:
        """
        Return the height and width of the event data.
        
        Returns:
            Tuple of (height, width)
        
        Note: Modify these values to match your sensor resolution.
        Common resolutions:
            - DVS128: (128, 128)
            - DAVIS240: (180, 240)
            - DAVIS346: (260, 346)
        """
        return 160, 160  # Modify this to match your actual sensor resolution
    
    @staticmethod
    def load_origin_data(file_name: str) -> Dict:
        """
        Load event data from an NPZ file.
        
        Args:
            file_name: Path to the NPZ file
        
        Returns:
            Dictionary with keys ['t', 'x', 'y', 'p'] containing numpy arrays
        
        The NPZ file must contain:
            - 't': timestamps (monotonically increasing)
            - 'x': x-coordinates (0 to W-1)
            - 'y': y-coordinates (0 to H-1)
            - 'p': polarity (binary 0 or 1)
        """
        try:
            # Load the NPZ file
            data = np.load(file_name, allow_pickle=True)
            
            # Extract event data
            t = np.asarray(data['t'], dtype=np.int64)
            x = np.asarray(data['x'], dtype=np.int16)
            y = np.asarray(data['y'], dtype=np.int16)
            p = np.asarray(data['p'], dtype=np.int16)
            
            # Check for empty data
            if len(t) == 0:
                raise ValueError(f"No events in file {file_name}")
            
            # Ensure all arrays have the same length
            if not (len(t) == len(x) == len(y) == len(p)):
                raise ValueError(f"Array length mismatch in {file_name}: "
                               f"t={len(t)}, x={len(x)}, y={len(y)}, p={len(p)}")
            
            # Get sensor dimensions
            H, W = EHWGesture.get_H_W()
            
            # Clip coordinates to valid range (instead of asserting)
            x = np.clip(x, 0, W - 1).astype(np.int16)
            y = np.clip(y, 0, H - 1).astype(np.int16)
            
            # Ensure polarity is binary
            p = np.clip(p, 0, 1).astype(np.int16)
            
            # Ensure timestamps are sorted
            if not np.all(t[1:] >= t[:-1]):
                # Sort events by timestamp
                sort_idx = np.argsort(t)
                t = t[sort_idx]
                x = x[sort_idx]
                y = y[sort_idx]
                p = p[sort_idx]
            
            # Return as dictionary
            return {
                't': t,
                'x': x,
                'y': y,
                'p': p
            }
            
        except Exception as e:
            print(f"ERROR loading {file_name}: {str(e)}")
            raise
    
    @staticmethod
    def resource_url_md5() -> list:
        """
        Return download URLs and MD5 checksums for the dataset.
        
        Returns:
            List of tuples (filename, url, md5)
        
        Note: Implement this if you want automatic dataset downloading.
        Return empty list if dataset must be downloaded manually.
        """
        # Return empty list for manual download
        return []
    
    @staticmethod
    def downloadable() -> bool:
        """
        Return whether the dataset can be automatically downloaded.
        
        Returns:
            Boolean indicating if automatic download is supported
        """
        return False  # Set to True if you implement resource_url_md5()
    
    @staticmethod
    def extract_downloaded_files(download_root: str, extract_root: str):
        """
        Extract downloaded dataset files.
        
        Args:
            download_root: Directory containing downloaded files
            extract_root: Directory to extract files to
        
        Note: Implement this if you have compressed dataset files.
        """
        raise NotImplementedError(
            "Manual dataset preparation required. "
            "Organize NPZ files in class folders under the root directory."
        )
    
    @staticmethod
    def create_events_np_files(extract_root: str, events_np_root: str):
        """
        Convert raw event files to NPZ format.
        
        Args:
            extract_root: Directory containing raw event files
            events_np_root: Directory to save NPZ files
        
        Note: Implement this if you need to convert from another format.
        """
        raise NotImplementedError(
            "Dataset should already be in NPZ format. "
            "If converting from another format, implement this method."
        )


# Example usage and testing
if __name__ == "__main__":
    """
    Example code to test the dataset loading.
    """
    import torch
    from torch.utils.data import DataLoader
    
    # Path to your dataset
    dataset_root = ""  # Modify this path
    
    print("Testing EHWGesture dataset loading...")
    print("-" * 50)
    
    # Test 1: Load as events
    print("\n1. Loading as event data...")
    try:
        dataset_events = EHWGesture(
            root=dataset_root,
            data_type='event'
        )
        print(f"   Dataset size: {len(dataset_events)} samples")
        
        # Load one sample
        sample, label = dataset_events[0]
        print(f"   Sample keys: {sample.keys()}")
        print(f"   Number of events: {len(sample['t'])}")
        print(f"   Label: {label}")
        print(f"   Event data shapes - t:{sample['t'].shape}, x:{sample['x'].shape}, "
              f"y:{sample['y'].shape}, p:{sample['p'].shape}")
        print("   ✓ Event loading successful!")
    except Exception as e:
        print(f"   ✗ Event loading failed: {e}")
    
    # Test 2: Load as frames
    print("\n2. Loading as frame data...")
    try:
        dataset_frames = EHWGesture(
            root=dataset_root,
            data_type='frame',
            frames_number=16,
            split_by='number'
        )
        print(f"   Dataset size: {len(dataset_frames)} samples")
        
        # Load one sample
        frames, label = dataset_frames[0]
        print(f"   Frame shape: {frames.shape}")  # Should be (T, C, H, W)
        print(f"   Label: {label}")
        print(f"   Expected shape: (16, 2, 128, 128)")
        print("   ✓ Frame loading successful!")
    except Exception as e:
        print(f"   ✗ Frame loading failed: {e}")
    
    # Test 3: DataLoader integration
    print("\n3. Testing with PyTorch DataLoader...")
    try:
        dataset = EHWGesture(
            root=dataset_root,
            data_type='frame',
            frames_number=16,
            split_by='number'
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0  # Use 0 for testing, increase for actual training
        )
        
        # Load one batch
        batch_frames, batch_labels = next(iter(dataloader))
        print(f"   Batch frames shape: {batch_frames.shape}")  # (B, T, C, H, W)
        print(f"   Batch labels shape: {batch_labels.shape}")  # (B,)
        print("   ✓ DataLoader integration successful!")
    except Exception as e:
        print(f"   ✗ DataLoader integration failed: {e}")
    
    print("\n" + "-" * 50)
    print("Testing complete!")
    print("\nNext steps:")
    print("1. Verify the output shapes match your expectations")
    print("2. Check that frames are properly integrated from events")
    print("3. Integrate this dataset into your training script")
    print("4. Adjust get_H_W() if your sensor has different resolution")