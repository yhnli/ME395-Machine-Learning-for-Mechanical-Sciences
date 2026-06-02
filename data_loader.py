"""UCI Dataset Loader for Blood Pressure Estimation.

This module handles loading .mat files from the UCI Cuff-less Blood Pressure
Estimation dataset and generates windowed samples with BP targets.
"""

import numpy as np
from scipy import signal
from scipy.io import loadmat, test
import h5py
from typing import Tuple, List, Optional
from tqdm import tqdm
import warnings

class UCILoader:
    """Load UCI Cuff-less BP dataset and generate windowed samples.
    
    The dataset contains synchronized PPG, ABP, and ECG signals sampled at 125 Hz.
    We extract 4-second windows.
    """
    
    SAMPLING_RATE = 125  # Hz
    WINDOW_SECONDS = 4
    WINDOW_SIZE = SAMPLING_RATE * WINDOW_SECONDS  # 500 samples

    BP_MIN, BP_MAX = 50, 220  
    MIN_PEAKS = 2  
    
    def __init__(self, verbose: bool = True):
        """Initialize the UCI loader.
        
        Args:
            verbose: If True, show progress bars and warnings
        """
        self.verbose = verbose
        self.window_patient_ids: List[int] = []  
        self.total_samples = 0

    def load_part(self, filename: str, max_windows: Optional[int] = None) -> Tuple[List[np.ndarray], np.ndarray]:
        """Load a .mat file and extract aligned ECG/PPG samples
        
        Args:
            filename: Path to .mat file (e.g., 'data/Part_1.mat')
            max_windows: Maximum number of windows to extract (for debugging)
        
        Returns:
            X: ndarray of shape (N, 2, 500)
                X[:, 0, :] = PPG
                X[:, 1, :] = ECG

            y: ndarray of shape (N, 500)
                ABP waveform targets
        """
        if self.verbose:
            print(f"Loading {filename}...")
            self.window_patient_ids = []  
            self.total_samples = 0

        # Try loading as HDF5 (MATLAB v7.3) first
        try:
            with h5py.File(filename, 'r') as f:
                # Find the main data variable
                data_key = None

                for key in f.keys():
                    if key in ['p', 'data', 'signals'] or not key.startswith('#'):
                        data_key = key
                        break
                
                if data_key is None:
                    raise ValueError(f"Could not find data in HDF5 file. Available keys: {list(f.keys())}")
                
                patient_records = f[data_key]
                
                if self.verbose:
                    print(f"Found {len(patient_records)} patient records (HDF5 format)")
                
                return self._process_hdf5_data(patient_records, max_windows)
        
        except (OSError, KeyError):
            # Fall back to scipy.io.loadmat for older MATLAB formats
            if self.verbose:
                print("Trying older MATLAB format...")
            
            try:
                data = loadmat(filename)
            except Exception as e:
                raise IOError(f"Failed to load {filename}: {e}")
            
            # Find the main data variable
            data_key = None
            for key in ['p', 'data', 'signals']:
                if key in data:
                    data_key = key
                    break
            
            if data_key is None:
                available_keys = [k for k in data.keys() if not k.startswith('__')]
                raise ValueError(f"Could not find data in .mat file. Available keys: {available_keys}")
            
            patient_records = data[data_key]
            
            if self.verbose:
                print(f"Found {len(patient_records)} patient records")
            
            return self._process_legacy_data(patient_records, max_windows)
    
    def _process_hdf5_data(self, patient_records, max_windows: Optional[int] = None) -> Tuple[List[np.ndarray], np.ndarray]:
        """Process HDF5 (MATLAB v7.3) format data."""
        X_raw = []
        y = []
        
        total_windows = 0
        skipped_windows = 0
        
        # Process each patient record
        # Shape is (3000, 1) so we iterate through dimension 0
        n_patients = patient_records.shape[0]
        iterator = tqdm(range(n_patients), desc="Processing patients") if self.verbose else range(n_patients)
        
        for i in iterator:
            try:
                # Get reference to patient data
                patient_ref = patient_records[i, 0]
                
                # Dereference if needed
                if isinstance(patient_ref, h5py.Reference):
                    patient = patient_records.file[patient_ref]
                else:
                    patient = patient_ref
                
                # Extract signals 
                signals = np.array(patient)
                
                # Handle transposed data
                if signals.shape[0] > signals.shape[1]:
                    signals = signals.T
                
                if signals.shape[0] < 3:
                    continue
                
                ppg = signals[0, :].flatten()
                abp = signals[1, :].flatten()
                ecg = signals[2, :].flatten()
                
                # Generate windows from this patient
                patient_X, patient_y, skipped = self._window_signals(ppg,ecg,abp)
                
                X_raw.extend(patient_X)
                y.extend(patient_y)
                self.window_patient_ids.extend([i] * len(patient_X))
                
                total_windows += len(patient_X) + skipped
                skipped_windows += skipped
                
                # Check if we've reached max_windows
                if max_windows and len(X_raw) >= max_windows:
                    X_raw = np.array(X_raw[:max_windows], dtype=np.float32)
                    y = np.array(y[:max_windows], dtype=np.float32)

                    self.window_patient_ids = self.window_patient_ids[:max_windows]
                    self.total_samples = X_raw.shape[0]

                    return X_raw, y
                    
            except Exception as e:
                if self.verbose and np.random.random() < 0.01:  # Print 1% of errors
                    print(f"\nSkipping patient {i}: {str(e)[:80]}")
                continue
        
        if self.verbose:
            print(f"\nTotal windows extracted: {len(X_raw)}")
            print(f"Skipped windows (quality issues): {skipped_windows}")
            if total_windows > 0:
                print(f"Success rate: {len(X_raw)/total_windows*100:.1f}%")
        
        X_raw = np.array(X_raw, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        self.total_samples = X_raw.shape[0]

        return X_raw, y
    
    def _process_legacy_data(self, patient_records, max_windows: Optional[int] = None) -> Tuple[List[np.ndarray], np.ndarray]:
        """Process legacy MATLAB format data."""
        X_raw = []
        y = []
        
        total_windows = 0
        skipped_windows = 0
        
        # Process each patient record
        iterator = tqdm(patient_records, desc="Processing patients") if self.verbose else patient_records
        
        for i, patient in iterator:
            # Extract signals (shape: 3 x N)
            if patient.shape[0] < 3:
                continue
            
            ppg = patient[0, :].flatten()
            abp = patient[1, :].flatten()
            ecg = patient[2, :].flatten()
            
            # Generate windows from this patient
            patient_X, patient_y, skipped = self._window_signals(ppg, ecg, abp)
            
            X_raw.extend(patient_X)
            y.extend(patient_y)
            self.window_patient_ids.extend([i] * len(patient_X))   # ← add this; i from enumerate
            
            total_windows += len(patient_X) + skipped
            skipped_windows += skipped
            
            # Check if we've reached max_windows
            if max_windows and len(X_raw) >= max_windows:
                X_raw = X_raw[:max_windows]
                y = y[:max_windows]
                self.window_patient_ids = self.window_patient_ids[:max_windows]  # ← add this
                break
        
        if self.verbose:
            print(f"\nTotal windows extracted: {len(X_raw)}")
            print(f"Skipped windows (quality issues): {skipped_windows}")
            print(f"Success rate: {len(X_raw)/total_windows*100:.1f}%")
        
        X_raw = np.array(X_raw, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        self.total_samples = X_raw.shape[0]

        return X_raw, y

    def _align_signals(self,ppg_window: np.ndarray,ecg_window: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Align two physiological signals using cross-correlation."""

        ppg_norm = ppg_window - np.mean(ppg_window)
        ecg_norm = ecg_window - np.mean(ecg_window)

        correlation = signal.correlate(ppg_norm, ecg_norm, mode='full')
        lag = np.argmax(correlation) - (len(ecg_norm) - 1)

        # Shift second signal to align with first
        aligned_signal = np.roll(ecg_window, lag)

        return ppg_window, aligned_signal
    
    def _window_signals(self,
        ppg: np.ndarray,
        ecg: np.ndarray,
        abp: np.ndarray
    ) -> Tuple[List[np.ndarray], List[np.ndarray], int]:
        """Create aligned ECG+PPG windows."""

        input_windows = []
        abp_windows = []
        skipped = 0

        min_len = min(len(ppg), len(ecg), len(abp))

        ppg = ppg[:min_len]
        ecg = ecg[:min_len]
        abp = abp[:min_len]

        num_windows = min_len // self.WINDOW_SIZE

        for i in range(num_windows):
            start_idx = i * self.WINDOW_SIZE
            end_idx = start_idx + self.WINDOW_SIZE

            ppg_window = ppg[start_idx:end_idx]
            ecg_window = ecg[start_idx:end_idx]
            abp_window = abp[start_idx:end_idx]

            if (
                np.isnan(ppg_window).any()
                or np.isnan(ecg_window).any()
                or np.isnan(abp_window).any()
            ):
                skipped += 1
                continue

            if np.std(abp_window) < 1e-6:
                skipped += 1
                continue

            # Align PPG and ABP
            aligned_ppg, aligned_abp = self._align_signals(
                ppg_window,
                abp_window
            )
            aligned_ecg = ecg_window

            # Stack as 2-channel input
            # shape = (2, 500)
            multi_channel_input = np.stack(
                [aligned_ppg, aligned_ecg],
                axis=0
            )

            input_windows.append(multi_channel_input)
            abp_windows.append(aligned_abp)

        return input_windows, abp_windows, skipped

# # ============================================================
# # Test
# # ============================================================
# if __name__ == "__main__":

#     loader = UCILoader(verbose=True)

#     X, y = loader.load_part(
#         "data/Part_1.mat",
#         max_windows=5
#     )

#     print("\nTest results:")
#     print(f"X shape: {X.shape}")
#     print(f"y shape: {y.shape}")
#     print(f"Total samples: {loader.total_samples}")

#     print("\nChannel mapping:")
#     print("X[:, 0, :] -> PPG")
#     print("X[:, 1, :] -> ECG")

# # plot
# import matplotlib.pyplot as plt
# import numpy as np

# # Select one sample window
# idx = 3

# # Time axis
# t = np.linspace(0, 5, 500)

# # Extract signals
# ppg = X[idx, 0, :]
# ecg = X[idx, 1, :]
# abp = y[idx]

# # Create plots
# fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# # PPG
# axes[0].plot(t, ppg, lw=1.2)
# axes[0].set_title("PPG Signal")
# axes[0].set_ylabel("Amplitude")

# # ECG
# axes[1].plot(t, ecg, lw=1.2)
# axes[1].set_title("ECG Signal")
# axes[1].set_ylabel("Amplitude")

# # ABP
# axes[2].plot(t, abp, lw=1.2)
# axes[2].set_title("ABP Signal")
# axes[2].set_ylabel("mmHg")
# axes[2].set_xlabel("Time (s)")

# plt.tight_layout()
# plt.show()

# # save figure 
# fig.savefig("waveform_samples.png", dpi=150)