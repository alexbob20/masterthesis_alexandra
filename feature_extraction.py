import os
import glob
import re
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from disvoice.phonation import Phonation
from disvoice.articulation import Articulation
from disvoice.prosody import Prosody
from pyAudioAnalysis import ShortTermFeatures as stf
from pyAudioAnalysis import audioBasicIO
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# === Config ===
input_folder = "baseline_separated"
output_folder = "baseline_features"
os.makedirs(output_folder, exist_ok=True)
output_path = os.path.join(output_folder, "baseline_all_features.csv")

# === Init DisVoice modules ===
phonation = Phonation()
articulation = Articulation()
prosody = Prosody()

# === Helper to extract ID ===
def extract_id(filename):
    match = re.search(r"separated_IB-(\d+)_adolescent\.wav", filename)
    return match.group(1) if match else None

# === PyAudioAnalysis summary extractor ===
def extract_summary_features(audio_file):
    Fs, x = audioBasicIO.read_audio_file(audio_file)
    if len(x) == 0:
        print(f"{audio_file} is empty")
        return {}

    features, feature_names = stf.feature_extraction(x, Fs, 0.05 * Fs, 0.025 * Fs)

    summary = {}
    for i, name in enumerate(feature_names):
        vals = features[i, :]
        summary[f"baseline_paa_{name}_mean"] = np.mean(vals)
        summary[f"baseline_paa_{name}_std"] = np.std(vals)
        summary[f"baseline_paa_{name}_min"] = np.min(vals)
        summary[f"baseline_paa_{name}_max"] = np.max(vals)
        summary[f"baseline_paa_{name}_skew"] = skew(vals)
        summary[f"baseline_paa_{name}_kurtosis"] = kurtosis(vals)
    return summary

# === Feature extraction loop ===
wav_files = sorted(glob.glob(os.path.join(input_folder, "*.wav")))
print(f"Found {len(wav_files)} audio files.")

batch_rows = []
header_written = False

for i, filepath in enumerate(wav_files):
    filename = os.path.basename(filepath)
    audio_id = extract_id(filename)
    print(f"Processing file {i+1}/{len(wav_files)}: {filename}")
    
    if not audio_id:
        print(f"Skipping malformed filename: {filename}")
        continue 

    row = {"id": audio_id}

    try:
        # DisVoice features
        row.update({f"baseline_{k}": v for k, v in phonation.extract_features_file(filepath, static=True, plots=False, fmt="dataframe").iloc[0].items()})
        row.update({f"baseline_{k}": v for k, v in articulation.extract_features_file(filepath, static=True, plots=False, fmt="dataframe").iloc[0].items()})
        row.update({f"baseline_{k}": v for k, v in prosody.extract_features_file(filepath, static=True, plots=False, fmt="dataframe").iloc[0].items()})

        # PyAudioAnalysis
        paa_summary = extract_summary_features(filepath)
        row.update(paa_summary)

        batch_rows.append(row)

    except Exception as e:
        print(f"Error processing {filename}: {e}")
        continue

    # Save every 25 files
    if len(batch_rows) == 25:
        df = pd.DataFrame(batch_rows)
        df.to_csv(output_path, mode='a', index=False, header=not header_written)
        header_written = True
        batch_rows = []
        print(f"Saved batch up to file {i+1}")

# Save remaining files
if batch_rows:
    df = pd.DataFrame(batch_rows)
    df.to_csv(output_path, mode='a', index=False, header=not header_written)
    print(f"Saved final batch of {len(batch_rows)} files")

print(f"All baseline features saved to {output_path}")
