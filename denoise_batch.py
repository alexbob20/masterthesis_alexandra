import os
import soundfile as sf
import noisereduce as nr
import re
import sys

input_folder = "T2_audios"
output_folder = "T2_denoised"
pattern = re.compile(r"IB-(\d{5})")

# Get audio filenames from input file
with open(sys.argv[1]) as f:
    audio_files = [line.strip() for line in f if line.strip()]

total_files = len(audio_files)

for i, filename in enumerate(audio_files, 1):
    match = pattern.search(filename)
    if not match:
        print(f"skipped{filename}")
        continue

    file_id = match.group(1)
    input_path = os.path.join(input_folder, filename)

    try:
        audio, rate = sf.read(input_path)
    except:
        print(f"Failed to read: {filename}")
        continue

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    reduced_noise = nr.reduce_noise(y=audio, sr=rate, n_std_thresh_stationary=0.2, stationary=True)

    output_filename = f"T2_{file_id}_denoised.wav"
    output_path = os.path.join(output_folder, output_filename)
    sf.write(output_path, reduced_noise, rate)

    print(f"[{i}/{total_files}] processed {filename} -> {output_filename}")
