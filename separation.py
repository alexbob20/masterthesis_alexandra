import os
import re
from collections import defaultdict
from pydub import AudioSegment
from pyannote.audio import Pipeline
import time
import sys

# Get list of files to process
with open(sys.argv[1], 'r') as f:
    audio_files = [line.strip() for line in f if line.strip()]

print(f"Loaded {len(audio_files)} filenames from {sys.argv[1]}")

# === Configuration ===
input_folder = "T2_denoised"
output_folder_adolescent = "T2_separated"
os.makedirs(output_folder_adolescent, exist_ok=True)

# Regular expression pattern for extracting file ID
pattern = re.compile(r"T2_(\d{5})_denoised\.wav", re.IGNORECASE)

# Load Hugging Face model

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=True
).to(device)

pipeline = pipeline.instantiate({
    "clustering": {
        "method": "centroid",
        "min_cluster_size": 12,
        "threshold": 0.5  
    },
    "segmentation": {
        "min_duration_off": 0.0
    }
})

# Get all audio files
existing_ids = set()
total_files = len(audio_files)

# === Main loop ===
for i, filename in enumerate(audio_files, 1):
    start_time = time.time()  # ⏱ Start timer

    match = pattern.search(filename)
    if not match:
        continue
    file_id = match.group(1)

    if file_id in existing_ids:
        print("skipped", f"{file_id}")
        continue

    input_path = os.path.join(input_folder, filename)
    print(f"Processing {filename} ({i}/{total_files})")
    try:
        print(f"Processing: {filename}")
        diarization = pipeline(input_path)
    except Exception as e:
        print(f"Diarization failed for {filename}: {e}")
        continue

    audio = AudioSegment.from_wav(input_path)
    all_segments = []
    speaker_label_map = {}
    first_speaker_set = False

    for speech_turn, track, speaker in diarization.itertracks(yield_label=True):
        start_ms = int(speech_turn.start * 1000)
        end_ms = int(speech_turn.end * 1000)

        if not first_speaker_set:
            speaker_label_map[speaker] = "interviewer"
            first_speaker = speaker
            first_speaker_set = True
        elif speaker not in speaker_label_map:
            speaker_label_map[speaker] = "adolescent" if speaker != first_speaker else "interviewer"

        all_segments.append((start_ms, end_ms, speaker))

    timeline = defaultdict(set)
    for start, end, speaker in all_segments:
        for ms in range(start, end):
            timeline[ms].add(speaker)

    clean_segments = defaultdict(list)
    current_speaker = None
    current_start = None
    sorted_ms = sorted(timeline.keys())

    for j, ms in enumerate(sorted_ms):
        active_speakers = timeline[ms]

        if len(active_speakers) == 1:
            speaker = next(iter(active_speakers))
            if speaker != current_speaker:
                if current_speaker is not None and current_start is not None:
                    clean_segments[current_speaker].append((current_start, ms))
                current_start = ms
                current_speaker = speaker
        else:
            if current_speaker is not None and current_start is not None:
                clean_segments[current_speaker].append((current_start, ms))
                current_speaker = None
                current_start = None

    if current_speaker is not None and current_start is not None:
        clean_segments[current_speaker].append((current_start, sorted_ms[-1]))

    # Only export adolescent segments
    speaker_segments = defaultdict(lambda: AudioSegment.silent(duration=0))
    for speaker, segments in clean_segments.items():
        label = speaker_label_map.get(speaker, f"speaker_{speaker}")
        if label.lower() != "adolescent":
            continue

        for start, end in segments:
            speaker_segments[speaker] += audio[start:end]

        output_filename = os.path.join(output_folder_adolescent, f"T2_separated_{file_id}_{label}.wav")
        speaker_segments[speaker].export(output_filename, format="wav")
        print(f"Saved: {output_filename}")

    elapsed = time.time() - start_time
    print(f"{i} out of {total_files} processed {filename} in {elapsed:.2f} seconds")

