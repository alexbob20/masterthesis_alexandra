import os
import torch
import torchaudio
import numpy as np
from scipy.spatial.distance import cosine
from pyannote.audio import Pipeline
from pydub import AudioSegment
from speechbrain.inference.speaker import SpeakerRecognition

# Load ECAPA2 model from SpeechBrain
spkrec = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec",
    run_opts={"device": "cuda"}
)

# Load pyannote VAD
vad = Pipeline.from_pretrained(
    "pyannote/voice-activity-detection",
    use_auth_token="INSERT YOUR HUGGING FACE TOKEN"
)

def detect_speech_segments(audio_path):
    vad_output = vad(audio_path)
    return [(segment.start, segment.end) for segment in vad_output.get_timeline()]

def extract_segment_embedding(audio_path, start, end, sample_rate=16000):
    waveform, sr = torchaudio.load(audio_path)
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    start_sample = int(start * sample_rate)
    end_sample = int(end * sample_rate)
    segment = waveform[:, start_sample:end_sample]
    if segment.shape[0] > 1:
        segment = segment.mean(dim=0, keepdim=True)
    if segment.shape[1] < sample_rate:
        segment = torch.nn.functional.pad(segment, (0, sample_rate - segment.shape[1]))
    segment = segment.squeeze(0)
    emb = spkrec.encode_batch(segment.unsqueeze(0).to("cuda"))
    return emb.squeeze()

def normalize_embedding(embedding):
    embedding = embedding.detach().cpu().numpy()
    norm = np.linalg.norm(embedding)
    return embedding / norm if norm != 0 else embedding

def assign_speakers_normalized(embeddings, threshold=0.6):
    ref = normalize_embedding(embeddings[0]['embedding'])
    labeled = []
    for seg in embeddings:
        emb = normalize_embedding(seg['embedding'])
        sim = 1 - cosine(ref, emb)
        label = 'interviewer' if sim > threshold else 'patient'
        labeled.append({**seg, "label": label})
    return labeled

def average_embedding(segments):
    all_embs = [normalize_embedding(seg['embedding']) for seg in segments]
    mean_emb = np.mean(all_embs, axis=0)
    norm = np.linalg.norm(mean_emb)
    return mean_emb / norm if norm > 0 else mean_emb

def filter_adolescent_segments(segments, threshold=0.6):
    if not segments:
        return []
    ref = average_embedding(segments)
    filtered = []
    for seg in segments:
        emb = normalize_embedding(seg['embedding'])
        sim = 1 - cosine(ref, emb)
        if sim >= threshold:
            filtered.append(seg)
    return filtered

def merge_segments(segments, speaker_label="patient", max_gap=20.0):
    merged = []
    current = None
    for seg in segments:
        if seg["label"] != speaker_label:
            continue
        if current is None:
            current = {"start": seg["start"], "end": seg["end"]}
        elif seg["start"] - current["end"] <= max_gap:
            current["end"] = seg["end"]
        else:
            merged.append(current)
            current = {"start": seg["start"], "end": seg["end"]}
    if current:
        merged.append(current)
    return merged

def process_file(fname):
    input_path = os.path.join("T1_denoised", fname)

    # Extract ID from filename
    id_code = fname.replace("T1_", "").replace("_denoised.wav", "")
    output_filename = f"T1_separated_{id_code}_adolescent.wav"
    output_dir = "trial"
    output_path = os.path.join(output_dir, output_filename)

    os.makedirs(output_dir, exist_ok=True)

    print(f"▶️ Processing {fname} → {output_filename}")

    try:
        segments = detect_speech_segments(input_path)
        embeddings = [
            {"start": s, "end": e, "embedding": extract_segment_embedding(input_path, s, e)}
            for s, e in segments
        ]

        labeled = assign_speakers_normalized(embeddings, threshold=0.6)
        adolescent_segments = [seg for seg in labeled if seg['label'] == 'patient']
        filtered_segments = filter_adolescent_segments(adolescent_segments, threshold=0.6)

        for seg in filtered_segments:
            seg['label'] = 'patient'

        merged = merge_segments(filtered_segments)
        if merged:
            audio = AudioSegment.from_wav(input_path)
            patient_audio = AudioSegment.empty()
            for seg in merged:
                start_ms = int(seg["start"] * 1000)
                end_ms = int(seg["end"] * 1000)
                patient_audio += audio[start_ms:end_ms]
            patient_audio.export(output_path, format="wav")
            print(f"Saved to {output_path}")
        else:
            print(f"⚠️ No consistent adolescent segments found for {fname}")

    except Exception as e:
        print(f"Error processing {fname}: {e}")

def main():
    with open("missing_T1_separation.txt") as f:
        file_list = [line.strip() for line in f if line.strip()]
    for fname in file_list:
        process_file(fname)

if __name__ == "__main__":
    main()
