import os
import csv
import librosa
import matplotlib.pyplot as plt
import numpy as np

RAW_DIR = '../data/training'
CSV_OUTPUT_FILE = 'data/training/training.csv'


def extract_chroma(input_path, noise_gate=0.01):
    y, sr = librosa.load(input_path, sr=44100, mono=True)

    y[np.abs(y) < noise_gate] = 0
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    return chroma_mean


def save_chroma_csv(rows):
    os.makedirs(os.path.dirname(CSV_OUTPUT_FILE), exist_ok=True)
    with open(CSV_OUTPUT_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['label', 'C', 'C#', 'D', 'D#', 'E',
                        'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'])
        writer.writerows(rows)


# Save just the labels to a separate CSV file
def save_labels_csv(rows):
    label_output_file = os.path.join(
        os.path.dirname(CSV_OUTPUT_FILE), 'labels.csv')
    os.makedirs(os.path.dirname(label_output_file), exist_ok=True)
    labels = sorted(set(row[0] for row in rows))
    with open(label_output_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['label'])
        for label in labels:
            writer.writerow([label])


def process_audio_to_chroma():
    csv_rows = []
    for label in os.listdir(RAW_DIR):
        label_dir = os.path.join(RAW_DIR, label)
        if not os.path.isdir(label_dir):
            continue

        for file_name in os.listdir(label_dir):
            if file_name.lower().endswith('.wav'):
                input_file = os.path.join(label_dir, file_name)
                chroma = extract_chroma(input_file, noise_gate=0.01)
                if chroma is not None:
                    csv_rows.append([label] + chroma.tolist())

    save_chroma_csv(csv_rows)

    save_labels_csv(csv_rows)


if __name__ == '__main__':
    process_audio_to_chroma()
