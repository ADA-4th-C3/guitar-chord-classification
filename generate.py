import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

RAW_DIR = 'raw/training'
OUTPUT_DIR = 'data/training'


def extract_chroma(input_path, noise_gate=0.01):
    y, sr = librosa.load(input_path, sr=44100, mono=True)

    y[np.abs(y) < noise_gate] = 0
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    return chroma


def save_chroma_image(chroma, output_path):
    plt.figure(figsize=(4, 4))
    librosa.display.specshow(chroma, y_axis='chroma',
                             x_axis='time', cmap='gray_r')
    plt.axis('off')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def extract_and_save_chroma(input_path, output_path, threshold=0.01):
    chroma = extract_chroma(input_path, threshold)
    if chroma is None:
        return
    save_chroma_image(chroma, output_path)


def process_audio_to_chroma():
    for label in os.listdir(RAW_DIR):
        label_dir = os.path.join(RAW_DIR, label)
        if not os.path.isdir(label_dir):
            continue

        for file_name in os.listdir(label_dir):
            if file_name.lower().endswith('.wav'):
                input_file = os.path.join(label_dir, file_name)
                output_label_dir = os.path.join(OUTPUT_DIR, label)
                output_file = os.path.join(
                    output_label_dir, file_name.replace('.wav', '.png'))

                extract_and_save_chroma(
                    input_file, output_file, threshold=0.01)
                print(f"Saved chroma image: {output_file}")


if __name__ == '__main__':
    process_audio_to_chroma()
