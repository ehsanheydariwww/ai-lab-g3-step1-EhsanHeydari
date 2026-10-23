"""Music genre classification with the classroom 'noisy sensor' challenge.

Three decision styles inspired by the classroom agents:
random, simple reflex (RMS), and learned model (Random Forest).
Train only on clean audio; compare their predictions on the SAME test songs
in clean form and after adding Gaussian noise to the audio waveform.

Run: python src/genre_agent.py --data-dir data --snr-db 10
When the data folder is missing, a clearly labelled synthetic demo is used.
For real datasets: data/<genre>/<song.wav>.
"""

from __future__ import annotations

import argparse
import math
import tempfile
import wave
from collections import Counter
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

SAMPLE_RATE = 22050
DURATION = 15
SEED = 42
EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg"}



def make_synthetic_demo(folder: Path, n_per_class: int = 12) -> None:
    """Create two different synthetic sound patterns, not real musical genres.

    This is only a smoke test of the noise-sensor experiment and code flow.
    Files are generated in a temporary folder and automatically deleted.
    """
    rng = np.random.default_rng(SEED)
    sr = SAMPLE_RATE
    seconds = 2.0
    t = np.arange(int(seconds * sr), dtype=np.float64) / sr

    for sound_type in ("synthetic_pulse", "synthetic_drone"):
        subfolder = folder / sound_type
        subfolder.mkdir(parents=True, exist_ok=True)
        for k in range(n_per_class):
            freq = (180 if sound_type == "synthetic_pulse" else 510) + rng.uniform(-30, 30)
            phase = rng.uniform(0, 2 * np.pi)
            fundamental = np.sin(2 * np.pi * freq * t + phase)
            overtone = 0.25 * np.sin(2 * np.pi * (2 * freq) * t + phase)
            if sound_type == "synthetic_pulse":
                envelope = 0.2 + 0.8 * (np.sin(2 * np.pi * (3 + rng.uniform(-0.2, 0.2)) * t) > 0)
            else:
                envelope = 0.65 + 0.1 * np.sin(2 * np.pi * 0.5 * t)
            y = (0.35 + rng.uniform(-0.05, 0.05)) * envelope * (fundamental + overtone)
            y += rng.normal(0, 0.008, size=y.shape)
            pcm = np.int16(np.clip(y, -1, 1) * 32767)
            with wave.open(str(subfolder / f"sample_{k + 1:02d}.wav"), "wb") as out:
                out.setnchannels(1)
                out.setsampwidth(2)
                out.setframerate(sr)
                out.writeframes(pcm.tobytes())


def run_experiment(data_dir: Path, snr_db: float) -> None:
    x, labels, paths = load_dataset(data_dir)
    results = evaluate_noise(x, labels, paths, snr_db=snr_db)
    show_chart(results, snr_db=snr_db)
    print("Small datasets are not reliable benchmarks.")

def features_from_audio(path: Path, snr_db: float | None = None,
                        seed: int = SEED) -> np.ndarray:
    """Extract MFCC/chroma/RMS from clean audio or noisy audio waveform.

    If snr_db is set, noise is generated relative to this song's RMS level.
    Smaller SNR means stronger noise. Noise is added BEFORE feature extraction.
    """
    y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True, duration=DURATION)
    if len(y) < sr:
        raise ValueError("Audio must be at least one second long")

    if snr_db is not None:
        signal_rms = float(np.sqrt(np.mean(np.square(y.astype(float)))))
        if signal_rms == 0:
            raise ValueError("Cannot set SNR for completely silent audio")
        noise_rms = signal_rms * 10 ** (-snr_db / 20.0)
        rng = np.random.default_rng(seed)
        y = y + rng.normal(0, noise_rms, size=len(y)).astype(y.dtype)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)
    vector = np.concatenate([
        mfcc.mean(axis=1),
        mfcc.std(axis=1),
        chroma.mean(axis=1),
        np.array([rms.mean()]),
    ]).astype(float)
    if not np.isfinite(vector).all():
        raise ValueError("Invalid extracted features")
    return vector


def load_dataset(data_dir: Path) -> tuple[np.ndarray, np.ndarray, list[Path]]:
    """Read labelled tracks: a subfolder name is the ground-truth genre."""
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data not found: {data_dir}; create data/<genre>/<song.wav>")

    rows, labels, files = [], [], []
    for genre_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        for path in sorted(genre_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
                continue
            try:
                rows.append(features_from_audio(path))
                labels.append(genre_dir.name)
                files.append(path)
            except Exception as exc:
                print(f"Skipped {path}: {exc}")

    counts = Counter(labels)
    if len(counts) < 2 or min(counts.values()) < 4:
        raise ValueError(f"At least two genres and four readable files per genre: {dict(counts)}")
    print("Files per genre:", dict(counts))
    return np.vstack(rows), np.asarray(labels), files


def evaluate_noise(x: np.ndarray, labels: np.ndarray, paths: list[Path],
                   snr_db: float) -> dict[str, dict[str, dict[str, float]]]:
    """Same held-out songs, same trained model, clean vs noisy observation."""
    genres = sorted(set(labels))
    n_test = max(len(genres), math.ceil(len(labels) * 0.25))
    train_idx, test_idx = train_test_split(
        np.arange(len(labels)), test_size=n_test,
        stratify=labels, random_state=SEED
    )
    x_train, y_train = x[train_idx], labels[train_idx]
    x_clean, y_test = x[test_idx], labels[test_idx]

    # Classroom challenge transferred to music: corrupt sensor input itself.
    # Training data and ground truth do not change; only test waveforms change.
    x_noisy = np.vstack([
        features_from_audio(paths[int(i)], snr_db=snr_db, seed=SEED + int(i))
        for i in test_idx
    ])

    genre_rms_means = {
        genre: float(np.mean(x_train[y_train == genre, -1])) for genre in genres
    }
    model = RandomForestClassifier(n_estimators=100, random_state=SEED)
    model.fit(x_train, y_train)

    # Same random guesses for the two scenarios: noise cannot affect a blind agent.
    rng = np.random.default_rng(SEED)
    random_predictions = rng.choice(genres, size=len(y_test))

    results = {}
    for condition, x_test in (("Clean", x_clean), ("Noisy", x_noisy)):
        simple_predictions = np.asarray([
            min(genres, key=lambda genre: abs(row[-1] - genre_rms_means[genre]))
            for row in x_test
        ])
        predictions = {
            "Random": random_predictions,
            "Reflex (RMS)": simple_predictions,
            "Model (forest)": model.predict(x_test),
        }
        results[condition] = {}
        for agent, guesses in predictions.items():
            scores = {
                "accuracy": float(accuracy_score(y_test, guesses)),
                "macro_f1": float(f1_score(y_test, guesses, average="macro", zero_division=0)),
            }
            results[condition][agent] = scores
            print(f"{condition:5} | {agent:14} | Accuracy: {scores['accuracy']:.3f} "
                  f"| Macro-F1: {scores['macro_f1']:.3f}")
    return results


def show_chart(results: dict, snr_db: float) -> None:
    agents = list(results["Clean"])
    x = np.arange(len(agents))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 4))
    for offset, condition in enumerate(("Clean", "Noisy")):
        values = [results[condition][agent]["macro_f1"] for agent in agents]
        label = "Clean audio" if condition == "Clean" else f"Noisy audio ({snr_db:g} dB SNR)"
        ax.bar(x + (offset - 0.5) * width, values, width, label=label)
    ax.set_xticks(x, agents)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Macro-F1 on the same held-out songs")
    ax.set_title("Music genre recognition: robustness to sensor noise")
    ax.legend()
    fig.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Music genre classification with noisy audio")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--snr-db", type=float, default=10.0,
                        help="Signal-to-noise ratio in dB; lower = noisier (default: 10)")
    args = parser.parse_args()
    if not np.isfinite(args.snr_db):
        parser.error("--snr-db must be finite")
    if args.data_dir.exists():
        run_experiment(args.data_dir, args.snr_db)
    elif args.data_dir == Path("data"):
        print("Data folder not found. Running SYNTHETIC AUDIO DEMO instead.")
        print("These are generated sound patterns, NOT music recordings or real genres.")
        with tempfile.TemporaryDirectory(prefix="music_step1_demo_") as temp:
            demo_dir = Path(temp)
            make_synthetic_demo(demo_dir)
            run_experiment(demo_dir, args.snr_db)
        print("Synthetic results must NOT be reported as accuracy on real music.")
    else:
        raise FileNotFoundError(
            f"Data folder not found: {args.data_dir}. "
            "Use the default 'data' path to run a synthetic demo."
        )


if __name__ == "__main__":
    main()
