"""Extension of the classroom four-room vacuum agent: noisy dirt sensor.

The core classroom setup is kept: four rooms, 50 steps per run, 100 runs,
three policies (random / reflex / model), and cleanliness-based score.
Extension: the dirt sensor sometimes reports the opposite of reality.
The model agent confirms each percept using three independent readings.

Run: python src/vacuum_noisy_sensor.py
"""

import matplotlib.pyplot as plt
import numpy as np

N_ROOMS = 4
STEPS = 50
RUNS = 100
NOISE_LEVELS = (0.0, 0.2)
AGENTS = ("random", "reflex", "model")


def simulate(agent: str, seed: int, noise_probability: float = 0.0) -> int:
    """Return the total number of clean rooms accumulated over all steps."""
    if not 0.0 <= noise_probability <= 1.0:
        raise ValueError("noise_probability must be between zero and one")
    if agent not in AGENTS:
        raise ValueError(f"Unknown agent: {agent}")

    # Independent random streams keep the initial dirt layout identical
    # across policies and noise levels for each seed.
    environment_rng = np.random.default_rng(seed)
    sensor_rng = np.random.default_rng(seed + 10_000)
    action_rng = np.random.default_rng(seed + 20_000)

    dirt = environment_rng.random(N_ROOMS) < 0.5
    known_clean = np.zeros(N_ROOMS, dtype=bool)
    pos, direction, score = 0, 1, 0

    for _ in range(STEPS):
        # Class challenge: a sensor may flip dirty <-> clean.
        # The model agent uses three readings (majority vote) to reduce errors.
        readings = 3 if agent == "model" else 1
        sensed_dirty = []
        for _ in range(readings):
            flipped = sensor_rng.random() < noise_probability
            sensed_dirty.append(bool(dirt[pos]) != bool(flipped))
        observed_dirty = sum(sensed_dirty) > readings // 2

        if agent == "random":
            action = action_rng.choice(["Left", "Right", "Suck"])
        else:
            known_clean[pos] = not observed_dirty
            if observed_dirty:
                action = "Suck"
            elif agent == "reflex":
                action = action_rng.choice(["Left", "Right"])
            elif known_clean.all():
                action = "NoOp"
            else:
                if pos == N_ROOMS - 1:
                    direction = -1
                if pos == 0:
                    direction = 1
                action = "Right" if direction == 1 else "Left"

        # Same movement, cleaning and score definition as the class exercise.
        if action == "Suck":
            dirt[pos] = False
        elif action == "Right":
            pos = min(pos + 1, N_ROOMS - 1)
        elif action == "Left":
            pos = max(pos - 1, 0)

        score += int((~dirt).sum())

    return score


def run_experiment() -> dict[float, dict[str, tuple[float, float]]]:
    """100 repeated runs per policy and noise setting, as in the class."""
    results = {}
    for noise in NOISE_LEVELS:
        results[noise] = {}
        for agent in AGENTS:
            scores = [simulate(agent, seed, noise) for seed in range(RUNS)]
            results[noise][agent] = (float(np.mean(scores)), float(np.std(scores)))
            print(
                f"sensor noise={noise:.0%} | {agent:6} "
                f"| mean={np.mean(scores):6.1f} | std={np.std(scores):5.1f}"
            )
    return results


def plot_results(results: dict) -> None:
    x = np.arange(len(AGENTS))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    for idx, noise in enumerate(NOISE_LEVELS):
        means = [results[noise][name][0] for name in AGENTS]
        ax.bar(x + (idx - 0.5) * width, means, width=width,
               label=f"Sensor noise {noise:.0%}")
    ax.set_xticks(x, AGENTS)
    ax.set_ylabel("Mean cleanliness score over 100 runs")
    ax.set_title("Classroom vacuum: accurate vs noisy sensor")
    ax.legend()
    fig.tight_layout()
    plt.show()


def main() -> None:
    results = run_experiment()
    plot_results(results)


if __name__ == "__main__":
    main()
