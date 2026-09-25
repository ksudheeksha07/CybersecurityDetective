import time
from pathlib import Path

import pandas as pd
import numpy as np

from detection_pipeline import DetectionPipeline


BASE_DIR = Path(__file__).resolve().parent.parent

TEST_FILE = BASE_DIR / "data" / "processed" / "test.csv"
OUTPUT_FILE = BASE_DIR / "experiments" / "detection_latency_benchmark.txt"

SAMPLE_SIZE = 100


def main():

    print("Loading test data...")

    df = pd.read_csv(
        TEST_FILE,
        nrows=SAMPLE_SIZE
    )

    df.columns = df.columns.str.strip()

    features = (
        df.drop(columns=["Label"])
        .to_dict(orient="records")
    )

    print(f"Flows benchmarked: {len(features)}")

    print("Loading detection pipeline...")

    pipeline = DetectionPipeline()

    # Warm-up
    pipeline.analyze(features[0])

    print("Running benchmark...")

    latencies = []

    for flow in features:

        start = time.perf_counter()

        pipeline.analyze(flow)

        end = time.perf_counter()

        latency_ms = (end - start) * 1000

        latencies.append(latency_ms)

    latencies = np.array(latencies)

    mean_latency = float(np.mean(latencies))
    median_latency = float(np.median(latencies))
    p95_latency = float(np.percentile(latencies, 95))
    p99_latency = float(np.percentile(latencies, 99))
    min_latency = float(np.min(latencies))
    max_latency = float(np.max(latencies))

    throughput = (
        1000 / mean_latency
        if mean_latency > 0
        else 0
    )

    print("\n" + "=" * 60)
    print("DETECTION LATENCY BENCHMARK")
    print("=" * 60)

    print(f"Flows tested:        {len(features)}")
    print(f"Mean latency:        {mean_latency:.3f} ms")
    print(f"Median latency:      {median_latency:.3f} ms")
    print(f"P95 latency:         {p95_latency:.3f} ms")
    print(f"P99 latency:         {p99_latency:.3f} ms")
    print(f"Minimum latency:     {min_latency:.3f} ms")
    print(f"Maximum latency:     {max_latency:.3f} ms")
    print(f"Approx throughput:   {throughput:.2f} flows/sec")

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "Detection Latency Benchmark\n"
        )

        file.write(
            "=" * 60 + "\n"
        )

        file.write(
            f"Flows tested: {len(features)}\n"
        )

        file.write(
            f"Mean latency: {mean_latency:.3f} ms\n"
        )

        file.write(
            f"Median latency: {median_latency:.3f} ms\n"
        )

        file.write(
            f"P95 latency: {p95_latency:.3f} ms\n"
        )

        file.write(
            f"P99 latency: {p99_latency:.3f} ms\n"
        )

        file.write(
            f"Minimum latency: {min_latency:.3f} ms\n"
        )

        file.write(
            f"Maximum latency: {max_latency:.3f} ms\n"
        )

        file.write(
            f"Approx throughput: {throughput:.2f} flows/sec\n"
        )

    print("\nBenchmark saved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()