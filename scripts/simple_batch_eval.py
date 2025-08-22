#!/usr/bin/env python3
"""
Simple Working Batch Evaluation Script with Automatic CSV Generation
"""

import os
import json
import subprocess
import argparse
import pandas as pd
import time
import random
from datetime import datetime
from pathlib import Path

def run_evaluation(model, dataset, noise_type, max_samples, output_dir, dataset_args=None):
    """Run a single evaluation - optimized for paid tier."""
    
    if dataset_args:
        print(f"Running: {model} + dataset_args={dataset_args}")
    else:
        print(f"Running: {model} + {noise_type}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = model.replace("/", "_").replace("-", "_")
    if dataset_args:
        safe_args = dataset_args.replace(":", "_").replace(",", "_").replace("=", "_")
        output_file = output_path / f"{safe_model}_multi_{timestamp}.json"
        cmd = [
            "karma", "eval",
            "--model", model,
            "--datasets", dataset,
            "--dataset-args", dataset_args,
            "--max-samples", str(max_samples),
            "--output", str(output_file)
        ]
    else:
        safe_noise_type = noise_type.replace(":", "_")  # Replace colon with underscore for filename
        output_file = output_path / f"{safe_model}_{safe_noise_type}_{timestamp}.json"
        cmd = [
            "karma", "eval",
            "--model", model,
            "--datasets", dataset,
            "--dataset-args", f"{dataset}:noise_type={noise_type},language=hi",
            "--max-samples", str(max_samples),
            "--output", str(output_file)
        ]
    
    # Run command with simple retry logic for 503/UNAVAILABLE errors
    max_retries = 3
    delay = 5  # seconds
    for attempt in range(1, max_retries + 1):
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd="/root/projects/KARMA-OpenMedEvalKit")
            if result.returncode == 0:
                print(f"Success: {model} + {noise_type}")
                return True, str(output_file)
            else:
                err = result.stderr or ""
                print(f"Failed: {model} + {noise_type}")
                print(f"Error: {err}")
                if ("503" in err or "UNAVAILABLE" in err) and attempt < max_retries:
                    print(f"[Retry] Attempt {attempt} failed with 503/UNAVAILABLE, retrying in {delay}s...")
                    time.sleep(delay)
                    continue
                return False, None
        except Exception as e:
            print(f"Exception: {e}")
            if ("503" in str(e) or "UNAVAILABLE" in str(e)) and attempt < max_retries:
                print(f"[Retry] Exception on attempt {attempt} with 503/UNAVAILABLE, retrying in {delay}s...")
                time.sleep(delay)
                continue
            return False, None

def main():
    parser = argparse.ArgumentParser(description="Simple Batch Evaluation with Auto CSV Generation")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--noise-types", nargs="+", default=["clean", "gaussian"])
    parser.add_argument("--max-samples", type=int, default=2)
    parser.add_argument("--output-dir", default="batch_results")
    parser.add_argument("--dataset-args", type=str, default=None, help="Full dataset args string to pass to karma eval (overrides --noise-types)")

    args = parser.parse_args()

    print(f"Models: {args.models}")
    if args.dataset_args:
        print(f"Dataset Args: {args.dataset_args}")
    else:
        print(f"Noise Types: {args.noise_types}")

    success_count = 0
    total_count = 0
    json_files = []

    if args.dataset_args:
        # Single run per model with all noise types in dataset_args
        for model in args.models:
            total_count += 1
            success, json_file = run_evaluation(model, args.dataset, noise_type=None, max_samples=args.max_samples, output_dir=args.output_dir, dataset_args=args.dataset_args)
            if success and json_file:
                success_count += 1
                json_files.append(json_file)
    else:
        # Old behavior: loop over noise types
        for model in args.models:
            for noise_type in args.noise_types:
                total_count += 1
                success, json_file = run_evaluation(model, args.dataset, noise_type, args.max_samples, args.output_dir)
                if success and json_file:
                    success_count += 1
                    json_files.append(json_file)

    print(f"JSON Results saved in: {args.output_dir}")

if __name__ == "__main__":
    main()
