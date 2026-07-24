import sys
import argparse
from evolution import run_experiment

# 5 Named Wind Sweep Conditions
WIND_CONDITIONS = {
    "calm": 0.00,
    "mild": 0.25,
    "moderate": 0.50,
    "high": 0.75,
    "severe": 1.00
}

CONDITION_KEYS = list(WIND_CONDITIONS.keys())

# Seed Batches
SEED_BATCHES = {
    1: [201, 202, 203, 204, 205],
    2: [206, 207, 208, 209, 210],
    3: [201, 202, 203, 204, 205, 206, 207, 208, 209, 210]
}


def parse_args():
    parser = argparse.ArgumentParser(description="5-Machine Adaptive Flocking Experiment Runner")
    parser.add_argument(
        "--condition",
        type=str,
        choices=CONDITION_KEYS,
        help="Wind condition for this machine (calm, mild, moderate, high, severe)"
    )
    parser.add_argument(
        "--batch",
        type=int,
        choices=[1, 2, 3],
        help="Seed batch: 1 (Seeds 201..205), 2 (Seeds 206..210), 3 (All 10 seeds)"
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=100,
        help="Number of generations (default: 100)"
    )
    parser.add_argument(
        "--pop",
        type=int,
        default=50,
        help="Population size (default: 50)"
    )
    return parser.parse_args()


def select_condition_interactive():
    print("\n=======================================================")
    print(" ADAPTIVE FLOCKING: 5-MACHINE EXPERIMENT RUNNER       ")
    print("=======================================================")
    print("Select Wind Condition assigned to THIS machine:\n")

    for idx, cond_name in enumerate(CONDITION_KEYS, start=1):
        strength = WIND_CONDITIONS[cond_name]
        print(f"  [{idx}] {cond_name.upper():<10} (Wind Strength = {strength:.2f})")

    print("\n-------------------------------------------------------")
    while True:
        try:
            choice = input(f"Enter choice [1-{len(CONDITION_KEYS)}] or condition name: ").strip().lower()
            if not choice:
                continue
            if choice in WIND_CONDITIONS:
                return choice
            if choice.isdigit() and 1 <= int(choice) <= len(CONDITION_KEYS):
                return CONDITION_KEYS[int(choice) - 1]
            print(f"Please enter a choice between 1 and {len(CONDITION_KEYS)} or a valid condition name.")
        except (ValueError, KeyboardInterrupt, EOFError):
            print("\nExiting.")
            sys.exit(0)


def select_batch_interactive():
    print("\n-------------------------------------------------------")
    print("Select EA Seed Batch to run on this machine:\n")
    print("  [1] Batch 1: Seeds 201..205 (5 Runs)")
    print("  [2] Batch 2: Seeds 206..210 (5 Runs)")
    print("  [3] All 10 Seeds: 201..210 (10 Runs)")
    print("-------------------------------------------------------")

    while True:
        try:
            choice = input("Enter choice [1-3] (default: 1): ").strip()
            if not choice:
                return 1
            if choice in ["1", "2", "3"]:
                return int(choice)
            print("Please enter 1, 2, or 3.")
        except (ValueError, KeyboardInterrupt, EOFError):
            print("\nExiting.")
            sys.exit(0)


def main():
    args = parse_args()

    condition = args.condition if args.condition else select_condition_interactive()
    batch_num = args.batch if args.batch else select_batch_interactive()

    wind_strength = WIND_CONDITIONS[condition]
    seeds_to_run = SEED_BATCHES[batch_num]

    print("\n=======================================================")
    print(f" LAUNCHING EXPERIMENT FOR CONDITION: {condition.upper()}")
    print(f" Wind Strength  : {wind_strength:.2f}")
    print(f" EA Seed Batch  : Batch {batch_num} ({len(seeds_to_run)} seeds: {seeds_to_run})")
    print(f" Generations    : {args.generations} | Population: {args.pop} | Eval Seeds: 10")
    print(f" Output Folder  : results/condition_{condition}/")
    print("=======================================================\n")

    for i, seed in enumerate(seeds_to_run, start=1):
        print(f"\n>>> [{i}/{len(seeds_to_run)}] Running {condition.upper()} with EA Seed {seed}...")
        run_experiment(
            generations=args.generations,
            pop_size=args.pop,
            wind_strength=wind_strength,
            experiment_name=condition,
            ea_seed=seed
        )

    print("\n=======================================================")
    print(f" ALL RUNS COMPLETED FOR CONDITION: {condition.upper()}!")
    print(f" Output files saved to results/condition_{condition}/")
    print("=======================================================")


if __name__ == "__main__":
    main()
