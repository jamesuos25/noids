import os
import random
import functools
import time
import multiprocessing as mp
# pyrefly: ignore [missing-import]
import numpy as np
from numpy.random import SeedSequence, default_rng
import pandas as pd
# pyrefly: ignore [missing-import]
from deap import base, creator, tools
from tqdm import tqdm

from boid import Boid
from environment import Environment
from fitness import compute_frame_fitness

# =====================================================================
# DEAP Setup & Type Creation
# =====================================================================
# Maximize a single-objective fitness value
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Simulation Parameters
NUM_BOIDS = 35
SIM_FRAMES = 700
WARMUP_FRAMES = 100
EVAL_FRAMES = SIM_FRAMES - WARMUP_FRAMES  # 600
PERCEPTION_RADIUS = 150.0
ARENA_BOUNDS = (800, 600)
EVAL_SEEDS = [101, 102, 103, 104, 105]  # Fixed seed suite for Common Random Numbers

# Genome Configuration: 43 Genes (42 Neural Parameters + 1 Sigma Gene)
NUM_GENES = 43
GENE_BOUND_LOW = -3.0
GENE_BOUND_HIGH = 3.0
SIGMA_GENE_MIN = 0.001
SIGMA_GENE_MAX = 0.500


def create_individual():
    """Generates a random 43-gene candidate solution."""
    # Genes 0-41: Initial neural weights/biases sampled [-1, 1]
    params = [random.uniform(-1.0, 1.0) for _ in range(42)]
    # Gene 42: Initial variance gene (sigma) sampled [0.01, 0.10]
    sigma_gene = random.uniform(0.01, 0.10)
    params.append(sigma_gene)
    return creator.Individual(params)


def clamp_individual(individual):
    """
    Enforces physical and mathematical boundary constraints on modified genes
    following crossover or mutation operators.
    """
    # Clamp neural parameters to [-3.0, 3.0]
    for i in range(42):
        individual[i] = max(-3.0, min(3.0, individual[i]))

    # Clamp sigma gene to strictly positive bounds [0.001, 0.500]
    individual[42] = max(SIGMA_GENE_MIN, min(SIGMA_GENE_MAX, individual[42]))


def evaluate_individual(individual, wind_strength=0.0):
    """
    Evaluates a candidate solution across the 5 fixed environmental seeds.
    Returns overall fitness (mean), seed_std, alignment, cohesion, and separation.
    """
    mean_params = individual[:42]
    sigma_gene = individual[42]

    seed_fitnesses = []
    seed_alignments = []
    seed_cohesions = []
    seed_separations = []

    for seed in EVAL_SEEDS:
        # Create statistically independent RNG streams via SeedSequence splitting
        ss = SeedSequence(seed)
        wind_ss, spawn_ss, pheno_base_ss = ss.spawn(3)
        wind_rng = default_rng(wind_ss)
        spawn_rng = default_rng(spawn_ss)

        # Initialize environment with isolated wind RNG stream
        env = Environment(
            width=ARENA_BOUNDS[0],
            height=ARENA_BOUNDS[1],
            wind_strength=wind_strength,
            perception_radius=PERCEPTION_RADIUS,
            rng=wind_rng
        )

        # Spawn flock using spawn_rng for initial positions/headings and isolated phenotype RNG per boid
        pheno_streams = pheno_base_ss.spawn(NUM_BOIDS)
        boids = []
        for i in range(NUM_BOIDS):
            phenotype_rng = default_rng(pheno_streams[i])
            boid = Boid(
                x=spawn_rng.uniform(100, ARENA_BOUNDS[0] - 100),
                y=spawn_rng.uniform(100, ARENA_BOUNDS[1] - 100),
                mean_params=mean_params,
                sigma=sigma_gene,
                rng=phenotype_rng,
                spawn_rng=spawn_rng
            )
            boids.append(boid)

        total_sim_fitness = 0.0
        total_alignment = 0.0
        total_cohesion = 0.0
        total_separation = 0.0
        
        # Step simulation for 700 frames (100 warm up, 600 evaluation)
        for _ in range(SIM_FRAMES):
            env.step(boids)
            if _ >= WARMUP_FRAMES:
                fit, align, coh, sep = compute_frame_fitness(boids, ARENA_BOUNDS)
                total_sim_fitness += fit
                total_alignment += align
                total_cohesion += coh
                total_separation += sep

        seed_fitnesses.append(total_sim_fitness / EVAL_FRAMES)
        seed_alignments.append(total_alignment / EVAL_FRAMES)
        seed_cohesions.append(total_cohesion / EVAL_FRAMES)
        seed_separations.append(total_separation / EVAL_FRAMES)

    # Overall fitness = average across all 5 seeds; seed_std = variance across seeds
    overall_fitness = float(np.mean(seed_fitnesses))
    seed_std = float(np.std(seed_fitnesses))
    alignment = float(np.mean(seed_alignments))
    cohesion = float(np.mean(seed_cohesions))
    separation = float(np.mean(seed_separations))
    return overall_fitness, seed_std, alignment, cohesion, separation


def save_checkpoint(history, population, experiment_name, ea_seed, batch_num=None):
    """
    Saves generational metrics, population snapshots, and champion genomes to disk
    after every generation to guarantee zero data loss if an execution is interrupted.
    """
    batch_str = f"_batch_{batch_num}" if batch_num is not None else ""
    results_dir = f"results/condition_{experiment_name}{batch_str}"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # 1. Save Generational Metrics CSV
    df = pd.DataFrame(history)
    df["ea_seed"] = ea_seed if ea_seed is not None else 0
    df["wind_condition"] = experiment_name
    cols = ["ea_seed", "wind_condition"] + [c for c in df.columns if c not in ["ea_seed", "wind_condition"]]
    df = df[cols]

    csv_path = os.path.join(results_dir, "generational_metrics.csv")
    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        if "ea_seed" in existing_df.columns:
            existing_df = existing_df[existing_df["ea_seed"] != df["ea_seed"].iloc[0]]
        combined_df = pd.concat([existing_df, df], ignore_index=True)
        combined_df.to_csv(csv_path, index=False)
    else:
        df.to_csv(csv_path, index=False)

    # 2. Save Final/Current Population Snapshot CSV
    pop_data = []
    for ind in population:
        pop_data.append({
            "ea_seed": ea_seed if ea_seed is not None else 0,
            "wind_condition": experiment_name,
            "fitness": ind.fitness.values[0] if ind.fitness.valid else 0.0,
            "sigma": ind[42],
            "seed_std": getattr(ind, 'seed_std', 0.0),
            "alignment": getattr(ind, 'alignment', 0.0),
            "cohesion": getattr(ind, 'cohesion', 0.0),
            "separation": getattr(ind, 'separation', 0.0)
        })
    pop_df = pd.DataFrame(pop_data)
    pop_csv_path = os.path.join(results_dir, "final_populations.csv")
    if os.path.exists(pop_csv_path):
        existing_pop = pd.read_csv(pop_csv_path)
        if "ea_seed" in existing_pop.columns:
            existing_pop = existing_pop[existing_pop["ea_seed"] != pop_df["ea_seed"].iloc[0]]
        combined_pop = pd.concat([existing_pop, pop_df], ignore_index=True)
        combined_pop.to_csv(pop_csv_path, index=False)
    else:
        pop_df.to_csv(pop_csv_path, index=False)

    # 3. Save Champion Genomes
    valid_inds = [ind for ind in population if ind.fitness.valid]
    if valid_inds:
        best_ind = tools.selBest(valid_inds, 1)[0]
        best_genome_arr = np.array(best_ind)
        seed_tag = f"_seed_{ea_seed}" if ea_seed is not None else ""
        np.save(os.path.join(results_dir, f"best_genome{seed_tag}.npy"), best_genome_arr)
        np.save(f"results/best_genome_{experiment_name}.npy", best_genome_arr)


def run_experiment(generations=60, pop_size=50, wind_strength=0.0, experiment_name="calm", ea_seed=None, batch_num=None, cxpb=0.8, mutpb=0.3):
    """
    Main Evolutionary Loop using DEAP and multiprocessing across all available CPU cores.
    Runs for N generations and logs detailed statistics to CSV for plot generation.
    Accepts an explicit ea_seed and batch_num for statistical replication across multiple machine runs.
    """
    if ea_seed is not None:
        random.seed(ea_seed)
        np.random.seed(ea_seed)

    num_cores = 14
    seed_str = f" | EA Seed = {ea_seed}" if ea_seed is not None else ""
    print(f"\n=======================================================")
    print(f" Starting Experiment: {experiment_name.upper()} (Wind = {wind_strength}{seed_str})")
    print(f" Parallelizing across {num_cores} CPU Cores")
    print(f"=======================================================")

    exp_start_time = time.time()

    # Initialize DEAP Toolbox
    toolbox = base.Toolbox()
    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Custom evaluation wrapper bound to current wind strength
    toolbox.register("evaluate", evaluate_individual, wind_strength=wind_strength)

    # Genetic Operators
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.1, indpb=0.15)

    population = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)

    # Bind wind strength to worker evaluation function
    eval_func = functools.partial(evaluate_individual, wind_strength=wind_strength)

    total_evaluations_count = 0

    # Create Multiprocessing Process Pool utilizing available cores
    with mp.Pool(processes=num_cores) as pool:

        # Evaluate Initial Population in Parallel
        print("Evaluating initial population...")
        gen0_start = time.time()
        results = list(tqdm(
            pool.imap(eval_func, population),
            total=len(population),
            desc="Gen 00/Initial",
            leave=False
        ))
        for ind, (fit, seed_std, align, coh, sep) in zip(population, results):
            ind.fitness.values = (fit,)
            ind.seed_std = seed_std
            ind.alignment = align
            ind.cohesion = coh
            ind.separation = sep

        total_evaluations_count += len(population)
        hof.update(population)
        gen0_time = time.time() - gen0_start

        # Print Gen 00 Baseline
        init_fits = [ind.fitness.values[0] for ind in population]
        init_sigmas = [ind[42] for ind in population]
        init_seed_stds = [getattr(ind, 'seed_std', 0.0) for ind in population]
        init_aligns = [getattr(ind, 'alignment', 0.0) for ind in population]
        init_cohs = [getattr(ind, 'cohesion', 0.0) for ind in population]
        init_seps = [getattr(ind, 'separation', 0.0) for ind in population]
        best_init_idx = np.argmax(init_fits)

        print(
            f"Gen 00/Initial | "
            f"Elite Fit: {init_fits[best_init_idx]:.4f} | "
            f"Mean Fit: {np.mean(init_fits):.4f} | "
            f"Elite Sig: {init_sigmas[best_init_idx]:.4f} | "
            f"Mean Sig: {np.mean(init_sigmas):.4f} (Std: {np.std(init_sigmas):.4f}) | "
            f"A/C/S: {init_aligns[best_init_idx]:.2f}/{init_cohs[best_init_idx]:.2f}/{init_seps[best_init_idx]:.2f} | "
            f"Time: {gen0_time:.2f}s"
        )

        # Data Logging
        history = {
            "generation": [0],
            "elite_fitness": [init_fits[best_init_idx]],
            "mean_fitness": [np.mean(init_fits)],
            "std_fitness": [np.std(init_fits)],
            "elite_sigma": [init_sigmas[best_init_idx]],
            "mean_sigma": [np.mean(init_sigmas)],
            "std_sigma": [np.std(init_sigmas)],
            "min_sigma": [np.min(init_sigmas)],
            "max_sigma": [np.max(init_sigmas)],
            "elite_seed_std": [init_seed_stds[best_init_idx]],
            "mean_seed_std": [np.mean(init_seed_stds)],
            "elite_alignment": [init_aligns[best_init_idx]],
            "mean_alignment": [np.mean(init_aligns)],
            "elite_cohesion": [init_cohs[best_init_idx]],
            "mean_cohesion": [np.mean(init_cohs)],
            "elite_separation": [init_seps[best_init_idx]],
            "mean_separation": [np.mean(init_seps)],
            "gen_time_sec": [gen0_time],
            "evaluations_count": [len(population)]
        }

        # Save Gen 0 Checkpoint
        save_checkpoint(history, population, experiment_name, ea_seed, batch_num=batch_num)

        for gen in range(1, generations + 1):
            gen_start_time = time.time()
            # 1. Select Next Generation
            offspring = toolbox.select(population, len(population))
            offspring = list(map(toolbox.clone, offspring))

            # 2. Apply Crossover (Blend)
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < cxpb:
                    toolbox.mate(child1, child2)
                    clamp_individual(child1)
                    clamp_individual(child2)
                    del child1.fitness.values
                    del child2.fitness.values

            # 3. Apply Mutation (Gaussian)
            for mutant in offspring:
                if random.random() < mutpb:
                    toolbox.mutate(mutant)
                    clamp_individual(mutant)
                    del mutant.fitness.values

            # 4. Evaluate Invalid Individuals in Parallel
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            num_evals = len(invalid_ind)
            total_evaluations_count += num_evals
            if invalid_ind:
                results = list(tqdm(
                    pool.imap(eval_func, invalid_ind),
                    total=len(invalid_ind),
                    desc=f"Gen {gen:02d}/{generations}",
                    leave=False
                ))
                for ind, (fit, seed_std, align, coh, sep) in zip(invalid_ind, results):
                    ind.fitness.values = (fit,)
                    ind.seed_std = seed_std
                    ind.alignment = align
                    ind.cohesion = coh
                    ind.separation = sep

            # Apply 1-ELITISM: Check if the best offspring beat our historical elite
            offspring_fitnesses = [ind.fitness.values[0] for ind in offspring]
            worst_idx = np.argmin(offspring_fitnesses)
            best_offspring_fit = max(offspring_fitnesses)

            if len(hof) > 0:
                elite_fit = hof[0].fitness.values[0]

                if best_offspring_fit < elite_fit:
                    elite_clone = toolbox.clone(hof[0])
                    for attr in ('seed_std', 'alignment', 'cohesion', 'separation'):
                        setattr(elite_clone, attr, getattr(hof[0], attr, 0.0))
                    offspring[worst_idx] = elite_clone

            # Replace Population
            population[:] = offspring
            hof.update(population)

            gen_duration = time.time() - gen_start_time

            # 5. Record Detailed Statistics
            fits = [ind.fitness.values[0] for ind in population]
            sigmas = [ind[42] for ind in population]
            seed_stds = [getattr(ind, 'seed_std', 0.0) for ind in population]
            aligns = [getattr(ind, 'alignment', 0.0) for ind in population]
            cohs = [getattr(ind, 'cohesion', 0.0) for ind in population]
            seps = [getattr(ind, 'separation', 0.0) for ind in population]

            best_idx = np.argmax(fits)
            elite_fit = fits[best_idx]
            mean_fit = np.mean(fits)
            std_fit = np.std(fits)

            elite_sig = sigmas[best_idx]
            mean_sig = np.mean(sigmas)
            std_sig = np.std(sigmas)
            min_sig = np.min(sigmas)
            max_sig = np.max(sigmas)

            elite_seed_std = seed_stds[best_idx]
            mean_seed_std = np.mean(seed_stds)

            elite_align = aligns[best_idx]
            mean_align = np.mean(aligns)
            elite_coh = cohs[best_idx]
            mean_coh = np.mean(cohs)
            elite_sep = seps[best_idx]
            mean_sep = np.mean(seps)

            history["generation"].append(gen)
            history["elite_fitness"].append(elite_fit)
            history["mean_fitness"].append(mean_fit)
            history["std_fitness"].append(std_fit)
            history["elite_sigma"].append(elite_sig)
            history["mean_sigma"].append(mean_sig)
            history["std_sigma"].append(std_sig)
            history["min_sigma"].append(min_sig)
            history["max_sigma"].append(max_sig)
            history["elite_seed_std"].append(elite_seed_std)
            history["mean_seed_std"].append(mean_seed_std)
            history["elite_alignment"].append(elite_align)
            history["mean_alignment"].append(mean_align)
            history["elite_cohesion"].append(elite_coh)
            history["mean_cohesion"].append(mean_coh)
            history["elite_separation"].append(elite_sep)
            history["mean_separation"].append(mean_sep)
            history["gen_time_sec"].append(gen_duration)
            history["evaluations_count"].append(num_evals)

            print(
                f"Gen {gen:02d}/{generations} | "
                f"Elite Fit: {elite_fit:.4f} | "
                f"Mean Fit: {mean_fit:.4f} | "
                f"Elite Sig: {elite_sig:.4f} | "
                f"Mean Sig: {mean_sig:.4f} (Std: {std_sig:.4f}) | "
                f"A/C/S: {elite_align:.2f}/{elite_coh:.2f}/{elite_sep:.2f} | "
                f"Time: {gen_duration:.2f}s"
            )

            # Flush generation metrics & population checkpoint to disk immediately
            save_checkpoint(history, population, experiment_name, ea_seed, batch_num=batch_num)

    # Final Timings Summary
    total_exp_time = time.time() - exp_start_time
    total_evaluations_count = sum(history["evaluations_count"])
    avg_gen_time = total_exp_time / (generations + 1)

    avg_candidate_wall_time = total_exp_time / total_evaluations_count
    total_sim_runs = total_evaluations_count * len(EVAL_SEEDS)
    avg_sim_wall_time = total_exp_time / total_sim_runs

    mins, secs = divmod(total_exp_time, 60)

    print("\n-------------------------------------------------------")
    print(f" EXPERIMENT TIMING SUMMARY ({experiment_name.upper()}{seed_str})")
    print("-------------------------------------------------------")
    print(f" Total Execution Time      : {int(mins)}m {secs:.2f}s ({total_exp_time:.2f} seconds)")
    print(f" Avg Time / Generation     : {avg_gen_time:.2f} seconds")
    print(f" Total Candidates Evaluated: {total_evaluations_count}")
    print(f" Avg Time / Candidate      : {avg_candidate_wall_time:.3f} seconds (across 5 seeds)")
    print(f" Avg Time / Simulation Run : {avg_sim_wall_time * 1000:.1f} ms (single 700-frame run)")
    print("-------------------------------------------------------\n")

    # Final Checkpoint Save
    save_checkpoint(history, population, experiment_name, ea_seed, batch_num=batch_num)

    # Return best individual genome for visualization
    best_overall_ind = tools.selBest(population, 1)[0]
    return best_overall_ind
