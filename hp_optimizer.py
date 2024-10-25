# Torch imports
import torch
from torch.utils.tensorboard import SummaryWriter

# Gym imports
import gym
from gym.vector import SyncVectorEnv

# Parallel imports
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed

# Miscellanous imports
import uuid, warnings
import numpy as np
import matplotlib.pyplot as plt


def make_env(env_kwargs):
    return gym.make(**env_kwargs)


def run_thread(
    env_kwargs,
    num_envs,
    network_class,
    network_kwargs,
    ppo_class,
    ppo_kwargs,
    generations,
):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        envs_vector = SyncVectorEnv([lambda: make_env(env_kwargs)] * num_envs)
        envs_vector.reset()

        # Create instances
        neural_network_instance = network_class(**network_kwargs)
        ppo_instance = ppo_class(envs_vector, neural_network_instance, **ppo_kwargs)

        # Run training and return evolution
        evolution = ppo_instance.train(generations=generations)
        return evolution


class HPOptimizer:
    def __init__(
        self,
        env_kwargs,
        num_envs,
        network_class,
        network_kwargs,
        ppo_class,
        ppo_kwargs,
    ):
        self.uuid = uuid
        self.env_kwargs = env_kwargs
        self.num_envs = num_envs
        self.network_class = network_class
        self.network_kwargs = network_kwargs
        self.ppo_class = ppo_class
        self.ppo_kwargs = ppo_kwargs

    def run_parallel(self, generations, num_threads):
        evolutions = np.zeros((num_threads, generations))
        futures = []

        # Using ProcessPoolExecutor to parallelize runs
        with ProcessPoolExecutor(max_workers=num_threads) as executor:
            for _ in range(num_threads):
                future = executor.submit(
                    run_thread,
                    self.env_kwargs,
                    self.num_envs,
                    self.network_class,
                    self.network_kwargs,
                    self.ppo_class,
                    self.ppo_kwargs,
                    generations,
                )
                futures.append(future)

            # Collect results as they complete
            # for future in as_completed(futures):
            #     evolutions = future.result()
            for i, future in enumerate(as_completed(futures)):
                evolutions[i] = future.result()

        return evolutions

    def distribute_threads(self, threads, max_concurrent_threads):
        # Step 1: Calculate the number of bins (denominator)
        denominator = 1
        conc_threads = threads
        while conc_threads > max_concurrent_threads:
            denominator += 1
            conc_threads = threads / denominator

        # Step 2: Distribute threads into bins
        conc_threads_sequence = []
        for i in range(denominator):
            # Calculate the number of threads to assign to this bin
            t = (
                threads + denominator - 1 - i
            ) // denominator  # Distribute threads as evenly as possible
            conc_threads_sequence.append(t)

        return conc_threads_sequence

    def run_trials(self, generations, num_trials, max_threads=8):
        multi_thread_sequence = self.distribute_threads(num_trials, max_threads)
        evolutions = np.zeros((num_trials, generations))

        offset = 0
        for i, threads in enumerate(multi_thread_sequence):
            evos = self.run_parallel(generations, threads)

            # Store results
            evolutions[offset : offset + threads] = evos

            offset += threads

        return evolutions

    def plot_series(
        self,
        series_mean,
        series_std,
        title,
        legend,
        x_label="Reward",
        y_label="Generation",
    ):
        plt.figure(figsize=(10, 5))
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        for i, s in enumerate(series_mean):
            plt.plot(s, label=legend[i])
            plt.fill_between(
                range(len(s)),
                s - series_std[i],
                s + series_std[i],
                alpha=0.06,
            )
        plt.legend()
        plt.show()

    def optimize_hyperparameters(
        self, parameters, generations, num_trials=1, change_rate=0.3
    ):
        parameter_idx = 0
        no_change_counter = 0

        while True:
            p = parameters[parameter_idx]
            p_val = self.ppo_kwargs[p]
            p_dtype = type(p_val)

            # Generate values to optimize
            scaler = 1 + change_rate
            if p_dtype == int:
                p_vals = [int(p_val / scaler), p_val, int(p_val * scaler)]

            elif p_dtype == float:
                p_vals = [p_val / scaler, p_val, p_val * scaler]

            elif p_dtype == bool:
                p_vals = [not p_val, p_val]

            else:
                raise ValueError(f"Unsupported data type {p_dtype}")

            print(f"Optimizing {p} with values {p_vals}")

            # Run trials for each value
            serieses_mean = np.zeros((len(p_vals), generations))
            serieses_std = np.zeros((len(p_vals), generations))
            for i, pv in enumerate(p_vals):
                # TODO: check if run has already beed done

                self.ppo_kwargs[p] = pv
                series = self.run_trials(generations, num_trials)
                serieses_mean[i] = np.mean(series, axis=0)
                serieses_std[i] = np.std(series, axis=0)

            # Find the best value
            best_idx = np.argmax(np.sum(serieses_mean, axis=1))

            # Update the parameter
            self.ppo_kwargs[p] = p_vals[best_idx]

            # Print the best value
            print(f"Best value for {p}: {p_vals[best_idx]}")

            # Plot the results
            self.plot_series(
                serieses_mean,
                serieses_std,
                title=f"Optimizing: {p}",
                legend=[str(pv) for pv in p_vals],
            )

            # Check if we should continue
            if best_idx == 1:
                no_change_counter += 1
                print(f"No change in {p}, total no change: {no_change_counter}")
            else:
                no_change_counter = 0

            if no_change_counter >= len(parameters):
                break

            # Move to the next parameter
            parameter_idx = (parameter_idx + 1) % len(parameters)
