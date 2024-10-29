# Torch imports
import torch
from torch.utils.tensorboard import SummaryWriter

# Gym imports
import gym
from gym.vector import SyncVectorEnv

# Parallel imports
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed

# Video imports
import cv2

# Miscellanous imports
import uuid, warnings, os
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
        series_score,
        title,
        legend,
        x_label="Generation",
        y_label="Reward",
        y_label2="Comulative Score",
    ):
        plt.figure(figsize=(10, 5))
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        # Plot for left y-axis
        for i, s in enumerate(series_mean):
            plt.plot(s, label=legend[i])
            plt.fill_between(
                range(len(s)),
                s - series_std[i],
                s + series_std[i],
                alpha=0.08,
            )
        plt.legend()

        # Plot for right y-axis
        plt.twinx()
        plt.ylabel(y_label2)
        for i, s in enumerate(series_score):
            cumsum = np.cumsum(s)
            plt.plot(cumsum, label=legend[i], linestyle="--", linewidth=0.5)

        plt.show()

    def optimize_hyperparameters(self, parameters, generations, num_trials=1):
        parameter_idx = 0
        no_change_idx = np.zeros(len(parameters))

        while True:
            p = parameters[parameter_idx]
            p_val = self.ppo_kwargs[p]
            p_dtype = type(p_val)

            # Generate values to optimize
            scaler = np.sqrt(2)  #! MAGIC NUMBER
            if p_dtype == int or p_dtype == np.int64:
                p_vals = [
                    int(np.round(p_val / scaler)),
                    p_val,
                    int(np.round(p_val * scaler)),
                ]
            elif p_dtype == float or p_dtype == np.float64:
                p_vals = [p_val / scaler, p_val, p_val * scaler]

            elif p_dtype == bool or p_dtype == np.bool_:
                p_vals = [not p_val, p_val]

            else:
                raise ValueError(f"Unsupported data type {p_dtype}")

            print(f"Optimizing {p} with values {p_vals}")

            # Run trials for each value
            serieses_mean = np.zeros((len(p_vals), generations))
            serieses_std = np.zeros((len(p_vals), generations))
            series_score = np.zeros((len(p_vals), generations))
            for i, pv in enumerate(p_vals):
                # TODO: check if run has already beed done

                self.ppo_kwargs[p] = pv
                series = self.run_trials(generations, num_trials)
                serieses_mean[i] = np.mean(series, axis=0)
                serieses_std[i] = np.std(series, axis=0)
                # Score is mean - std/2
                # Want high mean and low std
                series_score[i] = (
                    serieses_mean[i] - serieses_std[i] * 0.5  #! MAGIC NUMBER
                )

                # TODO: save the results, ppo_kwargs as key

            # Find the best value
            best_idx = np.argmax(np.sum(series_score, axis=1))

            # Update the parameter
            self.ppo_kwargs[p] = p_vals[best_idx]

            # Print the best value
            print(f"Best value for {p}: {p_vals[best_idx]}")

            # Check for change
            if best_idx == 1:
                no_change_idx[parameter_idx] = 1
                print(
                    f"No change in {p}, no change ratio: {np.round(sum(no_change_idx)/len(parameters), 2)}"
                )
            else:
                no_change_idx = np.zeros(len(parameters))

            # Plot the results
            self.plot_series(
                serieses_mean,
                serieses_std,
                series_score,
                title=f"Optimizing: {p}",
                legend=[str(np.round(pv, 4)) for pv in p_vals],
            )

            # Check if we should stop
            if no_change_idx.all():
                break

            # Move to the next parameter
            parameter_idx = (parameter_idx + 1) % len(parameters)

    def process_frame(self, frame, text=None):
        # Add text to the frame
        if text is not None:
            cv2.putText(
                frame,
                text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (125, 125, 125),
                2,
                cv2.LINE_AA,
            )

        return frame

    def evolution_video(
        self,
        generations,
        video_folder=None,
        increments=5,
        max_frames=80,
        keep_models=False,
    ):

        # Create environments
        envs_vector = SyncVectorEnv([lambda: make_env(self.env_kwargs)] * self.num_envs)
        states, infos = envs_vector.reset()

        # Create instances
        neural_network_instance = self.network_class(**self.network_kwargs)
        ppo_instance = self.ppo_class(
            envs_vector, neural_network_instance, **self.ppo_kwargs
        )

        # Make the sequence for saving models
        if increments == 1:
            generations_sequence = [0, generations]
        else:
            gens_per = generations // (increments)
            if gens_per < 1:
                gens_per = 1
            generations_sequence = [0]

            while generations_sequence[-1] < generations:
                generations_sequence.append(generations_sequence[-1] + gens_per)

        print(f"Running evolution with save generations: {generations_sequence}")

        # Create a folder for model saves
        save_folder = f"models/{str(uuid.uuid4())[:8]}"
        os.makedirs(save_folder)

        # Run the evolution
        ppo_instance.train(
            generations=generations_sequence[-1]+1,
            save_folder=save_folder,
            save_sequence=generations_sequence,
        )

        #### Create a video of the evolution ####

        # Make a single environment
        env_kwargs_single = self.env_kwargs.copy()
        env_kwargs_single["render_mode"] = "rgb_array"
        env_single = gym.make(**env_kwargs_single)

        ppo_single = self.ppo_class(
            env_single, neural_network_instance, **self.ppo_kwargs
        )

        # Video writer
        short_uuid = str(uuid.uuid4())[:8]
        video_file = f"evo_video_{generations}_gens_{short_uuid}.mp4"
        if video_folder is None:
            video_path = f"{save_folder}" + "/" + video_file
        else:
            video_path = video_folder + "/" + video_file
            if not os.path.exists(video_folder):
                os.makedirs(video_folder)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = None

        # Run from the saved models, capture frames
        for i, gen in enumerate(generations_sequence):
            # Load the model
            ppo_instance.load(f"{save_folder}/model_{gen}")

            # Run the model
            state, info = env_single.reset()
            done = False
            frames = 0
            while not done and frames < max_frames:
                state_tensor = torch.tensor(state, dtype=torch.float32)
                policy, _ = ppo_instance.network(state_tensor)

                action = torch.argmax(policy).item()

                state, reward, done, truncated, info = env_single.step(action)

                if truncated:
                    done = True

                frames += 1

                # Capture frame
                frame = env_single.render()
                if video_writer is None:
                    height, width, _ = frame.shape
                    video_writer = cv2.VideoWriter(
                        video_path, fourcc, 48, (width, height)
                    )
                frame = self.process_frame(frame, text=f"Generation: {gen}")
                video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


        # Release video writer and environment
        video_writer.release()
        env_single.close()

        # Print the video filename
        print(f"Video saved to {video_path}")

        # Delete the save folder if video folder is provided
        if not keep_models:
            files = os.listdir(save_folder)
            for f in files:
                os.remove(f"{save_folder}/{f}")
            os.rmdir(save_folder)
