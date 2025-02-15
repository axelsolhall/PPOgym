# Torch imports
import torch


# Gym imports
import gym
from gym.vector import SyncVectorEnv

# Parallel imports
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed

# Video imports
import cv2

# Miscellanous imports
import uuid, warnings, os, json
import numpy as np
import matplotlib.pyplot as plt


def make_env(env_kwargs):
    return gym.make(**env_kwargs)


def run_thread(
    env_kwargs,
    num_envs,
    ppo_class,
    ppo_kwargs,
    generations,
):
    with warnings.catch_warnings():
        # Ignore deprecation warnings
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        envs_vector = SyncVectorEnv([lambda: make_env(env_kwargs)] * num_envs)
        envs_vector.reset()

        # Create instance
        ppo_instance = ppo_class(envs_vector, **ppo_kwargs)

        # Run training and return evolution
        return ppo_instance.train(generations=generations)


class HPTuner:
    def __init__(
        self,
        env_kwargs,
        num_envs,
        ppo_class,
        ppo_kwargs,
    ):
        self.env_kwargs = env_kwargs
        self.num_envs = num_envs
        self.ppo_class = ppo_class
        self.ppo_kwargs = ppo_kwargs

        self.save_file_name = f"hp_tuning_log.json"

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
        # Distribute threads
        multi_thread_sequence = self.distribute_threads(num_trials, max_threads)
        evolutions = np.zeros((num_trials, generations))

        # Run trials, up to max_threads at a time
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

        # Set up the plot
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

    def make_serializable(self, kwargs):
        """Recursively convert non-serializable objects to serializable."""
        serializable_kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, dict):
                # Recursively process nested dictionaries
                serializable_kwargs[k] = self.make_serializable(v)
            elif isinstance(v, list):
                # Process lists, keeping only serializable elements
                serializable_kwargs[k] = [
                    (
                        self.make_serializable(item)
                        if isinstance(item, dict)
                        else item.__name__ if isinstance(item, type) else item
                    )
                    for item in v
                ]
            elif isinstance(v, type):  # Handle Python class types like nn.ReLU
                # Just take the class name
                serializable_kwargs[k] = v.__name__
            elif isinstance(v, (int, float, str, bool)):
                # Directly serializable types
                serializable_kwargs[k] = v
            else:
                # Fallback: Convert other objects to strings
                serializable_kwargs[k] = str(v)
        return serializable_kwargs

    def log_results_for_hp(self, mean, std, score, final_score, generations):
        # Define the path for saving the results
        path = f"{self.save_file_name}"

        # Create the results file if it doesn't exist
        if not os.path.exists(path):
            with open(path, "w") as f:
                json.dump({}, f)

        # Create a unique key from the network and PPO kwargs
        serializable_kwargs = self.make_serializable(self.ppo_kwargs)
        key = json.dumps(
            {
                **serializable_kwargs,
                "generations": generations,
            },
            sort_keys=True,
        )

        # Load existing results
        with open(path, "r") as f:
            results = json.load(f)

        # Add new score if the configuration hasn’t been logged
        results[key] = {
            "mean": [round(m, 2) for m in mean.tolist()],
            "std": [round(s, 2) for s in std.tolist()],
            "score": [round(sc, 2) for sc in score.tolist()],
            "final_score": round(final_score, 2),
        }

        # Write updated results back to the file
        with open(path, "w") as f:
            json.dump(results, f, indent=4, separators=(", ", ": "), sort_keys=True)

    def check_results_for_hp(self, generations):
        # Define the path for the results file
        path = f"{self.save_file_name}"

        # Return None if the file does not exist
        if not os.path.exists(path):
            return None

        # Create a unique key from the network and PPO kwargs

        serializable_kwargs = self.make_serializable(self.ppo_kwargs)
        key = json.dumps(
            {
                **serializable_kwargs,
                "generations": generations,
            },
            sort_keys=True,
        )

        # Load the results
        with open(path, "r") as f:
            results = json.load(f)

        # Return the score if the key exists, otherwise None
        if key in results:
            data = results[key]
            mean = data["mean"]
            std = data["std"]
            score = data["score"]
            final_score = data["final_score"]
            return mean, std, score, final_score
        else:
            return None

    def sweep_values(self, p):
        # Handle provided values
        if isinstance(p, tuple):
            p, p_vals = p

        # Dynamic values
        else:
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
                p_vals = [np.round(pv, 4) for pv in p_vals]

            elif p_dtype == bool or p_dtype == np.bool_:
                p_vals = [p_val, not p_val]

            else:
                raise ValueError(f"Unsupported data type {p_dtype}")

        return p, p_vals

    def optimize_hyperparameters(self, parameters, generations, num_trials=1):
        parameter_idx = 0
        no_change_idx = np.zeros(len(parameters))

        while True:
            p = parameters[parameter_idx]

            p, p_vals = self.sweep_values(p)

            print(f"Optimizing {p} with values: {p_vals}")

            # Store the current value to check for change
            p_val_before = self.ppo_kwargs[p]

            # Pre-allocate storage
            serieses_mean = np.zeros((len(p_vals), generations))
            serieses_std = np.zeros((len(p_vals), generations))
            serieses_score = np.zeros((len(p_vals), generations))
            final_score = np.zeros(len(p_vals))

            # Run trials for each value
            for i, pv in enumerate(p_vals):

                #  Update the parameter
                self.ppo_kwargs[p] = pv

                # Check if the results are already logged ...
                score_load = self.check_results_for_hp(generations)
                if score_load is not None:
                    mean, std, score, final_score_value = score_load
                    print(f"Skipping {p} = {pv}, score: {final_score_value:.2f}")

                    # Store the results
                    serieses_mean[i] = mean
                    serieses_std[i] = std
                    serieses_score[i] = score
                    final_score[i] = final_score_value

                    # Move to the next parameter
                    continue

                # ... else run trials for those values
                print(f"Running trials for {p} = {pv}")

                # Run the trials
                series = self.run_trials(generations, num_trials)
                serieses_mean[i] = np.mean(series, axis=0)
                serieses_std[i] = np.std(series, axis=0)

                # Score is mean - sqrt(std) #! MAGIC FORMULA
                serieses_score[i] = serieses_mean[i] - np.sqrt(serieses_std[i])

                final_score[i] = np.sum(serieses_score[i])

                # Log the results
                self.log_results_for_hp(
                    serieses_mean[i],
                    serieses_std[i],
                    serieses_score[i],
                    final_score[i],
                    generations,
                )

            # Find the best value
            best_idx = np.argmax(final_score)

            # Update the parameter
            self.ppo_kwargs[p] = p_vals[best_idx]

            # Print the best value
            print(f"Best value for {p}: {p_vals[best_idx]}")

            # Check for change
            if p_vals[best_idx] == p_val_before:
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
                serieses_score,
                title=f"Optimizing: {p}",
                legend=[
                    str(pv) if isinstance(pv, bool) else str(np.round(pv, 4))
                    for pv in p_vals
                ],
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
        envs_vector = SyncVectorEnv([lambda: make_env(self.env_kwargs)] * 1)
        states, infos = envs_vector.reset()

        # Create instance
        ppo_kwargs_single = self.ppo_kwargs.copy()
        ppo_kwargs_single["num_envs"] = 1
        ppo_instance = self.ppo_class(envs_vector, **ppo_kwargs_single)

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
            generations=generations_sequence[-1] + 1,
            save_folder=save_folder,
            save_sequence=generations_sequence,
        )

        #### Create a video of the evolution ####

        # Make a single environment
        env_kwargs_single = self.env_kwargs.copy()
        env_kwargs_single["render_mode"] = "rgb_array"
        env_single = gym.make(**env_kwargs_single)

        ppo_single = self.ppo_class(env_single, **ppo_kwargs_single)

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
                action = ppo_instance.eval_actions(state_tensor)

                state, reward, done, truncated, info = env_single.step(action.numpy())

                if truncated:
                    done = True

                frames += 1

                # Capture frame
                frame = env_single.render()
                if video_writer is None:
                    height, width, _ = frame.shape
                    video_writer = cv2.VideoWriter(
                        video_path, fourcc, 60, (width, height)
                    )
                frame = self.process_frame(frame, text=f"Generation: {gen}")
                video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        # Release video writer and environment
        video_writer.release()
        env_single.close()

        # Print the video filename
        print(f"Video saved to {video_path}")

        # Delete the save folder if not keeping models
        if not keep_models:
            files = os.listdir(save_folder)
            for f in files:
                os.remove(f"{save_folder}/{f}")
            os.rmdir(save_folder)
