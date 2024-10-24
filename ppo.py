import torch,  time, os, json, gym, warnings
import numpy as np
import matplotlib.pyplot as plt
from torch import nn, optim
import torch.nn.functional as F
#from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
from gym.vector import SyncVectorEnv

class PPONetwork(nn.Module):
    def __init__(self, *args, **kwargs):
        super(PPONetwork, self).__init__()
        self.input_dim = kwargs.get("input_dim")
        self.output_dim = kwargs.get("output_dim")
        self.hidden_dims = kwargs.get("hidden_dims")
        self.policy_hidden_dims = kwargs.get("policy_hidden_dims")
        self.value_hidden_dims = kwargs.get("value_hidden_dims")
        
        self.shared_layers = self.build_layers(self.input_dim, self.hidden_dims, normalize=True)
        
        # Policy head layers
        self.policy_layers = self.build_layers(self.hidden_dims[-1], self.policy_hidden_dims, normalize=True)
        self.policy_output = nn.Linear(self.policy_hidden_dims[-1], self.output_dim)
        
        # Value head layers
        self.value_layers = self.build_layers(self.hidden_dims[-1], self.value_hidden_dims, normalize=True)
        self.value_output = nn.Linear(self.value_hidden_dims[-1], 1)
        
        # Apply He initialization to the layers
        self.apply(self.he_initialization)
    
    def build_layers(self, input_size, layer_dims, normalize=False):
        layers = []
        for dim in layer_dims:
            layers.append(nn.Linear(input_size, dim))
            if normalize:
                layers.append(nn.LayerNorm(dim)) 
            layers.append(nn.ReLU())
            #layers.append(nn.LeakyReLU())
            input_size = dim  # Set input size to the output of the last layer
        return nn.Sequential(*layers)  # Return the layers as a sequential model

    def he_initialization(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')  # He initialization
            if module.bias is not None:
                nn.init.zeros_(module.bias)  # Initialize bias as zeros

    def forward(self, x):
        # Pass through shared layers
        x = self.shared_layers(x)
        
        # Policy head
        policy_x = self.policy_layers(x)
        policy = F.softmax(self.policy_output(policy_x), dim=-1)
        
        # Value head
        value_x = self.value_layers(x)
        value = self.value_output(value_x)
        
        return policy, value


class PPOWrapper:
    def __init__(self, env, network, *args, **kwargs):
        self.env = env
        self.network = network
        self.gamma = kwargs.get("gamma")
        self.lam = kwargs.get("lam")
        self.clip = kwargs.get("clip_epsilon")
        self.initial_lr = kwargs.get("initial_lr")
        self.final_lr = kwargs.get("final_lr")
        self.value_coef = kwargs.get("value_coef")
        self.entropy_coef = kwargs.get("entropy_coef")
        self.batch_size = kwargs.get("batch_size")
        self.batch_epochs = kwargs.get("batch_epochs")
        self.batch_shuffle = kwargs.get("batch_shuffle")
        self.checkpointing = kwargs.get("checkpointing")
        
        # Episode parameters # TODO: remove this logic
        self.truncated_reward = kwargs.get("truncated_reward")
        self.episode_steps = 1500 # Maximum step to start an new episode
        self.max_steps = 100000 # Maximum number of steps per episode
        self.eval_multiplier = 1 # Evaluate the policy for 5 times the number of steps

        # Optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.initial_lr)
        
        # Annealing
        self.anneal_clip = kwargs.get("anneal_clip")
        self.initial_clip = None
        self.anneal_entropy = kwargs.get("anneal_entropy")
        self.initial_entropy = None
        self.anneal_lambda = kwargs.get("anneal_lambda")
        self.initial_lambda = None
        
        # Checkpointing    
        self.checkpoint_path = "checkpoints"   
        self.ma_factor = 0.9
        self.ma_save_ratio = 1.05
        self.ma_load_ratio = 0.75
        self.minimum_episode_spacing = 50
        
        self.eval_ma = 0
        self.max_eval_ma = 0
        self.checkpoint_grace = 0.001
        self.latest_checkpoint_episode = 0
        self.model_save = None
        self.optimizer_save = None
        self.ma_save = None
        
        #Statistics
        self.avg_lifetime = None
        
    def predict(self, state):
        return self.network(state)
    
    def compute_advantages(self, rewards, dones, values):
        advantages = torch.zeros_like(rewards, dtype=torch.float32)
        returns = torch.zeros_like(rewards, dtype=torch.float32)
        gae = 0

        values = torch.cat((values, torch.tensor([0.0], dtype=torch.float32)))

        for i in reversed(range(len(rewards))):
            if dones[i]:
                next_value = 0
            else:
                next_value = values[i + 1]
            
            delta = rewards[i] + self.gamma * next_value - values[i]
            gae = delta + self.gamma * self.lam * (1 - dones[i]) * gae
            advantages[i] = gae
            returns[i] = gae + values[i]

        return advantages, returns

    def run_episode(self, steps):
        states, actions, rewards, log_probs, dones, values = [], [], [], [], [], []
        avg_lifetime = []
        while steps > 0:
            state, info = self.env.reset()
            lifetime = 0
            done = False
            while not done:
            #for _ in range(self.max_steps):
                state_tensor = torch.tensor(state, dtype=torch.float32)
                
                # Get the policy distribution and value prediction
                policy, value = self.predict(state_tensor)
                
                # Create a categorical distribution from the policy probabilities
                action_dist = torch.distributions.Categorical(policy)
                
                # Sample an action from the distribution
                action = action_dist.sample().item()
                
                # Take the action in the environment
                next_state, reward, done, truncated, info = self.env.step(action)
                
                # Handle truncated logic
                if truncated:
                    done = True
                    reward += self.truncated_reward
                
                # Get the log probability of the chosen action
                log_prob = action_dist.log_prob(torch.tensor(action)).detach()

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                log_probs.append(log_prob)
                dones.append(done)
                values.append(value)

                state = next_state
                
                steps -= 1
                lifetime += 1

                if done:
                    break
                
            avg_lifetime.append(lifetime)
                  
        self.avg_lifetime = np.mean(avg_lifetime)

        return states, actions, rewards, log_probs, dones, values
    
    def train(self, states, actions, rewards, old_log_probs, dones, values):
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        values = torch.tensor(values, dtype=torch.float32)

        advantages, returns = self.compute_advantages(rewards, dones, values)
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # TODO: Try normalize the returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        for _ in range(self.batch_epochs):
            # Shuffle the data
            if self.batch_shuffle:
                indices = np.arange(len(states))
                np.random.shuffle(indices)
                
                states = states[indices]
                actions = actions[indices]
                old_log_probs = old_log_probs[indices]
                advantages = advantages[indices]
                returns = returns[indices]
                values = values[indices]
            
            for i in range(0, len(states), self.batch_size):
                batch_states = states[i:i + self.batch_size]
                batch_actions = actions[i:i + self.batch_size]
                batch_old_log_probs = old_log_probs[i:i + self.batch_size]
                batch_advantages = advantages[i:i + self.batch_size]
                batch_returns = returns[i:i + self.batch_size]
                batch_values = values[i:i + self.batch_size]
                
                # Get new policy and value
                new_policy_probs, new_value = self.network(batch_states)
                new_action_dist = torch.distributions.Categorical(new_policy_probs)
                
                # Log probability of the action
                log_probs = new_action_dist.log_prob(batch_actions)
                #old_log_probs = new_action_dist.log_prob(batch_actions).detach()
                
                # Compute the ratio
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss: compare predicted values with returns (rewards-to-go)
                value_loss = self.value_coef * 0.5 * (batch_returns - new_value.squeeze()).pow(2).mean()
                
                # Entropy loss
                entropy = -torch.sum(new_policy_probs * torch.log(new_policy_probs + 1e-8), dim=-1).mean()
                entropy_loss = -self.entropy_coef * entropy
                
                # Total loss
                loss = policy_loss + value_loss + entropy_loss
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
    def adjust_learning_rate(self, current_epoch, total_epochs):
        # Linearly decay the learning rate
        new_lr = (self.initial_lr-self.final_lr) * (1 - current_epoch/total_epochs) + self.final_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
            
        # TODO try exponential decay
        
    def anneling(self, current_epoch, total_epochs):
        # Anneal the clip parameter
        if self.anneal_clip:
            if self.initial_clip is None:
                self.initial_clip = self.clip
            else:
                self.clip = self.initial_clip * (1 - current_epoch/total_epochs)
                
        # Anneal the entropy coefficient
        if self.anneal_entropy:
            if self.initial_entropy is None:
                self.initial_entropy = self.entropy_coef
            else:
                self.entropy_coef = self.initial_entropy * (1 - current_epoch/total_epochs)
                
        # Anneal the lambda parameter
        if self.anneal_lambda:
            if self.initial_lambda is None:
                self.initial_lambda = self.lam
            else:
                self.lam = self.initial_lambda * (1 - current_epoch/total_epochs)
    
    def evaluate_policy(self, episodes):
        # Increase the max steps for evaluation
        max_steps = self.env.spec.max_episode_steps
        self.env.spec.max_episode_steps = max_steps * self.eval_multiplier
        
        # # Evaluate the policy without randomness
        # scores = []
        # for i in range(episodes):
        #     state, info = self.env.reset()
            
        #     done = False
        #     score = 0
        #     steps_taken = 0
        #     while not done:
        #     #for _ in range(self.max_steps*self.eval_multiplier):
        #         policy, _ = self.predict(torch.tensor(state, dtype=torch.float32))
        #         action = torch.argmax(policy).item()
        #         state, reward, done, truncated, info = self.env.step(action)
                
        #         # if info dict is not empty, print info
        #         if len(info) > 0:
        #             # Print info in blue
        #             output = f"\t\033[94mInfo: {info}\033[00m"
                
        #         if truncated:
        #             done = True
        #             reward += self.truncated_reward

        #         score += reward
        #         steps_taken += 1
                
        #         if done:
        #             print(f"\tEval reward: {score:.2f}, steps taken: {steps_taken}")
        #             break
                
        #     scores.append(score)
        
        # Create a vectorized environment
        envs = SyncVectorEnv([lambda: self.env] * episodes)
        
        # Reset all environments at the start
        states, infos = envs.reset()
        
        # Initialize tracking variables
        scores = np.zeros(episodes)
        steps_taken = np.zeros(episodes)
        dones = np.zeros(episodes, dtype=bool)
        
        while not np.all(dones):
            # Predict actions for all environments, but mask those that are done
            active_envs = ~dones
            
            # Placeholder action for done environments
            actions = np.zeros(episodes, dtype=int)  # Default action for inactive envs (e.g., action 0)
            
            if np.any(active_envs):  # Ensure there are active environments
                policies, _ = self.predict(torch.tensor(states[active_envs], dtype=torch.float32))
                actions[active_envs] = torch.argmax(policies, dim=1).numpy()  # Only predict for active envs
                
            # Step the environments with actions for all envs
            next_states, rewards, done_flags, truncated_flags, infos = envs.step(actions)
            
            # Update scores for environments that aren't done
            scores[active_envs] += rewards[active_envs]
            steps_taken[active_envs] += 1
            
            # Update the 'done' flags
            dones[active_envs] = done_flags[active_envs] | truncated_flags[active_envs]

        # Print evaluation results for environments that are done
        for i in range(episodes):
            if dones[i]:
                print(f"\t - Env {i}: reward = {scores[i]:.2f}, steps taken = {steps_taken[i]}")
    
        # Calculate average score
        avg_score = np.mean(scores)
        
        # Close the environments
        envs.close()
        
        # Reset the max steps
        self.env.spec.max_episode_steps = max_steps 
            
        return np.mean(scores)
                
    def train_model(self, episodes, display=False):
        series = []
        for i in range(episodes):
            # Parameters annealing
            self.adjust_learning_rate(i, episodes)
            self.anneling(i, episodes)
            #print(f"epoch: {i}, clip: {self.clip}, entropy: {self.entropy_coef}, lambda: {self.lam}")
            
            # Run an episode
            states, actions, rewards, log_probs, dones, values = self.run_episode(self.episode_steps)
            self.train(states, actions, rewards, log_probs, dones, values)
            
            # Evaluate the policy
            eval_reward = self.evaluate_policy(10)
            series.append(eval_reward)

            # Checkpointing logic
            self.checkpoint(eval_reward, i, episodes)
            
            if display:
                print(f"Episode: {i}\t avg life: {self.avg_lifetime:.2f}, eval reward: {eval_reward:.2f}, eval ma: {self.eval_ma:.2f}")
            
        return series
            
    def checkpoint(self, eval_reward, episode, total_epochs):
        # TODO nothing here works
        
        # TODO move this calculation
        # Update the moving average
        self.eval_ma *= self.ma_factor
        self.eval_ma += (1 - self.ma_factor) * eval_reward
        
        if not self.checkpointing:
            return
        
        if episode < total_epochs * self.checkpoint_grace:
            return
    
        if self.max_eval_ma == 0:
            # Print in blue
            output = f"\033[94mFirst checkpoint, eval ma: {self.eval_ma}\033[00m"
            print(output)
            self.max_eval_ma = self.eval_ma
            
        # Calculate the ratio
        if self.max_eval_ma == 0:
            ratio = 0
        else:
            ratio = self.eval_ma / self.max_eval_ma
            if self.eval_ma < 0:
                ratio = 1/ratio
        
        # Saving logic
        #if ratio > self.ma_save_ratio and self.eval_ma > self.max_eval_ma:
        if self.eval_ma > self.max_eval_ma:
            # Update the maximum evaluation moving average
            self.max_eval_ma = self.eval_ma
            self.latest_checkpoint_episode = episode
            
            # Save the model and optimizer
            self.model_save = f"model_{episode}.pth"
            self.optimizer_save = f"optimizer_{episode}.pth"
            self.ma_save = f"ma_{episode}.txt"
            
            self.save_model(os.path.join(self.checkpoint_path, self.model_save))
            self.save_optimizer(os.path.join(self.checkpoint_path, self.optimizer_save))
            with open(os.path.join(self.checkpoint_path, self.ma_save), "w") as f:
                f.write(str(self.eval_ma))
                
            files = os.listdir(self.checkpoint_path)
            for f in files:
                if f != self.model_save and f != self.optimizer_save and f != self.ma_save:
                    os.remove(os.path.join(self.checkpoint_path, f))
                    
            # Print in green
            output = "\033[92mSaved model at episode {}\033[00m".format(episode)
            print(output)   
            
        # Calculate the spacing    
        episode_spacing = episode - self.latest_checkpoint_episode
        
        if self.model_save == None:
            return
        try:             
            # Restore logic
            if ratio < self.ma_load_ratio and episode_spacing > self.minimum_episode_spacing:
                self.load_model(os.path.join(self.checkpoint_path, self.model_save))
                self.load_optimizer(os.path.join(self.checkpoint_path, self.optimizer_save))
                with open(os.path.join(self.checkpoint_path, self.ma_save), "r") as f:
                    self.eval_ma = float(f.read())
                
                # Print in red
                output = "\033[91mRestored model from episode {}\033[00m".format(self.latest_checkpoint_episode)
                print(output)
                self.latest_checkpoint_episode = episode
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
        
    def save_model(self, path):
        torch.save(self.network.state_dict(), path)
        
    def load_model(self, path):
        self.network.load_state_dict(torch.load(path))
        
    def save_optimizer(self, path):
        torch.save(self.optimizer.state_dict(), path)
        
    def load_optimizer(self, path):
        self.optimizer.load_state_dict(torch.load(path))
        
    def save_hyperparameters(self, path):
        # TODO
        pass
    
    def load_hyperparameters(self, path):
        # TODO
        pass
    
def sweep_worker(parameter_name, value, episodes, env_kwargs, network_kwargs, ppo_kwargs):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        
        env = gym.make(**env_kwargs)
        network = PPONetwork(**network_kwargs)
        ppo_kwargs[parameter_name] = value
        ppo = PPOWrapper(env, network, **ppo_kwargs)
        
        eval_series = ppo.train_model(episodes)
        
        return eval_series
    
def distribute_threads(threads, max_concurrent_threads):
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
        t = (threads + denominator - 1 - i) // denominator  # Distribute threads as evenly as possible
        conc_threads_sequence.append(t)

    return conc_threads_sequence

def parameter_sweeper(episodes, threads, parameter_name, parameter_values, env_kwargs, network_kwargs, ppo_kwargs, max_concurrent_threads=16, custom_name="custom"):
    if parameter_name == None:
        parameter_name = custom_name
    if parameter_values == None:
        parameter_values = [0]
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"{parameter_name}_sweep_{timestamp}"
    folder = "param_sweeps/"

    print(f"Starting parameter: {parameter_name}, values: {parameter_values}")
    
    obj = {}
    meta_data = {}
    meta_data["parameter"] = parameter_name
    meta_data["episodes"] = episodes
    meta_data["threads"] = threads
    meta_data["env_kwargs"] = env_kwargs
    meta_data["network_kwargs"] = network_kwargs
    meta_data["ppo_kwargs"] = ppo_kwargs
    
    obj["meta_data"] = meta_data
    
    sweeps = {}
    for value in parameter_values:
        print(f"\tRunning value: {value}")
        
        # Distribute threads      
        conc_threads_sequence = distribute_threads(threads, max_concurrent_threads)      
        results = np.zeros((threads, episodes))
        
        offset = 0
        for t in conc_threads_sequence:
            with ProcessPoolExecutor(max_workers=t) as executor:
                futures = []
                
                # Submit the threads
                for _ in range(t):
                    futures.append(
                        executor.submit(
                            sweep_worker, parameter_name, value, episodes, env_kwargs, network_kwargs, ppo_kwargs.copy()
                        )
                    )
            
                # Wait for all threads to finish
                for index, future in enumerate(as_completed(futures)):
                    result = future.result()
                    results[offset+index] = result
                
                # Update the offset
                offset += t
                
        # Calculate the mean and std deviation as a series
        result_mean = np.mean(results, axis=0)
        result_mean = np.round(result_mean, 2)
        
        result_std = np.std(results, axis=0)
        result_std = np.round(result_std, 2)
                    
        sweeps[value] = {
            "mean": result_mean.tolist(),
            "std": result_std.tolist()
        }
        
    obj["sweeps"] = sweeps
        
    # Save the results 
    with open(os.path.join(folder, f"{filename}.json"), "w") as f:
        json.dump(obj, f, indent=4)

    # Return the filepath
    return f"{folder}{filename}.json"

def pick_parameter_sweep_results(filename):
    # Pick result with greatets AOC
    
    with open(filename, "r") as f:
        obj = json.load(f)
        
    param_name = obj["meta_data"]["parameter"]
        
    data = obj["sweeps"]
    
    best_parameter_value = None
    best_mean = 0
    
    for i, (param, values) in enumerate(data.items()):
        try:
            param_value = int(param)  # Try converting to int
        except ValueError:
            try:
                param_value = float(param)  # If int fails, try float
            except ValueError:
                param_value = param  # If both fail, keep as a string
        
        mean = np.array(values['mean'])
        if np.sum(mean) > best_mean:
            best_mean = np.sum(mean)
            best_parameter_value = param_value
            
    print(f"Best value for parameter {param_name}: {best_parameter_value}")
    
    return param_name, best_parameter_value

def moving_average_with_padding(data, window_size):
    pad_before = np.zeros(window_size//2 )
    pad_after = np.full((window_size//2,), data[-1])
    
    padded_data = np.concatenate((pad_before, data, pad_after))
    
    ma = np.convolve(padded_data, np.ones(window_size) / window_size, mode='valid')
    
    return ma

def plot_sweep_results(filename, moving_average=None):
    # Load the data
    with open(filename, "r") as f:
        obj = json.load(f)
        
    data = obj["sweeps"]
    
    # Create a plot
    fig, ax = plt.subplots()

    # Different colors for each line
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta']

    # Loop through the data and plot each mean with its std deviation as a shaded area
    for i, (param, values) in enumerate(data.items()):
        mean = np.array(values['mean'])
        std = np.array(values['std'])
        
        if moving_average:
            mean_ma = moving_average_with_padding(mean, moving_average)
            std_ma = moving_average_with_padding(std, moving_average)
            
            x = np.arange(len(mean_ma))
            ax.plot(x, mean_ma, color=colors[i], label=f'{param}')
            ax.fill_between(x, mean_ma - std_ma, mean_ma + std_ma, color=colors[i], alpha=0.07)
        else:
            x = np.arange(len(mean))
            ax.plot(x, mean, color=colors[i], label=f'{param}')
            ax.fill_between(x, mean - std, mean + std, color=colors[i], alpha=0.07)
            

    # Add labels and a legend
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Sweep: {}'.format(obj["meta_data"]["parameter"]))
    ax.legend()
    
    # Add grid
    ax.grid()
    
    # Show the plot
    plt.show()