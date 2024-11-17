import torch
from torch import nn, optim
import numpy as np


class MovingAverageNormalizer:
    def __init__(self, decay=0.99, use_torch=True):
        self.mean = 0.0
        self.var = 1.0
        self.decay = decay
        self.use_torch = use_torch  # Flag to specify PyTorch or NumPy

    def update(self, x):
        if self.use_torch:
            batch_mean = x.mean()
            batch_var = x.var()
        else:
            batch_mean = np.mean(x)
            batch_var = np.var(x)

        # Moving averages for mean and variance
        self.mean = self.decay * self.mean + (1 - self.decay) * batch_mean
        self.var = self.decay * self.var + (1 - self.decay) * batch_var

    def normalize(self, x):
        if self.use_torch:
            return (x - self.mean) / (torch.sqrt(self.var) + 1e-8)
        else:
            return (x - self.mean) / (np.sqrt(self.var) + 1e-8)


class PPOBase:
    def __init__(self, envs, *args, **kwargs):
        # Environments vector
        self.envs = envs
        self.num_envs = kwargs.get("num_envs", 1)  # Default is 1

        # Network
        self.network_class = kwargs.get("network_class", None)
        self.network_kwargs = kwargs.get("network_kwargs", None)
        self.network = self.network_class(**self.network_kwargs)
        self.lr = kwargs.get("lr", None)
        self.final_lr = kwargs.get("final_lr", None)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)

        assert self.network_class != None, "Network class must be provided"
        assert self.network_kwargs != None, "Network kwargs must be provided"
        assert self.lr != None, "Learning rate must be provided"

        # PPO parameters
        self.gamma = kwargs.get("gamma", None)
        self.lam = kwargs.get("lam", None)
        self.clip_eps = kwargs.get("clip_eps", None)
        self.final_clip_eps = kwargs.get("final_clip_eps", None)
        self.value_coef = kwargs.get("value_coef", None)
        self.entropy_coef = kwargs.get("entropy_coef", None)
        self.final_entropy_coef = kwargs.get("final_entropy_coef", None)

        assert self.gamma != None, "Gamma must be provided"
        assert self.lam != None, "Lambda must be provided"
        assert self.clip_eps != None, "Clip epsilon must be provided"
        assert self.value_coef != None, "Value coefficient must be provided"
        assert self.entropy_coef != None, "Entropy coefficient must be provided"

        # Reward normalization
        self.reward_normalization = kwargs.get("reward_normalization", False)
        self.reward_normalizer = MovingAverageNormalizer()

        # Batch parameters
        self.batch_size = kwargs.get("batch_size", None)
        self.mini_batch_size = kwargs.get("mini_batch_size", None)
        self.batch_epochs = kwargs.get("batch_epochs", None)
        self.batch_shuffle = kwargs.get("batch_shuffle", False)
        self.seperate_envs_shuffle = kwargs.get("seperate_envs_shuffle", False)

        assert self.batch_size != None, "Batch size must be provided"
        assert self.mini_batch_size != None, "Mini batch size must be provided"
        assert self.batch_epochs != None, "Batch epochs must be provided"

        # Checkpointing
        self.checkpointing = kwargs.get("checkpointing", False)
        self.checkpoint_folder = kwargs.get("checkpoint_folder", "checkpoints")

        # TODO: class for MA

        # Miscellaneous
        self.truncated_reward = kwargs.get("truncated_reward", 0)
        self.checkpointing = kwargs.get("checkpointing", False)
        self.debug_prints = kwargs.get("debug_prints", False)
        self.generation = 0

    def parameter_scheduler(self, gen, total_gens):
        # Linearly anneal the learning rate
        if self.final_lr is not None:
            if not hasattr(self, "initial_lr"):
                self.initial_lr = self.lr

            alpha = gen / total_gens
            self.lr = self.initial_lr + alpha * (self.final_lr - self.initial_lr)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.lr

            if self.debug_prints:
                print(f"Current learning rate: {self.lr}")

        # Linearly anneal the clip epsilon
        if self.final_clip_eps is not None:
            if not hasattr(self, "initial_clip_eps"):
                self.initial_clip_eps = self.clip_eps

            alpha = gen / total_gens
            self.clip_eps = self.initial_clip_eps + alpha * (
                self.final_clip_eps - self.initial_clip_eps
            )

            if self.debug_prints:
                print(f"Current clip epsilon: {self.clip_eps}")

        # Linearly anneal the entropy coefficient
        if self.final_entropy_coef is not None:
            if not hasattr(self, "initial_entropy_coef"):
                self.initial_entropy_coef = self.entropy_coef

            alpha = gen / total_gens
            self.entropy_coef = self.initial_entropy_coef + alpha * (
                self.final_entropy_coef - self.initial_entropy_coef
            )

            if self.debug_prints:
                print(f"Current entropy coefficient: {self.entropy_coef}")

    def collect_trajectories(self):

        # Pre-allocate storage
        states = torch.zeros(self.batch_size, self.num_envs, self.network.input_dims)
        actions = torch.zeros(self.batch_size, self.num_envs, self.action_dims)
        log_probs = torch.zeros(self.batch_size, self.num_envs)
        rewards = torch.zeros(self.batch_size, self.num_envs)
        dones = torch.zeros(self.batch_size, self.num_envs)
        truncateds = torch.zeros(self.batch_size, self.num_envs)
        values = torch.zeros(self.batch_size, self.num_envs)

        # Reset environments
        current_states, infos = self.envs.reset()
        # Iterate through iterations
        for itr in range(self.batch_size):
            # Convert states to tensor
            states_tensor = torch.tensor(current_states, dtype=torch.float32)

            # Get actions, log_probs and values
            current_actions, current_log_probs, current_values = (
                self.forward_pass_collect(states_tensor)
            )

            # Step through environments
            next_states, current_rewards, current_dones, current_truncateds, infos = (
                self.envs.step(current_actions.numpy())
            )

            # Check if current_actions is a scalar
            if current_actions.dim() == 1:
                current_actions = current_actions.unsqueeze(-1)

            # Add truncated rewards
            if current_truncateds.any():
                current_rewards += current_truncateds * self.truncated_reward
                current_dones = current_dones | current_truncateds

            current_states = next_states

            # Store data
            states[itr] = states_tensor
            actions[itr] = current_actions  # .unsqueeze(-1)  # & Changed here
            log_probs[itr] = current_log_probs.detach()
            rewards[itr] = torch.tensor(current_rewards, dtype=torch.float32)
            dones[itr] = torch.tensor(current_dones, dtype=torch.float32)
            truncateds[itr] = torch.tensor(current_truncateds, dtype=torch.float32)
            values[itr] = current_values.squeeze(-1).detach()

        return states, actions, log_probs, rewards, dones, truncateds, values

    def compute_advantages(self, rewards, dones, truncateds, values):
        # Normalize rewards
        if self.reward_normalization:
            self.reward_normalizer.update(rewards)
            rewards = self.reward_normalizer.normalize(rewards)

        # Initialize storage
        advantages = torch.zeros_like(rewards, dtype=torch.float32)
        returns = torch.zeros_like(rewards, dtype=torch.float32)
        gaes = torch.zeros(self.num_envs, dtype=torch.float32)

        # Add a zero value at the end
        values = torch.cat((values, torch.zeros((1, self.num_envs))), dim=0)

        # Compute advantages and returns
        for ti in reversed(range(rewards.size(0))):

            next_values = values[ti + 1]
            mask = (dones[ti] + truncateds[ti]) > 0
            next_values = next_values * (~mask)  # Mask with inverse (~mask)

            deltas = rewards[ti] + self.gamma * next_values - values[ti]
            gaes = deltas + self.gamma * self.lam * gaes * (1 - dones[ti])
            advantages[ti] = gaes
            returns[ti] = gaes + values[ti]

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        return advantages, returns

    def evaluate(self):
        # Pre-allocate storage
        scores_w_trunc_rew = np.zeros(self.num_envs)
        scores_wo_trunc_rew = np.zeros(self.num_envs)
        alive = np.ones(self.num_envs)

        # Reset environments
        current_states, infos = self.envs.reset()

        # Iterate through iterations
        for itr in range(self.batch_size):
            # Get policies, no need for values
            states_tensor = torch.tensor(current_states, dtype=torch.float32)

            # Argmax because we are in evaluation mode
            actions = self.eval_actions(states_tensor)

            # Step through environments
            next_states, current_rewards, current_dones, current_truncateds, infos = (
                self.envs.step(actions.numpy())
            )

            # Add truncated rewards
            current_rewards_w_trunc = current_rewards.copy()
            if current_truncateds.any():
                current_rewards_w_trunc += current_truncateds * self.truncated_reward
                current_dones = current_dones | current_truncateds

            # Break if all environments are done
            if not alive.any():
                break

            # Update states
            current_states = next_states

            # Update scores, w and w/o truncated rewards
            scores_w_trunc_rew += current_rewards_w_trunc * alive
            scores_wo_trunc_rew += current_rewards * alive

            # Kill environments that are done
            alive[current_dones] = 0

        # Return the mean scores, w and w/o truncated rewards
        return np.mean(scores_w_trunc_rew), np.mean(scores_wo_trunc_rew)

    def save(self, path):
        # Save the model
        torch.save(self.network.state_dict(), path)

        # Save the optimizer
        torch.save(self.optimizer.state_dict(), path + "_optimizer")

    def load(self, path):
        # Load the model
        self.network.load_state_dict(torch.load(path))

        # Load the optimizer
        self.optimizer.load_state_dict(torch.load(path + "_optimizer"))

    def checkpointer(self, eval_reward, gen, total_gens):

        # Update moving average
        if self.eval_ma is not None:
            self.eval_ma *= self.eval_ma_alpha
            self.eval_ma += (1 - self.eval_ma_alpha) * eval_reward
        else:
            self.eval_ma = eval_reward

        # If checkpointing is disabled, return
        if not self.checkpointing:
            return

        # If the grace period is not met, return
        if gen < self.grace_period * total_gens:
            return

        # Save if moving average is the best
        if self.eval_ma > self.eval_ma_max:
            self.eval_ma_max = self.eval_ma
            checkpoint_id = (
                f"{self.checkpoint_folder}/model_{gen}_{str(uuid.uuid4())[:8]}"
            )
            self.save(checkpoint_id)
            self.best_checkpoint = checkpoint_id
            self.last_checkpoint = gen

            if self.debug_prints:
                print(f"Checkpointing best model with reward: {self.eval_ma_max}")

        # Load if moving average is below threshold ...
        cond0 = self.eval_ma < (
            self.eval_ma_max / self.load_ratio
            if self.eval_ma < 0
            else self.eval_ma_max * self.load_ratio
        )
        # ... and grace period is met
        cond1 = gen > self.last_checkpoint + self.grace_period * total_gens

        if cond0 and cond1:
            # Try in case of non-existing checkpoint
            try:
                self.load(self.best_checkpoint)
                if self.debug_prints:
                    print(f"Loading best model with reward: {self.eval_ma_max}")
            except Exception as e:
                print(f"Error loading best checkpoint: {e}")

    def train(self, generations, save_folder=None, save_sequence=None):
        # Pre-allocate storage
        evolution = np.zeros(generations)

        # Iterate through generations
        for generation in range(generations):
            self.generation = generation

            # Collect trajectories
            states, actions, log_probs, rewards, dones, truncateds, values = (
                self.collect_trajectories()
            )

            # Compute advantages
            advantages, returns = self.compute_advantages(
                rewards, dones, truncateds, values
            )

            for epoch in range(self.batch_epochs):
                # Shuffle data
                if self.batch_shuffle:
                    if self.seperate_envs_shuffle:
                        # Shuffle along the sequence dimension for each environment
                        for env in range(states.size(1)):
                            env_indices = torch.randperm(states.size(0))

                            # Apply shuffling for each environment independently along the sequence dimension
                            states[:, env, :] = states[env_indices, env, :]
                            actions[:, env] = actions[env_indices, env]
                            log_probs[:, env] = log_probs[env_indices, env]
                            advantages[:, env] = advantages[env_indices, env]
                            returns[:, env] = returns[env_indices, env]
                    else:
                        # Shuffle along the sequence dimension
                        indices = torch.randperm(self.batch_size)

                        states = states[indices]
                        actions = actions[indices]
                        log_probs = log_probs[indices]
                        advantages = advantages[indices]
                        returns = returns[indices]

                # Iterate through batches
                for start in range(0, len(states), self.mini_batch_size):
                    # Get batch
                    end = start + self.mini_batch_size

                    batch_states = states[start:end]
                    batch_actions = actions[start:end]
                    batch_log_probs = log_probs[start:end]
                    batch_advantages = advantages[start:end]
                    batch_returns = returns[start:end]

                    # New policies and values
                    new_log_probs, new_values, entropy = self.forward_pass_train(
                        batch_states, batch_actions
                    )

                    # Check if new_values is a scalar
                    if new_values.shape != batch_returns.shape:
                        new_values = new_values.squeeze(-1)

                    # Policy loss
                    ratios = torch.exp(new_log_probs - batch_log_probs)
                    surr1 = ratios * batch_advantages
                    surr2 = (
                        torch.clamp(ratios, 1 - self.clip_eps, 1 + self.clip_eps)
                        * batch_advantages
                    )
                    policy_loss = -torch.min(surr1, surr2).mean()

                    # Value loss
                    value_loss = (
                        self.value_coef
                        * 0.5
                        * (batch_returns - new_values).pow(2).mean()
                    )

                    # Entropy loss #! WTF
                    # entropies = -torch.sum(
                    #     new_policies * torch.log(new_policies + 1e-8), dim=-1
                    # )
                    # entropy_loss = -self.entropy_coef * entropies.mean()
                    entropy_loss = -self.entropy_coef * entropy

                    # Total loss
                    loss = policy_loss + value_loss + entropy_loss

                    # Backpropagation
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            # Save model
            if save_folder is not None and save_sequence is not None:
                if generation in save_sequence:
                    path = f"{save_folder}/model_{generation}"
                    self.save(path)

            # Evaluate
            eval_reward_w_trunc, eval_reward_wo_trunc = self.evaluate()
            # Store the evaluation reward without truncated rewards ...
            # ... because trunc. rewards are artifical
            evolution[generation] = eval_reward_wo_trunc

            # TODO: Add a logger instead of print
            if self.debug_prints:
                print(
                    f"Generation {generation:>4} - Reward: {eval_reward_w_trunc:8.2f}, w/o trunc.: {eval_reward_wo_trunc:8.2f}"
                )

            # TODO: Checkpointing
            # self.checkpointer(eval_reward_wo_trunc, generation, generations)

            # Parameter scheduler
            self.parameter_scheduler(generation, generations)

        return np.round(evolution, 2)


class PPODiscrete(PPOBase):
    def __init__(self, envs, *args, **kwargs):
        super().__init__(envs, *args, **kwargs)

        self.action_dims = kwargs.get("action_dims", 1)

    def forward_pass_collect(self, states_tensor):
        policies, values = self.network(states_tensor)

        action_dist = torch.distributions.Categorical(policies)
        actions = action_dist.sample()

        log_probs = action_dist.log_prob(actions)

        return actions, log_probs, values

    def forward_pass_train(self, states_tensor, actions_tensor):
        policies, values = self.network(states_tensor)

        action_dist = torch.distributions.Categorical(policies)
        log_probs = action_dist.log_prob(actions_tensor.squeeze(-1))

        # entropy = action_dist.entropy().mean()
        entropy = -torch.sum(policies * torch.log(policies + 1e-8), dim=-1).mean()

        return log_probs, values, entropy

    def eval_actions(self, states_tensor):
        policies, _ = self.network(states_tensor)

        return torch.argmax(policies, dim=-1).detach()


class PPOContinuous(PPOBase):
    def __init__(self, envs, *args, **kwargs):
        super().__init__(envs, *args, **kwargs)

        self.action_dims = kwargs.get("action_dims", None)

        assert self.action_dims != None, "Action dimensions must be provided"

    def forward_pass_collect(self, states_tensor):
        means, log_vars, values = self.network(states_tensor)

        std_devs = torch.exp(0.5 * log_vars)

        action_dist = torch.distributions.Normal(means, std_devs)
        actions = action_dist.rsample()
        bounded_actions = torch.tanh(actions)

        log_probs = (
            action_dist.log_prob(actions) - torch.log(1 - bounded_actions.pow(2) + 1e-8)
        ).sum(dim=-1)

        return bounded_actions.detach(), log_probs, values

    def forward_pass_train(self, states_tensor, actions_tensor):
        means, log_vars, values = self.network(states_tensor)

        std_devs = torch.exp(0.5 * log_vars)

        action_dist = torch.distributions.Normal(means, std_devs)

        log_probs = (
            action_dist.log_prob(actions_tensor)
            - torch.log(1 - actions_tensor.pow(2) + 1e-8)
        ).sum(dim=-1)

        # entropy = action_dist.entropy().sum(dim=-1).mean()
        # entropy = action_dist.entropy().mean()
        entropy = -torch.sum(0.5 * (log_vars + np.log(2 * np.pi * np.e)), dim=-1).mean()

        return log_probs, values, entropy

    def eval_actions(self, states_tensor):
        means, _, _ = self.network(states_tensor)

        means = means.detach()

        return torch.tanh(means)
