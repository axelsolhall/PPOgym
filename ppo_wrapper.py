import torch
from torch import nn, optim
import numpy as np


class PPOWrapper:
    def __init__(self, envs, network, *args, **kwargs):
        # Environments vector
        self.envs = envs
        self.num_envs = kwargs.get("num_envs", 1)

        # Network
        self.network = network
        self.lr = kwargs.get("lr", None)
        self.final_lr = kwargs.get("final_lr", None)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)

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

        # Batch parameters
        self.batch_size = kwargs.get("batch_size", None)
        self.batch_epochs = kwargs.get("batch_epochs", None)
        self.batch_shuffle = kwargs.get("batch_shuffle", False)
        self.iterations = kwargs.get("iterations", None)

        assert self.batch_size != None, "Batch size must be provided"
        assert self.batch_epochs != None, "Batch epochs must be provided"
        assert self.iterations != None, "Iterations must be provided"

        # Miscellaneous
        self.truncated_reward = kwargs.get("truncated_reward", 0)
        self.checkpointing = kwargs.get("checkpointing", False)
        self.debug_prints = kwargs.get("debug_prints", False)

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
            if self.initial_clip_eps is not None:
                self.initial_clip_eps = self.clip_eps

            alpha = gen / total_gens
            self.clip_eps = self.initial_clip_eps + alpha * (
                self.final_clip_eps - self.initial_clip_eps
            )

            if self.debug_prints:
                print(f"Current clip epsilon: {self.clip_eps}")

        # Linearly anneal the entropy coefficient
        if self.final_entropy_coef is not None:
            if self.initial_entropy_coef is not None:
                self.initial_entropy_coef = self.entropy_coef

            alpha = gen / total_gens
            self.entropy_coef = self.initial_entropy_coef + alpha * (
                self.final_entropy_coef - self.initial_entropy_coef
            )

            if self.debug_prints:
                print(f"Current entropy coefficient: {self.entropy_coef}")

    def collect_trajectories(self):

        # Pre-allocate storage
        states = torch.zeros(self.iterations, self.num_envs, self.network.input_dims)
        actions = torch.zeros(self.iterations, self.num_envs)
        log_probs = torch.zeros(self.iterations, self.num_envs)
        rewards = torch.zeros(self.iterations, self.num_envs)
        dones = torch.zeros(self.iterations, self.num_envs)
        truncateds = torch.zeros(self.iterations, self.num_envs)
        values = torch.zeros(self.iterations, self.num_envs)

        # Reset environments
        current_states, infos = self.envs.reset()

        # Iterate through iterations

        for itr in range(self.iterations):

            states_tensor = torch.tensor(current_states, dtype=torch.float32)

            policies, current_values = self.network(states_tensor)

            action_dist = torch.distributions.Categorical(policies)
            current_actions = action_dist.sample()

            next_states, current_rewards, current_dones, current_truncateds, infos = (
                self.envs.step(current_actions.numpy())
            )

            if current_truncateds.any():
                current_rewards += current_truncateds * self.truncated_reward
                current_dones = current_dones | current_truncateds

            current_log_probs = action_dist.log_prob(current_actions)

            current_states = next_states

            # Store data
            states[itr] = states_tensor
            actions[itr] = current_actions
            log_probs[itr] = current_log_probs.detach()
            rewards[itr] = torch.tensor(current_rewards, dtype=torch.float32)
            dones[itr] = torch.tensor(current_dones, dtype=torch.float32)
            truncateds[itr] = torch.tensor(current_truncateds, dtype=torch.float32)
            values[itr] = current_values.squeeze(-1).detach()

        return states, actions, log_probs, rewards, dones, truncateds, values

    def compute_advantages(self, rewards, dones, truncateds, values):
        advantages = torch.zeros_like(rewards, dtype=torch.float32)
        returns = torch.zeros_like(rewards, dtype=torch.float32)
        gaes = torch.zeros(self.num_envs, dtype=torch.float32)

        values = torch.cat((values, torch.zeros((1, self.num_envs))), dim=0)

        for ti in reversed(range(rewards.size(0))):

            next_values = values[ti + 1]
            # mask = torch.where(
            #     (dones[ti] + truncateds[ti]) > 0, torch.tensor(0.0), torch.tensor(1.0)
            # )
            # next_values = next_values * (1 - mask)
            # next_values[dones[ti] + truncateds[ti]] = 0.0
            # next_values[dones[ti].numpy()] = 0.0

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
        scores = np.zeros(self.num_envs)
        alive = np.ones(self.num_envs)

        current_states, infos = self.envs.reset()

        for itr in range(self.iterations):
            states_tensor = torch.tensor(current_states, dtype=torch.float32)
            policies, _ = self.network(states_tensor)

            actions = torch.argmax(policies, dim=-1)

            next_states, current_rewards, current_dones, current_truncateds, infos = (
                self.envs.step(actions.numpy())
            )

            if current_truncateds.any():
                current_rewards += current_truncateds * self.truncated_reward
                current_dones = current_dones | current_truncateds

            if not alive.any():
                break

            current_states = next_states

            scores += current_rewards * alive
            # alive = alive * (1 - current_dones)
            alive[current_dones] = 0

        return np.mean(scores)

    def train(self, generations):
        evolution = np.zeros(generations)
        for generation in range(generations):

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
                    indices = torch.randperm(self.iterations)

                    states = states[indices]
                    actions = actions[indices]
                    log_probs = log_probs[indices]
                    advantages = advantages[indices]
                    returns = returns[indices]

                # Iterate through batches
                for start in range(0, len(states), self.batch_size):
                    end = start + self.batch_size

                    batch_states = states[start:end]
                    batch_actions = actions[start:end]
                    batch_log_probs = log_probs[start:end]
                    batch_advantages = advantages[start:end]
                    batch_returns = returns[start:end]

                    # New policies and values
                    new_policies, new_values = self.network(batch_states)
                    new_action_dist = torch.distributions.Categorical(new_policies)
                    new_log_probs = new_action_dist.log_prob(batch_actions)

                    if new_values.shape != batch_returns.shape:
                        new_values = new_values.squeeze(-1)

                    # Compute ratios
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

                    # Entropy loss
                    entropies = -torch.sum(
                        new_policies * torch.log(new_policies + 1e-8), dim=-1
                    )
                    entropy_loss = -self.entropy_coef * entropies.mean()

                    # Total loss
                    loss = policy_loss + value_loss + entropy_loss

                    # Backpropagation
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            # Evaluate
            eval_reward = self.evaluate()
            evolution[generation] = eval_reward
            if self.debug_prints:
                print(f"Generation {generation} - Reward: {eval_reward}")

            # Parameter scheduler
            self.parameter_scheduler(generation, generations)

        return np.round(evolution, 2)
