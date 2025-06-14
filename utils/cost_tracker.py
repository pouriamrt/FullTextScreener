import time
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

class APICostTracker:
    # OpenAI pricing in USD per 1K tokens (update as needed)
    PRICING = {
        "gpt-4":       {"prompt": 0.03,  "completion": 0.06},
        "gpt-4-0613":  {"prompt": 0.03,  "completion": 0.06},
        "gpt-4-1106-preview": {"prompt": 0.01, "completion": 0.03},
        "gpt-4o":      {"prompt": 0.005, "completion": 0.015},
        "gpt-4.1-mini": {"prompt": 0.002, "completion": 0.006},  # estimated
        "gpt-3.5-turbo": {"prompt": 0.001, "completion": 0.002},
    }

    def __init__(self, model="gpt-4.1-mini"):
        self.model = model
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost = 0.0
        self.calls = 0
        self.start_time = time.time()
        
        # Track history for plotting
        self.token_history = []
        self.cost_history = []
        self.cumulative_cost = []
        self.timestamps = []

    def update_from_usage(self, usage: dict):
        """Update tracker using OpenAI API response['usage'] dictionary."""
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        self.calls += 1
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens

        pricing = self.PRICING.get(self.model, {"prompt": 0.0, "completion": 0.0})
        p_cost = (prompt_tokens / 1000) * pricing["prompt"]
        c_cost = (completion_tokens / 1000) * pricing["completion"]
        current_cost = p_cost + c_cost
        self.total_cost += current_cost
        
        # Update history
        self.token_history.append(prompt_tokens + completion_tokens)
        self.cost_history.append(current_cost)
        self.cumulative_cost.append(self.total_cost)
        self.timestamps.append(datetime.now())

    def plot_usage(self, save_path=None):
        """Plot token usage and cumulative cost over time."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot tokens
        ax1.plot(self.timestamps, self.token_history, 'b-', label='Tokens per call')
        ax1.set_ylabel('Tokens')
        ax1.set_title('Token Usage Over Time')
        ax1.grid(True)
        
        # Plot cumulative cost
        ax2.plot(self.timestamps, self.cumulative_cost, 'r-', label='Cumulative Cost')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Cost (USD)')
        ax2.set_title('Cumulative Cost Over Time')
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def report(self):
        elapsed = time.time() - self.start_time
        print(f"\n🔎 API Usage Report ({self.model})")
        print(f"Calls made         : {self.calls}")
        print(f"Prompt tokens used : {self.total_prompt_tokens}")
        print(f"Completion tokens  : {self.total_completion_tokens}")
        print(f"Total tokens       : {self.total_prompt_tokens + self.total_completion_tokens}")
        print(f"Total cost (USD)   : ${self.total_cost:.5f}")
        print(f"Elapsed time       : {elapsed:.2f} seconds\n")
        
        # Generate usage plot
        self.plot_usage()
