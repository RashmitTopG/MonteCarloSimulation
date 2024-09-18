import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Parameters
num_simulations = 1000
inventory_initial = 100
demand_mean = 20
demand_std = 5
lead_time_mean = 7
lead_time_std = 2

# Simulation Function
def run_simulation():
    inventory = inventory_initial
    inventory_levels = []
    for _ in range(num_simulations):
        # Random demand and lead time
        demand = np.random.normal(demand_mean, demand_std)
        lead_time = np.random.normal(lead_time_mean, lead_time_std)
        
        # Update inventory
        inventory -= demand
        if inventory < 0:
            inventory = 0
        inventory_levels.append(inventory)
        
        # Replenish after lead time (simplified model)
        inventory += np.random.normal(demand_mean, demand_std)  # Supply replenishment
    return inventory_levels

# Run simulations
results = [run_simulation() for _ in range(num_simulations)]

# Analyze Results
results_df = pd.DataFrame(results).mean(axis=0)

# Plot Results
plt.figure(figsize=(12, 6))
plt.plot(results_df)
plt.title('Inventory Levels over Simulations')
plt.xlabel('Time')
plt.ylabel('Inventory Level')
plt.grid(True)
plt.show()
