import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv('C:/Users/Rashmit Mhatre/Downloads/archive564/supply_chain_data.csv')

# Key variables
price = data['Price']
products_sold = data['Number of products sold']
defect_rate = data['Defect rates']
shipping_costs = data['Shipping costs']
manufacturing_costs = data['Manufacturing costs']

# Number of simulations
num_simulations = 1000

# Arrays to store results of simulations
revenues = []

# Perform Monte Carlo simulation
for _ in range(num_simulations):
    # Randomly sample values based on normal distribution around mean
    simulated_price = np.random.normal(np.mean(price), np.std(price))
    simulated_products_sold = np.random.normal(np.mean(products_sold), np.std(products_sold))
    simulated_defect_rate = np.random.normal(np.mean(defect_rate), np.std(defect_rate))
    simulated_shipping_costs = np.random.normal(np.mean(shipping_costs), np.std(shipping_costs))
    simulated_manufacturing_costs = np.random.normal(np.mean(manufacturing_costs), np.std(manufacturing_costs))
    
    # Calculate revenue taking defect rate into account
    adjusted_sales = simulated_products_sold * (1 - simulated_defect_rate)
    revenue = (simulated_price * adjusted_sales) - (simulated_shipping_costs + simulated_manufacturing_costs)
    
    revenues.append(revenue)

# Convert to a NumPy array for analysis
revenues = np.array(revenues)

# Analyze Results
mean_revenue = np.mean(revenues)
std_revenue = np.std(revenues)
percentile_5 = np.percentile(revenues, 5)
percentile_95 = np.percentile(revenues, 95)

# Print results
print(f"Mean Revenue: {mean_revenue}")
print(f"Standard Deviation of Revenue: {std_revenue}")
print(f"5th Percentile of Revenue: {percentile_5}")
print(f"95th Percentile of Revenue: {percentile_95}")

# Plot histogram of simulation results
plt.hist(revenues, bins=50, color='blue', edgecolor='black')
plt.title('Monte Carlo Simulation of Supply Chain Revenue')
plt.xlabel('Revenue')
plt.ylabel('Frequency')
plt.show()
