import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dash import Dash, dcc, html, Input, Output
import plotly.express as px

# Load dataset
data = pd.read_csv('C:/Users/Rashmit Mhatre/Downloads/archive564/supply_chain_data.csv')

# Key variables
price = data['Price']
products_sold = data['Number of products sold']
defect_rate = data['Defect rates']
shipping_costs = data['Shipping costs']
manufacturing_costs = data['Manufacturing costs']

# Function to run Monte Carlo Simulation
def monte_carlo_simulation(num_simulations=1000):
    revenues = []
    for _ in range(num_simulations):
        # Randomly sample values based on normal distribution
        simulated_price = np.random.normal(np.mean(price), np.std(price))
        simulated_products_sold = np.random.normal(np.mean(products_sold), np.std(products_sold))
        simulated_defect_rate = np.random.normal(np.mean(defect_rate), np.std(defect_rate))
        simulated_shipping_costs = np.random.normal(np.mean(shipping_costs), np.std(shipping_costs))
        simulated_manufacturing_costs = np.random.normal(np.mean(manufacturing_costs), np.std(manufacturing_costs))
        
        # Calculate revenue
        adjusted_sales = simulated_products_sold * (1 - simulated_defect_rate)
        revenue = (simulated_price * adjusted_sales) - (simulated_shipping_costs + simulated_manufacturing_costs)
        
        revenues.append(revenue)
    
    revenues = np.array(revenues)
    return revenues

# Initialize Dash app
app = Dash(__name__)

# Layout of the Dashboard
app.layout = html.Div([
    html.H1("Monte Carlo Simulation for Supply Chain Revenue"),
    
    # Input: Slider to adjust number of simulations
    html.Label("Number of Simulations:"),
    dcc.Slider(id="num-sim-slider", min=100, max=5000, step=100, value=1000,
               marks={i: f'{i}' for i in range(500, 5500, 500)}),
    
    # Output: Graph for displaying simulation results
    dcc.Graph(id="revenue-distribution"),
    
    # Output: Display summary statistics (mean, percentiles)
    html.Div(id="stats-output")
])

# Callback to update the dashboard based on slider value
@app.callback(
    Output("revenue-distribution", "figure"),
    Output("stats-output", "children"),
    Input("num-sim-slider", "value")
)
def update_simulation(num_simulations):
    # Perform Monte Carlo simulation
    revenues = monte_carlo_simulation(num_simulations)
    
    # Calculate statistics
    mean_revenue = np.mean(revenues)
    std_revenue = np.std(revenues)
    p_5 = np.percentile(revenues, 5)
    p_95 = np.percentile(revenues, 95)
    
    # Create a histogram of the results using Plotly
    fig = px.histogram(revenues, nbins=50, title="Distribution of Simulated Revenues",
                       labels={'value': 'Revenue', 'count': 'Frequency'})
    
    # Display statistics
    stats = f"Mean Revenue: {mean_revenue:.2f} | Std Dev: {std_revenue:.2f} | 5th Percentile: {p_5:.2f} | 95th Percentile: {p_95:.2f}"
    
    return fig, stats

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
