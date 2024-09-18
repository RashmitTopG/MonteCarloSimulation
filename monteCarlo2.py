import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dash import Dash, html, dcc, Input, Output
import plotly.express as px

# Load dataset
data = pd.read_csv('C:/Users/Rashmit Mhatre/Downloads/archive564/supply_chain_data.csv')

# Key variables
price = data['Price']
products_sold = data['Number of products sold']
defect_rate = data['Defect rates']
shipping_costs = data['Shipping costs']
manufacturing_costs = data['Manufacturing costs']

# Function to perform Monte Carlo simulation
def monte_carlo_simulation(num_simulations=1000):
    revenues = []
    
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

    return np.array(revenues)

# Create the Dash app
app = Dash(__name__)

# Layout of the dashboard
app.layout = html.Div([
    html.H1("Monte Carlo Simulation Dashboard for Supply Chain Risks"),
    
    # Slider for selecting number of simulations
    html.Label("Number of Simulations:"),
    dcc.Slider(id="sim-slider", min=500, max=5000, step=500, value=1000,
               marks={i: f'{i}' for i in range(500, 5500, 500)}),
    
    # Display results (mean, std, percentiles)
    html.Div(id="simulation-results"),
    
    # Graph to display histogram of results
    dcc.Graph(id="simulation-graph"),
])

# Callback to update the simulation results and graph dynamically
@app.callback(
    [Output('simulation-results', 'children'),
     Output('simulation-graph', 'figure')],
    [Input('sim-slider', 'value')]
)
def update_simulation(num_simulations):
    # Perform simulation
    revenues = monte_carlo_simulation(num_simulations)
    
    # Calculate statistics
    mean_revenue = np.mean(revenues)
    std_revenue = np.std(revenues)
    percentile_5 = np.percentile(revenues, 5)
    percentile_95 = np.percentile(revenues, 95)
    
    # Update results display
    result_text = [
        f"Mean Revenue: {mean_revenue:.2f}",
        f"Standard Deviation of Revenue: {std_revenue:.2f}",
        f"5th Percentile of Revenue: {percentile_5:.2f}",
        f"95th Percentile of Revenue: {percentile_95:.2f}"
    ]
    
    # Create histogram for simulation results
    fig = px.histogram(revenues, nbins=50, title='Distribution of Simulated Revenues')
    fig.update_layout(xaxis_title='Revenue', yaxis_title='Frequency')
    
    return result_text, fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
