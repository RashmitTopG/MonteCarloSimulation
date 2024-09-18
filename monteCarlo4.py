import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dash import Dash, html, dcc, Input, Output
import plotly.express as px

# Load dataset
data = pd.read_csv('C:/Users/Rashmit Mhatre/Downloads/archive564/supply_chain_data.csv')

# Key variables for risk analysis
price = data['Price']
products_sold = data['Number of products sold']
defect_rate = data['Defect rates']
shipping_costs = data['Shipping costs']
manufacturing_costs = data['Manufacturing costs']
lead_time = data['Lead times']  # Added lead time for variability analysis

# Function to perform Monte Carlo simulation
def monte_carlo_simulation(num_simulations=1000):
    revenues = []
    total_costs = []
    lead_time_risks = []
    defect_rate_risks = []
    shipping_cost_risks = []
    manufacturing_cost_risks = []

    for _ in range(num_simulations):
        # Simulate values based on normal distribution around mean and std deviation
        simulated_price = np.random.normal(np.mean(price), np.std(price))
        simulated_products_sold = np.random.normal(np.mean(products_sold), np.std(products_sold))
        simulated_defect_rate = np.random.normal(np.mean(defect_rate), np.std(defect_rate))
        simulated_shipping_costs = np.random.normal(np.mean(shipping_costs), np.std(shipping_costs))
        simulated_manufacturing_costs = np.random.normal(np.mean(manufacturing_costs), np.std(manufacturing_costs))
        simulated_lead_time = np.random.normal(np.mean(lead_time), np.std(lead_time))  # Simulate lead time

        # Calculate revenue taking defect rate into account
        adjusted_sales = simulated_products_sold * (1 - simulated_defect_rate)
        revenue = (simulated_price * adjusted_sales)
        total_cost = simulated_shipping_costs + simulated_manufacturing_costs
        
        # Lead time risk calculation (higher lead times may incur additional penalties)
        lead_time_risk = simulated_lead_time * 0.05  # Hypothetical risk factor for lead time delays
        
        # Append to lists
        revenues.append(revenue - total_cost - lead_time_risk)
        total_costs.append(total_cost)
        lead_time_risks.append(lead_time_risk)
        defect_rate_risks.append(simulated_defect_rate)
        shipping_cost_risks.append(simulated_shipping_costs)
        manufacturing_cost_risks.append(simulated_manufacturing_costs)

    return (np.array(revenues), np.array(lead_time_risks), np.array(defect_rate_risks),
            np.array(shipping_cost_risks), np.array(manufacturing_cost_risks))

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
    
    # Graph to display histogram of revenue results
    dcc.Graph(id="simulation-graph"),
    
    # Graphs to display risks
    dcc.Graph(id="lead-time-risk-graph"),
    dcc.Graph(id="defect-rate-risk-graph"),
    dcc.Graph(id="shipping-cost-risk-graph"),
    dcc.Graph(id="manufacturing-cost-risk-graph"),
])

# Callback to update the simulation results and graph dynamically
@app.callback(
    [Output('simulation-results', 'children'),
     Output('simulation-graph', 'figure'),
     Output('lead-time-risk-graph', 'figure'),
     Output('defect-rate-risk-graph', 'figure'),
     Output('shipping-cost-risk-graph', 'figure'),
     Output('manufacturing-cost-risk-graph', 'figure')],
    [Input('sim-slider', 'value')]
)
def update_simulation(num_simulations):
    # Perform simulation
    revenues, lead_time_risks, defect_rate_risks, shipping_cost_risks, manufacturing_cost_risks = monte_carlo_simulation(num_simulations)
    
    # Calculate statistics for revenues
    mean_revenue = np.mean(revenues)
    std_revenue = np.std(revenues)
    percentile_5 = np.percentile(revenues, 5)
    percentile_95 = np.percentile(revenues, 95)

    # Calculate statistics for lead time risks
    mean_lead_time_risk = np.mean(lead_time_risks)
    percentile_5_lead_time = np.percentile(lead_time_risks, 5)
    percentile_95_lead_time = np.percentile(lead_time_risks, 95)

    # Update results display
    result_text = [
        f"Mean Revenue: {mean_revenue:.2f}",
        f"Standard Deviation of Revenue: {std_revenue:.2f}",
        f"5th Percentile of Revenue: {percentile_5:.2f}",
        f"95th Percentile of Revenue: {percentile_95:.2f}",
        f"Mean Lead Time Risk: {mean_lead_time_risk:.2f}",
        f"5th Percentile of Lead Time Risk: {percentile_5_lead_time:.2f}",
        f"95th Percentile of Lead Time Risk: {percentile_95_lead_time:.2f}"
    ]
    
    # Create histogram for revenue simulation results
    fig_revenue = px.histogram(revenues, nbins=50, title='Distribution of Simulated Revenues')
    fig_revenue.update_layout(xaxis_title='Revenue', yaxis_title='Frequency')

    # Create histogram for lead time risk results
    fig_lead_time_risk = px.histogram(lead_time_risks, nbins=50, title='Distribution of Lead Time Risks')
    fig_lead_time_risk.update_layout(xaxis_title='Lead Time Risk', yaxis_title='Frequency')

    # Create histogram for defect rate risk results
    fig_defect_rate_risk = px.histogram(defect_rate_risks, nbins=50, title='Distribution of Defect Rate Risks')
    fig_defect_rate_risk.update_layout(xaxis_title='Defect Rate', yaxis_title='Frequency')

    # Create histogram for shipping cost risk results
    fig_shipping_cost_risk = px.histogram(shipping_cost_risks, nbins=50, title='Distribution of Shipping Cost Risks')
    fig_shipping_cost_risk.update_layout(xaxis_title='Shipping Cost', yaxis_title='Frequency')

    # Create histogram for manufacturing cost risk results
    fig_manufacturing_cost_risk = px.histogram(manufacturing_cost_risks, nbins=50, title='Distribution of Manufacturing Cost Risks')
    fig_manufacturing_cost_risk.update_layout(xaxis_title='Manufacturing Cost', yaxis_title='Frequency')
    
    return result_text, fig_revenue, fig_lead_time_risk, fig_defect_rate_risk, fig_shipping_cost_risk, fig_manufacturing_cost_risk

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
