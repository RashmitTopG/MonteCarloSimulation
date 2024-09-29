import numpy as np
import pandas as pd
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
lead_time = data['Lead times']

# Function to perform Monte Carlo simulation with log-normal distribution
def monte_carlo_simulation(num_simulations=5000):
    def get_lognorm_params(data):
        data = data[data > 0]  # Avoid log of zero
        mu = np.mean(np.log(data))
        sigma = np.std(np.log(data))
        return mu, sigma

    mu_price, sigma_price = get_lognorm_params(price)
    mu_products_sold, sigma_products_sold = get_lognorm_params(products_sold)
    mu_defect_rate, sigma_defect_rate = get_lognorm_params(defect_rate)
    mu_shipping_costs, sigma_shipping_costs = get_lognorm_params(shipping_costs)
    mu_manufacturing_costs, sigma_manufacturing_costs = get_lognorm_params(manufacturing_costs)
    mu_lead_time, sigma_lead_time = get_lognorm_params(lead_time)

    simulated_prices = np.random.lognormal(mu_price, sigma_price, num_simulations)
    simulated_products_sold = np.random.lognormal(mu_products_sold, sigma_products_sold, num_simulations)
    simulated_defect_rates = np.random.lognormal(mu_defect_rate, sigma_defect_rate, num_simulations)
    simulated_shipping_costs = np.random.lognormal(mu_shipping_costs, sigma_shipping_costs, num_simulations)
    simulated_manufacturing_costs = np.random.lognormal(mu_manufacturing_costs, sigma_manufacturing_costs, num_simulations)
    simulated_lead_times = np.random.lognormal(mu_lead_time, sigma_lead_time, num_simulations)

    adjusted_sales = simulated_products_sold * (1 - simulated_defect_rates)
    revenues = np.maximum(0, (simulated_prices * adjusted_sales) - (simulated_shipping_costs + simulated_manufacturing_costs + simulated_lead_times * 0.05))

    return revenues, simulated_lead_times * 0.05, simulated_defect_rates, simulated_shipping_costs, simulated_manufacturing_costs

# Create the Dash app
app = Dash(__name__)

# Layout of the dashboard
app.layout = html.Div(
    style={
        'fontFamily': 'Arial',
        'padding': '20px',
        'backgroundColor': '#f0f0f0'
    },
    children=[
        html.H1(
            "Monte Carlo Simulation Dashboard for Supply Chain Risks",
            style={
                'textAlign': 'center',
                'color': '#003366',
                'marginBottom': '30px',
                'paddingTop': '20px'
            }
        ),
        html.Div(
            [
                html.Label(
                    "Number of Simulations:",
                    style={
                        'fontSize': '18px',
                        'fontWeight': 'bold',
                        'marginBottom': '10px'
                    }
                ),
                dcc.Slider(
                    id="sim-slider",
                    min=1000,
                    max=20000,
                    step=1000,
                    value=5000,
                    marks={i: {'label': f'{i}', 'style': {'fontSize': '14px'}} for i in range(1000, 21000, 2000)},
                    tooltip={'placement': 'bottom', 'always_visible': True}
                )
            ],
            style={'textAlign': 'center', 'marginBottom': '30px'}
        ),
        html.Div(
            id="simulation-results",
            style={
                'fontSize': '16px',
                'textAlign': 'center',
                'marginBottom': '30px',
                'fontWeight': 'bold',
                'backgroundColor': '#e8f4f8',
                'borderRadius': '10px',
                'padding': '20px',
                'boxShadow': '0 4px 8px rgba(0,0,0,0.1)'
            }
        ),
        html.Div(
            [
                html.Div(dcc.Graph(id="simulation-graph"), style={'width': '48%', 'display': 'inline-block'}),
                html.Div(dcc.Graph(id="lead-time-risk-graph"), style={'width': '48%', 'display': 'inline-block'})
            ],
            style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '30px'}
        ),
        html.Div(
            [
                html.Div(dcc.Graph(id="defect-rate-risk-graph"), style={'width': '48%', 'display': 'inline-block'}),
                html.Div(dcc.Graph(id="shipping-cost-risk-graph"), style={'width': '48%', 'display': 'inline-block'})
            ],
            style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '30px'}
        ),
        html.Div(
            [
                html.Div(dcc.Graph(id="manufacturing-cost-risk-graph"), style={'width': '48%', 'display': 'inline-block'}),
                html.Div(dcc.Graph(id="box-plot-graph"), style={'width': '48%', 'display': 'inline-block'})
            ],
            style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '30px'}
        )
    ]
)

# Callback to update the simulation results and graph dynamically
@app.callback(
    [Output('simulation-results', 'children'),
     Output('simulation-graph', 'figure'),
     Output('lead-time-risk-graph', 'figure'),
     Output('defect-rate-risk-graph', 'figure'),
     Output('shipping-cost-risk-graph', 'figure'),
     Output('manufacturing-cost-risk-graph', 'figure'),
     Output('box-plot-graph', 'figure')],
    [Input('sim-slider', 'value')]
)
def update_graphs(num_simulations):
    revenues, lead_time_risks, defect_rate_risks, shipping_cost_risks, manufacturing_cost_risks = monte_carlo_simulation(num_simulations)

    # Summary statistics
    stats = {
        'mean_revenue': np.mean(revenues),
        'std_revenue': np.std(revenues),
        'percentile_5': np.percentile(revenues, 5),
        'percentile_95': np.percentile(revenues, 95)
    }

    # Create graphs
    fig_simulation = px.histogram(revenues, nbins=100, title="Revenue Distribution").update_layout(xaxis_title='Revenue', yaxis_title='Frequency')

    fig_lead_time_risk = px.histogram(lead_time_risks, nbins=100, title="Lead Time Risk Distribution").update_layout(xaxis_title='Lead Time Risk', yaxis_title='Frequency')

    fig_defect_rate_risk = px.histogram(defect_rate_risks, nbins=100, title="Defect Rate Risk Distribution").update_layout(xaxis_title='Defect Rate Risk', yaxis_title='Frequency')

    fig_shipping_cost_risk = px.histogram(shipping_cost_risks, nbins=100, title="Shipping Cost Risk Distribution").update_layout(xaxis_title='Shipping Cost Risk', yaxis_title='Frequency')

    fig_manufacturing_cost_risk = px.histogram(manufacturing_cost_risks, nbins=100, title="Manufacturing Cost Risk Distribution").update_layout(xaxis_title='Manufacturing Cost Risk', yaxis_title='Frequency')

    fig_box_plot = px.box(pd.DataFrame({
        'Revenue': revenues,
        'Lead Time Risk': lead_time_risks,
        'Defect Rate Risk': defect_rate_risks,
        'Shipping Cost Risk': shipping_cost_risks,
        'Manufacturing Cost Risk': manufacturing_cost_risks
    }), title='Box Plot of Risks').update_layout(template='plotly_dark')

    return (
        f"Mean Revenue: {stats['mean_revenue']:.2f}, Std Dev: {stats['std_revenue']:.2f}, 5th Percentile: {stats['percentile_5']:.2f}, 95th Percentile: {stats['percentile_95']:.2f}",
        fig_simulation,
        fig_lead_time_risk,
        fig_defect_rate_risk,
        fig_shipping_cost_risk,
        fig_manufacturing_cost_risk,
        fig_box_plot
    )

if __name__ == '__main__':
    app.run_server(debug=True)
