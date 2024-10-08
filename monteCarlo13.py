import numpy as np
import pandas as pd
from dash import Dash, html, dcc, Input, Output
import plotly.express as px
import plotly.graph_objects as go

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

    return (revenues, simulated_lead_times * 0.05, simulated_defect_rates,
            simulated_shipping_costs, simulated_manufacturing_costs)

# Create the Dash app
app = Dash(__name__)

# Layout of the dashboard with enhanced styling
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
                    min=1000,  # Set minimum to 1000
                    max=20000,  # Set maximum to 20000
                    step=1000,  # Step remains the same
                    value=5000,  # Default value is now 5000
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
        ),
        html.Div(
            [
                html.Div(dcc.Graph(id="violin-plot-graph"), style={'width': '48%', 'display': 'inline-block'}),
                html.Div(dcc.Graph(id="scatter-plot-graph"), style={'width': '48%', 'display': 'inline-block'})
            ],
            style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '30px'}
        ),
        html.Div(
            [
                html.Div(dcc.Graph(id="heatmap-graph"), style={'width': '48%', 'display': 'inline-block'}),
                html.Div(dcc.Graph(id="pair-plot-graph"), style={'width': '48%', 'display': 'inline-block'})
            ],
            style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '30px'}
        ),
        html.Div(
            [
                html.Div(dcc.Graph(id="cdf-plot-graph"), style={'width': '48%', 'display': 'inline-block'}),
                html.Div(dcc.Graph(id="bar-chart-graph"), style={'width': '48%', 'display': 'inline-block'})
            ],
            style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '30px'}
        ),
        html.Div(
            [
                html.Div(dcc.Graph(id="defect-rate-histogram"), style={'width': '48%', 'display': 'inline-block'}),
                html.Div(dcc.Graph(id="pie-chart-risk-factors"), style={'width': '48%', 'display': 'inline-block'})
            ],
            style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '30px'}
        ),
        html.Div(
            [
                html.Div(dcc.Graph(id="area-chart-revenue"), style={'width': '48%', 'display': 'inline-block'}),
                html.Div(dcc.Graph(id="line-chart-lead-time-risk"), style={'width': '48%', 'display': 'inline-block'})
            ],
            style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '30px'}
        ),
        html.Div(
            [
                html.Div(dcc.Graph(id="major-risk-factor-bar-chart"), style={'width': '48%', 'display': 'inline-block'})
            ],
            style={'textAlign': 'center', 'marginBottom': '30px'}
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
     Output('box-plot-graph', 'figure'),
     Output('violin-plot-graph', 'figure'),
     Output('scatter-plot-graph', 'figure'),
     Output('heatmap-graph', 'figure'),
     Output('pair-plot-graph', 'figure'),
     Output('cdf-plot-graph', 'figure'),
     Output('bar-chart-graph', 'figure'),
     Output('defect-rate-histogram', 'figure'),
     Output('pie-chart-risk-factors', 'figure'),
     Output('area-chart-revenue', 'figure'),
     Output('line-chart-lead-time-risk', 'figure'),
     Output('major-risk-factor-bar-chart', 'figure')],
    [Input('sim-slider', 'value')]
)
def update_graphs(num_simulations):
    revenues, lead_time_risks, defect_rate_risks, shipping_cost_risks, manufacturing_cost_risks = monte_carlo_simulation(num_simulations)

    # Summary statistics
    stats = {
        'mean_revenue': np.mean(revenues),
        'std_revenue': np.std(revenues),
        'percentile_5': np.percentile(revenues, 5),
        'percentile_95': np.percentile(revenues, 95),
        'mean_lead_time_risk': np.mean(lead_time_risks),
        'percentile_5_lead_time': np.percentile(lead_time_risks, 5),
        'percentile_95_lead_time': np.percentile(lead_time_risks, 95),
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

    fig_violin_plot = px.violin(pd.DataFrame({
        'Revenue': revenues,
        'Lead Time Risk': lead_time_risks,
        'Defect Rate Risk': defect_rate_risks,
        'Shipping Cost Risk': shipping_cost_risks,
        'Manufacturing Cost Risk': manufacturing_cost_risks
    }), title='Violin Plot of Risks').update_layout(template='plotly_dark')

    fig_scatter_plot = px.scatter(x=revenues, y=lead_time_risks, title="Scatter Plot of Revenue vs Lead Time Risk").update_layout(xaxis_title='Revenue', yaxis_title='Lead Time Risk')

    fig_heatmap = px.density_heatmap(pd.DataFrame({
        'Revenue': revenues,
        'Lead Time Risk': lead_time_risks,
        'Defect Rate Risk': defect_rate_risks,
        'Shipping Cost Risk': shipping_cost_risks,
        'Manufacturing Cost Risk': manufacturing_cost_risks
    }), title='Heatmap of Risks').update_layout(template='plotly_dark')

    fig_pair_plot = px.scatter_matrix(pd.DataFrame({
        'Revenue': revenues,
        'Lead Time Risk': lead_time_risks,
        'Defect Rate Risk': defect_rate_risks,
        'Shipping Cost Risk': shipping_cost_risks,
        'Manufacturing Cost Risk': manufacturing_cost_risks
    }), title='Pair Plot of Risks').update_layout(template='plotly_dark')

    fig_cdf_plot = px.ecdf(revenues, title="Cumulative Distribution Function (CDF) of Revenue").update_layout(xaxis_title='Revenue', yaxis_title='Cumulative Probability')

    fig_bar_chart = px.bar(pd.DataFrame({
        'Risk Factor': ['Lead Time', 'Defect Rate', 'Shipping Cost', 'Manufacturing Cost'],
        'Average Impact': [np.mean(lead_time_risks), np.mean(defect_rate_risks), np.mean(shipping_cost_risks), np.mean(manufacturing_cost_risks)]
    }), x='Risk Factor', y='Average Impact', title='Bar Chart of Average Impact of Risk Factors').update_layout(template='plotly_dark')

    fig_defect_rate_histogram = px.histogram(defect_rate_risks, nbins=100, title="Defect Rate Risk Histogram").update_layout(xaxis_title='Defect Rate Risk', yaxis_title='Frequency')

    fig_pie_chart_risk_factors = px.pie(pd.DataFrame({
        'Risk Factor': ['Lead Time', 'Defect Rate', 'Shipping Cost', 'Manufacturing Cost'],
        'Average Impact': [np.mean(lead_time_risks), np.mean(defect_rate_risks), np.mean(shipping_cost_risks), np.mean(manufacturing_cost_risks)]
    }), names='Risk Factor', values='Average Impact', title='Pie Chart of Risk Factors').update_layout(template='plotly_dark')

    fig_area_chart_revenue = px.area(revenues, title="Area Chart of Revenue").update_layout(xaxis_title='Simulation', yaxis_title='Revenue')

    fig_line_chart_lead_time_risk = px.line(lead_time_risks, title="Line Chart of Lead Time Risk").update_layout(xaxis_title='Simulation', yaxis_title='Lead Time Risk')

    fig_major_risk_factor = px.bar(pd.DataFrame({
        'Risk Factor': ['Lead Time', 'Defect Rate', 'Shipping Cost', 'Manufacturing Cost'],
        'Average Impact': [np.mean(lead_time_risks), np.mean(defect_rate_risks), np.mean(shipping_cost_risks), np.mean(manufacturing_cost_risks)]
    }), x='Risk Factor', y='Average Impact', title='Major Risk Factors').update_layout(template='plotly_dark')

    # Simulation summary
    simulation_summary = f"""
    Simulation Summary:
    Mean Revenue: {stats['mean_revenue']:.2f}
    Standard Deviation of Revenue: {stats['std_revenue']:.2f}
    5th Percentile Revenue: {stats['percentile_5']:.2f}
    95th Percentile Revenue: {stats['percentile_95']:.2f}
    Mean Lead Time Risk: {stats['mean_lead_time_risk']:.2f}
    5th Percentile Lead Time Risk: {stats['percentile_5_lead_time']:.2f}
    95th Percentile Lead Time Risk: {stats['percentile_95_lead_time']:.2f}
    """

    return (simulation_summary, fig_simulation, fig_lead_time_risk, fig_defect_rate_risk, fig_shipping_cost_risk,
            fig_manufacturing_cost_risk, fig_box_plot, fig_violin_plot, fig_scatter_plot, fig_heatmap, fig_pair_plot,
            fig_cdf_plot, fig_bar_chart, fig_defect_rate_histogram, fig_pie_chart_risk_factors, fig_area_chart_revenue,
            fig_line_chart_lead_time_risk, fig_major_risk_factor)

if __name__ == '__main__':
    app.run_server(debug=True)
