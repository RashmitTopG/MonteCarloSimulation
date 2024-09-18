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
                        'fontSize': '16px',
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
                    marks={i: {'label': f'{i}', 'style': {'fontSize': '12px'}} for i in range(1000, 21000, 2000)},
                    tooltip={'placement': 'bottom', 'always_visible': True}
                )
            ],
            style={'textAlign': 'center', 'marginBottom': '30px'}
        ),
        html.Div(
            id="simulation-results",
            style={
                'fontSize': '14px',
                'textAlign': 'center',
                'marginBottom': '30px',
                'fontWeight': 'bold',
                'backgroundColor': '#e8f4f8',
                'borderRadius': '10px',
                'padding': '15px',
                'boxShadow': '0 4px 8px rgba(0,0,0,0.1)'
            }
        ),
        html.Div(
            [
                html.Div(dcc.Graph(id="simulation-graph"), style={'width': '80%', 'margin': '0 auto', 'marginBottom': '20px'}),
                html.Div(dcc.Graph(id="lead-time-risk-graph"), style={'width': '80%', 'margin': '0 auto', 'marginBottom': '20px'}),
                html.Div(dcc.Graph(id="defect-rate-risk-graph"), style={'width': '80%', 'margin': '0 auto', 'marginBottom': '20px'}),
                html.Div(dcc.Graph(id="shipping-cost-risk-graph"), style={'width': '80%', 'margin': '0 auto', 'marginBottom': '20px'}),
                html.Div(dcc.Graph(id="manufacturing-cost-risk-graph"), style={'width': '80%', 'margin': '0 auto', 'marginBottom': '20px'}),
                html.Div(dcc.Graph(id="box-plot-graph"), style={'width': '80%', 'margin': '0 auto', 'marginBottom': '20px'}),
                html.Div(dcc.Graph(id="violin-plot-graph"), style={'width': '80%', 'margin': '0 auto', 'marginBottom': '20px'}),
                html.Div(dcc.Graph(id="scatter-plot-graph"), style={'width': '80%', 'margin': '0 auto', 'marginBottom': '20px'}),
                html.Div(dcc.Graph(id="heatmap-graph"), style={'width': '80%', 'margin': '0 auto', 'marginBottom': '20px'}),
                html.Div(dcc.Graph(id="pair-plot-graph"), style={'width': '80%', 'margin': '0 auto', 'marginBottom': '20px'}),
                html.Div(dcc.Graph(id="cdf-plot-graph"), style={'width': '80%', 'margin': '0 auto', 'marginBottom': '20px'}),
                html.Div(dcc.Graph(id="bar-chart-graph"), style={'width': '80%', 'margin': '0 auto', 'marginBottom': '20px'}),
                html.Div(dcc.Graph(id="defect-rate-histogram"), style={'width': '80%', 'margin': '0 auto', 'marginBottom': '20px'}),
                html.Div(dcc.Graph(id="pie-chart-risk-factors"), style={'width': '80%', 'margin': '0 auto', 'marginBottom': '20px'}),
                html.Div(dcc.Graph(id="area-chart-revenue"), style={'width': '80%', 'margin': '0 auto', 'marginBottom': '20px'}),
                html.Div(dcc.Graph(id="line-chart-lead-time-risk"), style={'width': '80%', 'margin': '0 auto', 'marginBottom': '20px'}),
                html.Div(dcc.Graph(id="major-risk-factor-bar-chart"), style={'width': '80%', 'margin': '0 auto', 'marginBottom': '20px'})
            ]
        )
    ]
)

# Callback to update the simulation results and graphs dynamically
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

    # Calculate statistics
    stats = {
        'mean_revenue': np.mean(revenues),
        'std_revenue': np.std(revenues),
        'percentile_5': np.percentile(revenues, 5),
        'percentile_95': np.percentile(revenues, 95)
    }
    
    results_text = (f"Mean Revenue: ${stats['mean_revenue']:.2f}\n"
                    f"Standard Deviation: ${stats['std_revenue']:.2f}\n"
                    f"5th Percentile: ${stats['percentile_5']:.2f}\n"
                    f"95th Percentile: ${stats['percentile_95']:.2f}")
    
    # Create figures
    simulation_fig = go.Figure()
    simulation_fig.add_trace(go.Histogram(x=revenues, nbinsx=50, marker_color='#007bff'))
    simulation_fig.update_layout(title='Revenue Distribution', xaxis_title='Revenue', yaxis_title='Frequency')

    lead_time_risk_fig = px.histogram(lead_time_risks, nbins=50, title='Lead Time Risk Distribution')
    defect_rate_risk_fig = px.histogram(defect_rate_risks, nbins=50, title='Defect Rate Risk Distribution')
    shipping_cost_risk_fig = px.histogram(shipping_cost_risks, nbins=50, title='Shipping Cost Risk Distribution')
    manufacturing_cost_risk_fig = px.histogram(manufacturing_cost_risks, nbins=50, title='Manufacturing Cost Risk Distribution')

    # Placeholder plots for additional types
    box_plot_fig = go.Figure()
    box_plot_fig.add_trace(go.Box(y=revenues, marker_color='#007bff'))
    box_plot_fig.update_layout(title='Box Plot of Revenue')

    violin_plot_fig = go.Figure()
    violin_plot_fig.add_trace(go.Violin(y=revenues, marker_color='#007bff'))
    violin_plot_fig.update_layout(title='Violin Plot of Revenue')

    scatter_plot_fig = go.Figure()
    scatter_plot_fig.add_trace(go.Scatter(x=np.arange(len(revenues)), y=revenues, mode='markers', marker_color='#007bff'))
    scatter_plot_fig.update_layout(title='Scatter Plot of Revenue')

    heatmap_fig = go.Figure()
    heatmap_fig.add_trace(go.Heatmap(z=[revenues], colorscale='Viridis'))
    heatmap_fig.update_layout(title='Heatmap of Revenue')

    pair_plot_fig = go.Figure()
    pair_plot_fig.add_trace(go.Scatter(x=lead_time_risks, y=shipping_cost_risks, mode='markers', marker_color='#007bff'))
    pair_plot_fig.update_layout(title='Pair Plot of Lead Time Risk vs Shipping Cost Risk')

    cdf_plot_fig = go.Figure()
    cdf_plot_fig.add_trace(go.Histogram(x=revenues, cumulative_enabled=True, marker_color='#007bff'))
    cdf_plot_fig.update_layout(title='CDF Plot of Revenue')

    bar_chart_fig = go.Figure()
    bar_chart_fig.add_trace(go.Bar(x=['Lead Time Risk', 'Defect Rate Risk', 'Shipping Cost Risk', 'Manufacturing Cost Risk'],
                                   y=[np.mean(lead_time_risks), np.mean(defect_rate_risks),
                                      np.mean(shipping_cost_risks), np.mean(manufacturing_cost_risks)],
                                   marker_color='#007bff'))
    bar_chart_fig.update_layout(title='Bar Chart of Average Risk Factors')

    defect_rate_histogram_fig = px.histogram(defect_rate_risks, nbins=50, title='Defect Rate Histogram')

    pie_chart_fig = go.Figure()
    pie_chart_fig.add_trace(go.Pie(labels=['Lead Time Risk', 'Defect Rate Risk', 'Shipping Cost Risk', 'Manufacturing Cost Risk'],
                                  values=[np.mean(lead_time_risks), np.mean(defect_rate_risks),
                                          np.mean(shipping_cost_risks), np.mean(manufacturing_cost_risks)],
                                  marker_colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']))
    pie_chart_fig.update_layout(title='Pie Chart of Risk Factors')

    area_chart_fig = go.Figure()
    area_chart_fig.add_trace(go.Scatter(x=np.arange(len(revenues)), y=revenues, fill='tozeroy', marker_color='#007bff'))
    area_chart_fig.update_layout(title='Area Chart of Revenue')

    line_chart_fig = go.Figure()
    line_chart_fig.add_trace(go.Scatter(x=np.arange(len(lead_time_risks)), y=lead_time_risks, mode='lines', marker_color='#007bff'))
    line_chart_fig.update_layout(title='Line Chart of Lead Time Risk')

    major_risk_factor_bar_chart_fig = go.Figure()
    major_risk_factor_bar_chart_fig.add_trace(go.Bar(x=['Lead Time Risk', 'Defect Rate Risk', 'Shipping Cost Risk', 'Manufacturing Cost Risk'],
                                                    y=[np.max(lead_time_risks), np.max(defect_rate_risks),
                                                       np.max(shipping_cost_risks), np.max(manufacturing_cost_risks)],
                                                    marker_color='#007bff'))
    major_risk_factor_bar_chart_fig.update_layout(title='Bar Chart of Major Risk Factors')

    # Return all figures and text
    return (
        results_text, simulation_fig, lead_time_risk_fig, defect_rate_risk_fig,
        shipping_cost_risk_fig, manufacturing_cost_risk_fig, box_plot_fig,
        violin_plot_fig, scatter_plot_fig, heatmap_fig, pair_plot_fig,
        cdf_plot_fig, bar_chart_fig, defect_rate_histogram_fig, pie_chart_fig,
        area_chart_fig, line_chart_fig, major_risk_factor_bar_chart_fig
    )

if __name__ == '__main__':
    app.run_server(debug=True)
