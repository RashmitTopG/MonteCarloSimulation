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

    return (revenues, simulated_lead_times * 0.05, simulated_defect_rates,
            simulated_shipping_costs, simulated_manufacturing_costs)

# Create the Dash app
app = Dash(__name__)

# Layout of the dashboard with embedded CSS
app.layout = html.Div(
    style={
        'maxWidth': '1200px',
        'margin': 'auto',
        'padding': '20px',
        'backgroundColor': '#f8f9fa',
        'fontFamily': 'Arial, sans-serif'
    },
    children=[
        html.H1(
            "Monte Carlo Simulation Dashboard for Supply Chain Risks",
            style={
                'textAlign': 'center',
                'fontSize': '2.5em',
                'fontWeight': 'bold',
                'color': '#343a40',
                'marginBottom': '20px'
            }
        ),
        html.Div(
            [
                html.Label(
                    "Number of Simulations:",
                    style={
                        'fontSize': '1.2em',
                        'fontWeight': 'bold',
                        'color': '#495057',
                        'marginBottom': '10px'
                    }
                ),
                dcc.Slider(
                    id="sim-slider",
                    min=1000,
                    max=20000,
                    step=1000,
                    value=5000,
                    marks={i: {'label': str(i), 'style': {'fontSize': '14px'}} for i in range(1000, 21001, 2000)},
                    tooltip={'placement': 'bottom', 'always_visible': True},
                    updatemode='drag'
                )
            ],
            style={'textAlign': 'center', 'marginBottom': '20px'}
        ),
        html.Div(
            id="simulation-results",
            style={
                'fontSize': '1.2em',
                'textAlign': 'center',
                'marginBottom': '20px',
                'fontWeight': 'bold',
                'backgroundColor': '#d1ecf1',
                'borderLeft': '4px solid #17a2b8',
                'padding': '10px',
                'borderRadius': '5px',
                'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)'
            }
        ),
        html.Div(
            [
                html.Div(dcc.Graph(id="simulation-graph"), style={'flex': '1', 'margin': '10px'}),
                html.Div(dcc.Graph(id="lead-time-risk-graph"), style={'flex': '1', 'margin': '10px'})
            ],
            style={'display': 'flex', 'marginBottom': '20px'}
        ),
        html.Div(
            [
                html.Div(dcc.Graph(id="defect-rate-risk-graph"), style={'flex': '1', 'margin': '10px'}),
                html.Div(dcc.Graph(id="shipping-cost-risk-graph"), style={'flex': '1', 'margin': '10px'})
            ],
            style={'display': 'flex', 'marginBottom': '20px'}
        ),
        html.Div(
            [
                html.Div(dcc.Graph(id="manufacturing-cost-risk-graph"), style={'flex': '1', 'margin': '10px'}),
                html.Div(dcc.Graph(id="box-plot-graph"), style={'flex': '1', 'margin': '10px'})
            ],
            style={'display': 'flex', 'marginBottom': '20px'}
        ),
        html.Div(
            [
                html.Div(dcc.Graph(id="cdf-plot-graph"), style={'flex': '1', 'margin': '10px'}),
                html.Div(dcc.Graph(id="bar-chart-graph"), style={'flex': '1', 'margin': '10px'})
            ],
            style={'display': 'flex', 'marginBottom': '20px'}
        ),
        html.Div(
            [
                html.Div(dcc.Graph(id="defect-rate-histogram"), style={'flex': '1', 'margin': '10px'}),
                html.Div(dcc.Graph(id="pie-chart-risk-factors"), style={'flex': '1', 'margin': '10px'})
            ],
            style={'display': 'flex', 'marginBottom': '20px'}
        ),
        html.Div(
            [
                html.Div(dcc.Graph(id="major-risk-factor-bar-chart"), style={'flex': '1', 'margin': '10px'})
            ],
            style={'textAlign': 'center', 'marginBottom': '20px'}
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
     Output('cdf-plot-graph', 'figure'),
     Output('bar-chart-graph', 'figure'),
     Output('defect-rate-histogram', 'figure'),
     Output('pie-chart-risk-factors', 'figure'),
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

    # Create graphs with updated styling
    template = 'plotly_dark'

    fig_simulation = px.histogram(revenues, nbins=100, title="Revenue Distribution").update_layout(
        xaxis_title='Revenue', yaxis_title='Frequency', template=template,
        title_x=0.5, font=dict(family="Arial", size=14, color="white"), paper_bgcolor='#2c3e50',
        hoverlabel=dict(font_color="white", font_size=12)
    )

    fig_lead_time_risk = px.histogram(lead_time_risks, nbins=100, title="Lead Time Risk Distribution").update_layout(
        xaxis_title='Lead Time Risk', yaxis_title='Frequency', template=template,
        title_x=0.5, font=dict(family="Arial", size=14, color="white"), paper_bgcolor='#2c3e50',
        hoverlabel=dict(font_color="white", font_size=12)
    )

    fig_defect_rate_risk = px.histogram(defect_rate_risks, nbins=100, title="Defect Rate Risk Distribution").update_layout(
        xaxis_title='Defect Rate Risk', yaxis_title='Frequency', template=template,
        title_x=0.5, font=dict(family="Arial", size=14, color="white"), paper_bgcolor='#2c3e50',
        hoverlabel=dict(font_color="white", font_size=12)
    )

    fig_shipping_cost_risk = px.histogram(shipping_cost_risks, nbins=100, title="Shipping Cost Risk Distribution").update_layout(
        xaxis_title='Shipping Cost Risk', yaxis_title='Frequency', template=template,
        title_x=0.5, font=dict(family="Arial", size=14, color="white"), paper_bgcolor='#2c3e50',
        hoverlabel=dict(font_color="white", font_size=12)
    )

    fig_manufacturing_cost_risk = px.histogram(manufacturing_cost_risks, nbins=100, title="Manufacturing Cost Risk Distribution").update_layout(
        xaxis_title='Manufacturing Cost Risk', yaxis_title='Frequency', template=template,
        title_x=0.5, font=dict(family="Arial", size=14, color="white"), paper_bgcolor='#2c3e50',
        hoverlabel=dict(font_color="white", font_size=12)
    )

    fig_box_plot = px.box(pd.DataFrame({'Revenues': revenues}), title="Revenue Box Plot").update_layout(
        yaxis_title='Revenue', template=template,
        title_x=0.5, font=dict(family="Arial", size=14, color="white"), paper_bgcolor='#2c3e50'
    )

    fig_cdf_plot = px.ecdf(pd.DataFrame({'Revenues': revenues}), x='Revenues', title="Cumulative Distribution Function (CDF)").update_layout(
        xaxis_title='Revenue', yaxis_title='Cumulative Probability', template=template,
        title_x=0.5, font=dict(family="Arial", size=14, color="white"), paper_bgcolor='#2c3e50'
    )

    fig_bar_chart = px.bar(pd.DataFrame({'Revenues': revenues}), title="Revenue Bar Chart").update_layout(
        xaxis_title='Simulation Number', yaxis_title='Revenue', template=template,
        title_x=0.5, font=dict(family="Arial", size=14, color="white"), paper_bgcolor='#2c3e50'
    )

    fig_defect_rate_histogram = px.histogram(pd.DataFrame({'Defect Rates': defect_rate_risks}), nbins=50, title="Defect Rate Histogram").update_layout(
        xaxis_title='Defect Rate', yaxis_title='Frequency', template=template,
        title_x=0.5, font=dict(family="Arial", size=14, color="white"), paper_bgcolor='#2c3e50'
    )

    fig_pie_chart_risk_factors = px.pie(names=["Revenue", "Shipping Costs", "Manufacturing Costs"], values=[np.mean(revenues), np.mean(shipping_cost_risks), np.mean(manufacturing_cost_risks)], title="Risk Factors Distribution").update_traces(textinfo='percent+label')

    fig_major_risk_factor_bar_chart = px.bar(
        x=["Defect Rate", "Shipping Costs", "Manufacturing Costs", "Lead Time"],
        y=[np.mean(defect_rate_risks), np.mean(shipping_cost_risks), np.mean(manufacturing_cost_risks), np.mean(lead_time_risks)],
        title="Major Risk Factors Bar Chart"
    ).update_layout(
        xaxis_title='Risk Factor', yaxis_title='Average Impact', template=template,
        title_x=0.5, font=dict(family="Arial", size=14, color="white"), paper_bgcolor='#2c3e50'
    )

    results_text = f"Mean Revenue: {stats['mean_revenue']:.2f} | Std Dev Revenue: {stats['std_revenue']:.2f} | 5th Percentile: {stats['percentile_5']:.2f} | 95th Percentile: {stats['percentile_95']:.2f} | Mean Lead Time Risk: {stats['mean_lead_time_risk']:.2f}"

    return (results_text, fig_simulation, fig_lead_time_risk, fig_defect_rate_risk,
            fig_shipping_cost_risk, fig_manufacturing_cost_risk, fig_box_plot,
            fig_cdf_plot, fig_bar_chart, fig_defect_rate_histogram,
            fig_pie_chart_risk_factors, fig_major_risk_factor_bar_chart)

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
