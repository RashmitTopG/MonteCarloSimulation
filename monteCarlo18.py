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
                html.Div(dcc.Graph(id="simulation-graph", style={'height': '450px'}), style={'flex': '1', 'margin': '10px'}),
                html.Div(dcc.Graph(id="lead-time-risk-graph", style={'height': '450px'}), style={'flex': '1', 'margin': '10px'})
            ],
            style={'display': 'flex', 'marginBottom': '20px'}
        ),
        html.Div(
            [
                html.Div(dcc.Graph(id="defect-rate-risk-graph", style={'height': '450px'}), style={'flex': '1', 'margin': '10px'}),
                html.Div(dcc.Graph(id="shipping-cost-risk-graph", style={'height': '450px'}), style={'flex': '1', 'margin': '10px'})
            ],
            style={'display': 'flex', 'marginBottom': '20px'}
        ),
        html.Div(
            [
                html.Div(dcc.Graph(id="manufacturing-cost-risk-graph", style={'height': '450px'}), style={'flex': '1', 'margin': '10px'}),
                html.Div(dcc.Graph(id="box-plot-graph", style={'height': '450px'}), style={'flex': '1', 'margin': '10px'})
            ],
            style={'display': 'flex', 'marginBottom': '20px'}
        ),
        html.Div(
            [
                html.Div(dcc.Graph(id="cdf-plot-graph", style={'height': '450px'}), style={'flex': '1', 'margin': '10px'}),
                html.Div(dcc.Graph(id="bar-chart-graph", style={'height': '450px'}), style={'flex': '1', 'margin': '10px'})
            ],
            style={'display': 'flex', 'marginBottom': '20px'}
        ),
        html.Div(
            [
                html.Div(dcc.Graph(id="defect-rate-histogram", style={'height': '450px'}), style={'flex': '1', 'margin': '10px'}),
                html.Div(dcc.Graph(id="pie-chart-risk-factors", style={'height': '450px'}), style={'flex': '1', 'margin': '10px'})
            ],
            style={'display': 'flex', 'marginBottom': '20px'}
        ),
        html.Div(
            [
                html.Div(dcc.Graph(id="major-risk-factor-bar-chart", style={'height': '450px'}), style={'flex': '1', 'margin': '10px'})
            ],
            style={'textAlign': 'center', 'marginBottom': '20px'}
        ),
        # Add the conclusion section here
        html.Div(
            style={
                'backgroundColor': '#fff3cd',
                'borderLeft': '4px solid #ffc107',
                'padding': '20px',
                'borderRadius': '5px',
                'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)',
                'marginBottom': '20px'
            },
            children=[
                html.H2("Conclusion", style={'color': '#856404', 'textAlign': 'center', 'marginBottom': '10px'}),
                html.P(
                    """
                    The analysis highlights that manufacturing cost is the primary driver of risk in the business, accounting 
                    for the vast majority of total risk (85%). On the other hand, while revenues are skewed, a small number 
                    of outliers account for high revenues, with most revenue values falling below 100k.
                    """,
                    style={'color': '#856404', 'fontSize': '1.1em', 'textAlign': 'justify'}
                ),
                html.P(
                    """
                    To mitigate risk and enhance overall performance, the company should focus on controlling manufacturing 
                    costs since they have the largest impact on risk. Additionally, efforts to increase revenue generation 
                    beyond the lower range should be prioritized, particularly by addressing the extreme cases that generate 
                    higher revenue outliers.
                    """,
                    style={'color': '#856404', 'fontSize': '1.1em', 'textAlign': 'justify'}
                )
            ]
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
def update_simulation(num_simulations):
    revenues, lead_times, defect_rates, shipping_costs, manufacturing_costs = monte_carlo_simulation(num_simulations)

    # Calculate statistics
    mean_revenue = np.mean(revenues)
    std_revenue = np.std(revenues)

    mean_lead_time_cost = np.mean(lead_time)
    std_lead_time_cost = np.std(lead_time)

    mean_defect_rate = np.mean(defect_rates)
    std_defect_rate = np.std(defect_rates)

    mean_shipping_cost = np.mean(shipping_costs)
    std_shipping_cost = np.std(shipping_costs)

    mean_manufacturing_cost = np.mean(manufacturing_costs)
    std_manufacturing_cost = np.std(manufacturing_costs)

    # Update the output with calculated statistics
    simulation_results_text = (
        f"Simulations run: {num_simulations}\n"
        f"Mean Revenue: ${mean_revenue:,.2f} | Std Dev: ${std_revenue:,.2f}\n"
        f"Mean Lead Time Cost: ${mean_lead_time_cost:,.2f} | Std Dev: ${std_lead_time_cost:,.2f}\n"
        f"Mean Defect Rate: {mean_defect_rate:.2%} | Std Dev: {std_defect_rate:.2%}\n"
        f"Mean Shipping Cost: ${mean_shipping_cost:,.2f} | Std Dev: ${std_shipping_cost:,.2f}\n"
        f"Mean Manufacturing Cost: ${mean_manufacturing_cost:,.2f} | Std Dev: ${std_manufacturing_cost:,.2f}\n"
    )


    fig_revenue = px.histogram(revenues, nbins=50, title='Revenue Distribution',
                               labels={'x': 'Revenue', 'y': 'Frequency'},
                               template='plotly_dark')
    fig_lead_time = px.histogram(lead_times, nbins=50, title='Lead Time Distribution',
                                 labels={'x': 'Lead Time', 'y': 'Frequency'},
                                 template='plotly_dark')
    fig_defect_rate = px.histogram(defect_rates, nbins=50, title='Defect Rate Distribution',
                                   labels={'x': 'Defect Rate', 'y': 'Frequency'},
                                   template='plotly_dark')
    fig_shipping_cost = px.histogram(shipping_costs, nbins=50, title='Shipping Cost Distribution',
                                     labels={'x': 'Shipping Costs', 'y': 'Frequency'},
                                     template='plotly_dark')
    fig_manufacturing_cost = px.histogram(manufacturing_costs, nbins=50, title='Manufacturing Cost Distribution',
                                          labels={'x': 'Manufacturing Costs', 'y': 'Frequency'},
                                          template='plotly_dark')

    df_box = pd.DataFrame({
        'Revenues': revenues,
        'Lead Times': lead_times,
        'Defect Rates': defect_rates,
        'Shipping Costs': shipping_costs,
        'Manufacturing Costs': manufacturing_costs
    })

    fig_box_plot = px.box(df_box, title="Box Plot of Supply Chain Variables",
                          template='plotly_dark')
    fig_cdf = px.ecdf(revenues, title='CDF of Revenue Distribution',
                      labels={'x': 'Revenue', 'y': 'Cumulative Probability'},
                      template='plotly_dark')

    risk_contributions = {
        'Lead Times': np.mean(lead_times),
        'Defect Rates': np.mean(defect_rates),
        'Shipping Costs': np.mean(shipping_costs),
        'Manufacturing Costs': np.mean(manufacturing_costs)
    }

    fig_bar_chart = px.bar(x=list(risk_contributions.keys()), y=list(risk_contributions.values()),
                           title="Average Contribution of Each Risk Factor",
                           labels={'x': 'Risk Factor', 'y': 'Average Value'},
                           template='plotly_dark')

    fig_defect_rate_hist = px.histogram(defect_rates, nbins=50, title='Histogram of Defect Rates',
                                        labels={'x': 'Defect Rate', 'y': 'Frequency'},
                                        template='plotly_dark')

    fig_pie_risk_factors = px.pie(names=list(risk_contributions.keys()), values=list(risk_contributions.values()),
                                  title='Proportional Contribution of Risk Factors',
                                  template='plotly_dark')

    fig_major_risk = px.bar(x=list(risk_contributions.keys()), y=list(risk_contributions.values()),
                            title="Major Risk Factor Analysis",
                            labels={'x': 'Risk Factor', 'y': 'Impact'},
                            template='plotly_dark')

    return [
        simulation_results_text,
        fig_revenue,
        fig_lead_time,
        fig_defect_rate,
        fig_shipping_cost,
        fig_manufacturing_cost,
        fig_box_plot,
        fig_cdf,
        fig_bar_chart,
        fig_defect_rate_hist,
        fig_pie_risk_factors,
        fig_major_risk
    ]

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
