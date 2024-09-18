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

print(price)

# Function to perform Monte Carlo simulation
def monte_carlo_simulation(num_simulations=5000):  
    revenues = []
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
        lead_time_risks.append(lead_time_risk)
        defect_rate_risks.append(simulated_defect_rate)
        shipping_cost_risks.append(simulated_shipping_costs)
        manufacturing_cost_risks.append(simulated_manufacturing_costs)

    return (np.array(revenues), np.array(lead_time_risks), np.array(defect_rate_risks),
            np.array(shipping_cost_risks), np.array(manufacturing_cost_risks))

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
     Output('bar-chart-graph', 'figure')],
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

    # Create histogram for revenue simulation results
    fig_revenue = px.histogram(revenues, nbins=50, title='Distribution of Simulated Revenues')
    fig_revenue.update_layout(xaxis_title='Revenue', yaxis_title='Frequency', template='plotly_dark')

    # Create histogram for lead time risk results
    fig_lead_time_risk = px.histogram(lead_time_risks, nbins=50, title='Distribution of Lead Time Risks')
    fig_lead_time_risk.update_layout(xaxis_title='Lead Time Risk', yaxis_title='Frequency', template='plotly_dark')

    # Create histogram for defect rate risk results
    fig_defect_rate_risk = px.histogram(defect_rate_risks, nbins=50, title='Distribution of Defect Rate Risks')
    fig_defect_rate_risk.update_layout(xaxis_title='Defect Rate Risk', yaxis_title='Frequency', template='plotly_dark')

    # Create histogram for shipping cost risk results
    fig_shipping_cost_risk = px.histogram(shipping_cost_risks, nbins=50, title='Distribution of Shipping Cost Risks')
    fig_shipping_cost_risk.update_layout(xaxis_title='Shipping Cost Risk', yaxis_title='Frequency', template='plotly_dark')

    # Create histogram for manufacturing cost risk results
    fig_manufacturing_cost_risk = px.histogram(manufacturing_cost_risks, nbins=50, title='Distribution of Manufacturing Cost Risks')
    fig_manufacturing_cost_risk.update_layout(xaxis_title='Manufacturing Cost Risk', yaxis_title='Frequency', template='plotly_dark')

    # Create box plot for revenue
    fig_box_plot = px.box(revenues, title='Box Plot of Simulated Revenues')
    fig_box_plot.update_layout(template='plotly_dark')

    # Create violin plot for revenue
    fig_violin_plot = px.violin(revenues, title='Violin Plot of Simulated Revenues')
    fig_violin_plot.update_layout(template='plotly_dark')

    # Create scatter plot for shipping cost vs. revenue
    fig_scatter_plot = px.scatter(x=shipping_cost_risks, y=revenues, title='Scatter Plot of Shipping Cost vs. Revenue')
    fig_scatter_plot.update_layout(xaxis_title='Shipping Cost Risk', yaxis_title='Revenue', template='plotly_dark')

    # Create heatmap of correlations between risks and revenue
    risk_data = pd.DataFrame({
        'Revenue': revenues,
        'Lead Time Risk': lead_time_risks,
        'Defect Rate Risk': defect_rate_risks,
        'Shipping Cost Risk': shipping_cost_risks,
        'Manufacturing Cost Risk': manufacturing_cost_risks
    })
    correlation_matrix = risk_data.corr()
    fig_heatmap = px.imshow(correlation_matrix, title='Heatmap of Correlations')
    fig_heatmap.update_layout(template='plotly_dark')

    # Create pair plot (scatter matrix) for risk data
    fig_pair_plot = px.scatter_matrix(risk_data, title='Pair Plot of Risks and Revenue')
    fig_pair_plot.update_layout(template='plotly_dark')

    # Create CDF plot for revenue
    fig_cdf_plot = px.ecdf(revenues, title='CDF Plot of Simulated Revenues')
    fig_cdf_plot.update_layout(xaxis_title='Revenue', yaxis_title='CDF', template='plotly_dark')

    # Create bar chart for top risks
    top_risks = risk_data.mean().sort_values(ascending=False)
    fig_bar_chart = px.bar(top_risks, title='Bar Chart of Top Risks')
    fig_bar_chart.update_layout(xaxis_title='Risk', yaxis_title='Mean Value', template='plotly_dark')

    # Simulation result summary
    result_summary = [
        html.Div(f"Mean Simulated Revenue: ${mean_revenue:,.2f}"),
        html.Div(f"Standard Deviation of Revenue: ${std_revenue:,.2f}"),
        html.Div(f"5th Percentile Revenue: ${percentile_5:,.2f}"),
        html.Div(f"95th Percentile Revenue: ${percentile_95:,.2f}"),
        html.Div(f"Mean Lead Time Risk: {mean_lead_time_risk:.2f}"),
        html.Div(f"5th Percentile Lead Time Risk: {percentile_5_lead_time:.2f}"),
        html.Div(f"95th Percentile Lead Time Risk: {percentile_95_lead_time:.2f}")
    ]

    return (result_summary, fig_revenue, fig_lead_time_risk, fig_defect_rate_risk,
            fig_shipping_cost_risk, fig_manufacturing_cost_risk, fig_box_plot,
            fig_violin_plot, fig_scatter_plot, fig_heatmap, fig_pair_plot, fig_cdf_plot, fig_bar_chart)

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
