import numpy as np
import pandas as pd
from dash import Dash, html, dcc, Input, Output
import plotly.express as px
import plotly.graph_objs as go

# Load dataset
data = pd.read_csv('supply_chain_data.csv')

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
    revenues = np.maximum(0, (simulated_prices * adjusted_sales) - 
                          (simulated_shipping_costs + simulated_manufacturing_costs + simulated_lead_times * 0.05))

    return (revenues, simulated_lead_times * 0.05, simulated_defect_rates,
            simulated_shipping_costs, simulated_manufacturing_costs)

# Create the Dash app
app = Dash(__name__)

# Define styles
graph_container_style = {'flex': '1', 'margin': '10px'}
graph_style = {'height': '450px', 'width': '100%'}
description_style = {
    'marginTop': '10px',
    'padding': '10px',
    'backgroundColor': '#e9ecef',
    'borderRadius': '5px',
    'fontSize': '0.9em'
}

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
        # Graph Sections
        html.Div(
            [
                html.Div([
                    dcc.Graph(id="simulation-graph", style=graph_style),
                    html.Div(id="revenue-description", style=description_style)
                ], style=graph_container_style),
                html.Div([
                    dcc.Graph(id="lead-time-risk-graph", style=graph_style),
                    html.Div(id="lead-time-description", style=description_style)
                ], style=graph_container_style)
            ],
            style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'space-between'}
        ),
        html.Div(
            [
                html.Div([
                    dcc.Graph(id="defect-rate-risk-graph", style=graph_style),
                    html.Div(id="defect-rate-description", style=description_style)
                ], style=graph_container_style),
                html.Div([
                    dcc.Graph(id="shipping-cost-risk-graph", style=graph_style),
                    html.Div(id="shipping-cost-description", style=description_style)
                ], style=graph_container_style)
            ],
            style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'space-between'}
        ),
        html.Div(
            [
                html.Div([
                    dcc.Graph(id="manufacturing-cost-risk-graph", style=graph_style),
                    html.Div(id="manufacturing-cost-description", style=description_style)
                ], style=graph_container_style),
                html.Div([
                    dcc.Graph(id="box-plot-graph", style=graph_style),
                    html.Div(id="box-plot-description", style=description_style)
                ], style=graph_container_style)
            ],
            style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'space-between'}
        ),
        html.Div(
            [
                html.Div([
                    dcc.Graph(id="cdf-plot-graph", style=graph_style),
                    html.Div(id="cdf-description", style=description_style)
                ], style=graph_container_style),
                html.Div([
                    dcc.Graph(id="bar-chart-graph", style=graph_style),
                    html.Div(id="bar-chart-description", style=description_style)
                ], style=graph_container_style)
            ],
            style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'space-between'}
        ),
        html.Div(
            [
                html.Div([
                    dcc.Graph(id="defect-rate-histogram", style=graph_style),
                    html.Div(id="defect-rate-histogram-description", style=description_style)
                ], style=graph_container_style),
                html.Div([
                    dcc.Graph(id="pie-chart-risk-factors", style=graph_style),
                    html.Div(id="pie-chart-description", style=description_style)
                ], style=graph_container_style)
            ],
            style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'space-between'}
        ),
        html.Div(
            [
                html.Div([
                    dcc.Graph(id="major-risk-factor-bar-chart", style=graph_style),
                    html.Div(id="major-risk-factor-description", style=description_style)
                ], style={'flex': '1', 'margin': '10px', 'minWidth': '300px'})
            ],
            style={'display': 'flex', 'justifyContent': 'center', 'marginBottom': '20px'}
        ),
        html.Div(
            [
                html.Div([
                    dcc.Graph(id="risk-heatmap", style=graph_style),
                    html.Div(id="risk-heatmap-description", style=description_style)
                ], style=graph_container_style),
                html.Div([
                    dcc.Graph(id="tornado-diagram", style=graph_style),
                    html.Div(id="tornado-diagram-description", style=description_style)
                ], style=graph_container_style)
            ],
            style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'space-between'}
        ),html.Div(
            [
                html.Div([
                    dcc.Graph(id="waterfall-chart", style=graph_style),
                    html.Div(id="waterfall-chart-description", style=description_style)
                ], style=graph_container_style),
                html.Div([
                    dcc.Graph(id="bubble-chart", style=graph_style),
                    html.Div(id="bubble-chart-description", style=description_style)
                ], style=graph_container_style)
            ],
            style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'space-between'}
        ),
        html.Div(
            [
                html.Div([
                    dcc.Graph(id="scenario-analysis", style=graph_style),
                    html.Div(id="scenario-analysis-description", style=description_style)
                ], style=graph_container_style),
                html.Div([
                    dcc.Graph(id="risk-radar-chart", style=graph_style),
                    html.Div(id="risk-radar-chart-description", style=description_style)
                ], style=graph_container_style)
            ],
            style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'space-between'}
        ),
         html.Div(
            style={
                'backgroundColor': '#fff3cd',
                'borderLeft': '4px solid #ffc107',
                'padding': '20px',
                'borderRadius': '5px',
                'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)',
                'marginTop': '20px'
            },
            children=[
                html.H2("Key Insights", style={'color': '#856404', 'textAlign': 'center', 'marginBottom': '10px'}),
                html.Div(id="key-insights", style={'color': '#856404', 'fontSize': '1.1em', 'textAlign': 'justify'})
            ]
        ),
        
        # Conclusion Section
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
                    [
                        "This analysis is performed on the dataset named ", 
                        html.A("supply_chain_data", 
                                href="https://www.kaggle.com/datasets/harshsingh2209/supply-chain-analysis?resource=download", 
                                target="_blank", 
                                style={'fontWeight': 'bold' ,}),
                        ", which is present on Kaggle. The analysis highlights that manufacturing cost is the primary driver of risk in the business, accounting for the vast majority of total risk (85%). On the other hand, while revenues are skewed, a small number of outliers account for high revenues, with most revenue values falling below ₹100k."
                    ],
                    style={'color': '#856404', 'fontSize': '1.1em', 'textAlign': 'justify'}
                ),
                html.P(
                    [
                        "To mitigate risk and enhance overall performance, the company should focus on controlling manufacturing costs since they have the largest impact on risk. Additionally, efforts to increase revenue generation beyond the lower range should be prioritized, particularly by addressing the extreme cases that generate higher revenue outliers."
                    ],
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
     Output('revenue-description', 'children'),
     Output('lead-time-risk-graph', 'figure'),
     Output('lead-time-description', 'children'),
     Output('defect-rate-risk-graph', 'figure'),
     Output('defect-rate-description', 'children'),
     Output('shipping-cost-risk-graph', 'figure'),
     Output('shipping-cost-description', 'children'),
     Output('manufacturing-cost-risk-graph', 'figure'),
     Output('manufacturing-cost-description', 'children'),
     Output('box-plot-graph', 'figure'),
     Output('box-plot-description', 'children'),
     Output('cdf-plot-graph', 'figure'),
     Output('cdf-description', 'children'),
     Output('bar-chart-graph', 'figure'),
     Output('bar-chart-description', 'children'),
     Output('defect-rate-histogram', 'figure'),
     Output('defect-rate-histogram-description', 'children'),
     Output('pie-chart-risk-factors', 'figure'),
     Output('pie-chart-description', 'children'),
     Output('major-risk-factor-bar-chart', 'figure'),
     Output('major-risk-factor-description', 'children'),
     Output('risk-heatmap', 'figure'),
     Output('risk-heatmap-description', 'children'),Output('tornado-diagram', 'figure'),
     Output('tornado-diagram-description', 'children'),
     Output('waterfall-chart', 'figure'),
     Output('waterfall-chart-description', 'children'),
     Output('bubble-chart', 'figure'),
     Output('bubble-chart-description', 'children'),
     Output('scenario-analysis', 'figure'),
     Output('scenario-analysis-description', 'children'),
     Output('risk-radar-chart', 'figure'),
     Output('risk-radar-chart-description', 'children'),
     Output('key-insights', 'children')],
    [Input('sim-slider', 'value')]
)
def update_simulation(num_simulations):
    revenues, lead_times, defect_rates, shipping_costs_sim, manufacturing_costs_sim = monte_carlo_simulation(num_simulations)

    # Calculate statistics
    mean_revenue = np.mean(revenues)
    std_revenue = np.std(revenues)
    mean_lead_time_cost = np.mean(lead_times)
    std_lead_time_cost = np.std(lead_times)
    mean_defect_rate = np.mean(defect_rates)
    std_defect_rate = np.std(defect_rates)
    mean_shipping_cost = np.mean(shipping_costs_sim)
    std_shipping_cost = np.std(shipping_costs_sim)
    mean_manufacturing_cost = np.mean(manufacturing_costs_sim)
    std_manufacturing_cost = np.std(manufacturing_costs_sim)

    # Update the output with calculated statistics
    simulation_results_text = html.Div([
        html.P(f"Simulations run: {num_simulations}"),
        html.P(f"Mean Revenue: ₹{mean_revenue:,.2f} | Std Dev: ₹{std_revenue:,.2f}"),
        html.P(f"Mean Lead Time Cost: ₹{mean_lead_time_cost:,.2f} | Std Dev: ₹{std_lead_time_cost:,.2f}"),
        html.P(f"Mean Defect Rate: {mean_defect_rate:.2%} | Std Dev: {std_defect_rate:.2%}"),
        html.P(f"Mean Shipping Cost: ₹{mean_shipping_cost:,.2f} | Std Dev: ₹{std_shipping_cost:,.2f}"),
        html.P(f"Mean Manufacturing Cost: ₹{mean_manufacturing_cost:,.2f} | Std Dev: ₹{std_manufacturing_cost:,.2f}")
    ])

    # Define axis ranges based on percentiles to exclude extreme outliers
    revenue_min, revenue_max = np.percentile(revenues, [1, 99])
    lead_time_min, lead_time_max = np.percentile(lead_times, [1, 99])
    defect_rate_min, defect_rate_max = np.percentile(defect_rates, [1, 99])
    shipping_cost_min, shipping_cost_max = np.percentile(shipping_costs_sim, [1, 99])
    manufacturing_cost_min, manufacturing_cost_max = np.percentile(manufacturing_costs_sim, [1, 99])

    # Revenue Distribution Graph
    fig_revenue = px.histogram(
        revenues, 
        nbins=50, 
        title='Revenue Distribution',
        labels={'x': 'Revenue (₹)', 'y': 'Frequency'},
        template='plotly_dark'
    )
    # ... (continuing from the previous code)

    fig_revenue.update_layout(
        xaxis=dict(range=[revenue_min, revenue_max], title='Revenue (₹)'),
        yaxis=dict(title='Frequency'),
        bargap=0.05,
        showlegend=False
    )
    revenue_description = """
    This histogram shows the distribution of simulated revenues. The shape of the distribution indicates the range and likelihood of different revenue outcomes. A wide spread suggests high revenue volatility, while a narrow distribution indicates more predictable revenues. Skewness to the right may indicate potential for high profits, but also higher risk.
    """

    # Lead Time Risk Graph
    fig_lead_time = px.histogram(
        lead_times, 
        nbins=50, 
        title='Lead Time Distribution',
        labels={'x': 'Lead Time (days)', 'y': 'Frequency'},
        template='plotly_dark'
    )
    fig_lead_time.update_layout(
        xaxis=dict(range=[lead_time_min, lead_time_max], title='Lead Time (days)'),
        yaxis=dict(title='Frequency'),
        bargap=0.05,
        showlegend=False
    )
    lead_time_description = """
    The lead time distribution reveals the variability in delivery times. Longer lead times can result in increased inventory costs and reduced customer satisfaction. A wide distribution suggests unreliable supply chain performance, while a narrow distribution indicates more consistent and predictable lead times.
    """

    # Defect Rate Risk Graph
    fig_defect_rate = px.histogram(
        defect_rates, 
        nbins=50, 
        title='Defect Rate Distribution',
        labels={'x': 'Defect Rate (%)', 'y': 'Frequency'},
        template='plotly_dark'
    )
    fig_defect_rate.update_layout(
        xaxis=dict(range=[defect_rate_min, defect_rate_max], title='Defect Rate (%)'),
        yaxis=dict(title='Frequency'),
        bargap=0.05,
        showlegend=False
    )
    defect_rate_description = """
    This histogram illustrates the distribution of defect rates. Higher defect rates can lead to increased costs, customer dissatisfaction, and potential loss of business. A distribution skewed towards lower defect rates is ideal, while a wide spread or high average defect rate indicates a significant quality control risk.
    """

    # Shipping Cost Risk Graph
    fig_shipping_cost = px.histogram(
        shipping_costs_sim, 
        nbins=50, 
        title='Shipping Cost Distribution',
        labels={'x': 'Shipping Costs (₹)', 'y': 'Frequency'},
        template='plotly_dark'
    )
    fig_shipping_cost.update_layout(
        xaxis=dict(range=[shipping_cost_min, shipping_cost_max], title='Shipping Costs (₹)'),
        yaxis=dict(title='Frequency'),
        bargap=0.05,
        showlegend=False
    )
    shipping_cost_description = """
    The shipping cost distribution shows the variability in transportation expenses. High or unpredictable shipping costs can significantly impact profit margins. A wide distribution indicates potential for cost overruns, while a narrow distribution suggests more stable and predictable shipping expenses.
    """

    # Manufacturing Cost Risk Graph
    fig_manufacturing_cost = px.histogram(
        manufacturing_costs_sim, 
        nbins=50, 
        title='Manufacturing Cost Distribution',
        labels={'x': 'Manufacturing Costs (₹)', 'y': 'Frequency'},
        template='plotly_dark'
    )
    fig_manufacturing_cost.update_layout(
        xaxis=dict(range=[manufacturing_cost_min, manufacturing_cost_max], title='Manufacturing Costs (₹)'),
        yaxis=dict(title='Frequency'),
        bargap=0.05,
        showlegend=False
    )
    manufacturing_cost_description = """
    This histogram depicts the distribution of manufacturing costs. Manufacturing costs often represent a significant portion of overall expenses. A wide distribution or high average cost indicates potential for profit margin compression and increased financial risk. Identifying opportunities to reduce or stabilize these costs can significantly improve overall supply chain performance.
    """

    # Box Plot Graph
    df_box = pd.DataFrame({
        'Revenues': revenues,
        'Lead Times': lead_times,
        'Defect Rates': defect_rates,
        'Shipping Costs': shipping_costs_sim,
        'Manufacturing Costs': manufacturing_costs_sim
    })
    df_box_melted = df_box.melt(var_name='Variable', value_name='Value')
    fig_box_plot = px.box(
        df_box_melted, 
        x='Variable', 
        y='Value', 
        color='Variable',
        title="Box Plot of Supply Chain Variables",
        labels={'Value': 'Value', 'Variable': 'Supply Chain Variables'},
        template='plotly_dark'
    )
    fig_box_plot.update_layout(
        yaxis=dict(title='Value'),
        xaxis=dict(title='Supply Chain Variables'),
        showlegend=True
    )
    box_plot_description = """
    This box plot compares the distributions of different supply chain variables. It helps identify which factors have the highest variability and potential outliers. Variables with larger boxes or longer whiskers represent areas of higher risk and uncertainty in the supply chain. This visualization is crucial for prioritizing risk mitigation efforts and identifying areas that require closer monitoring or process improvements.
    """

    # CDF Plot Graph
    fig_cdf = px.ecdf(
        revenues, 
        title='CDF of Revenue Distribution',
        labels={'x': 'Revenue (₹)', 'y': 'Cumulative Probability'},
        template='plotly_dark'
    )
    fig_cdf.update_layout(
        xaxis=dict(title='Revenue (₹)'),
        yaxis=dict(title='Cumulative Probability'),
        showlegend=False
    )
    cdf_description = """
    The Cumulative Distribution Function (CDF) of revenues shows the probability of achieving a certain revenue level or lower. This helps in understanding the likelihood of meeting revenue targets and assessing the risk of falling below certain thresholds. The steepness of the curve indicates the concentration of revenue outcomes. This visualization is valuable for setting realistic revenue goals and understanding the probability of different financial scenarios.
    """

    # Average Contribution of Each Risk Factor
    risk_contributions = {
        'Lead Times': np.mean(lead_times),
        'Defect Rates': np.mean(defect_rates),
        'Shipping Costs': np.mean(shipping_costs_sim),
        'Manufacturing Costs': np.mean(manufacturing_costs_sim)
    }
    fig_bar_chart = px.bar(
        x=list(risk_contributions.keys()), 
        y=list(risk_contributions.values()),
        title="Average Contribution of Each Risk Factor",
        labels={'x': 'Risk Factor', 'y': 'Average Value (₹)'},
        template='plotly_dark'
    )
    fig_bar_chart.update_layout(
        xaxis=dict(title='Risk Factor'),
        yaxis=dict(title='Average Value (₹)', range=[0, max(risk_contributions.values()) * 1.1]),
        bargap=0.2,
        showlegend=False
    )
    bar_chart_description = """
    This bar chart shows the average contribution of each risk factor to the overall supply chain risk. Taller bars indicate risk factors with higher average impact. This visualization helps in identifying which aspects of the supply chain are contributing most significantly to overall risk, allowing for targeted risk mitigation strategies and resource allocation.
    """

    # Defect Rate Histogram
    fig_defect_rate_hist = px.histogram(
        defect_rates, 
        nbins=50, 
        title='Histogram of Defect Rates',
        labels={'x': 'Defect Rate (%)', 'y': 'Frequency'},
        template='plotly_dark'
    )
    fig_defect_rate_hist.update_layout(
        xaxis=dict(range=[defect_rate_min, defect_rate_max], title='Defect Rate (%)'),
        yaxis=dict(title='Frequency'),
        bargap=0.05,
        showlegend=False
    )
    defect_rate_histogram_description = """
    This histogram provides a detailed view of the defect rate distribution. It helps in understanding the frequency of different defect rates and identifying any patterns or anomalies. A distribution skewed towards lower defect rates is desirable, while frequent high defect rates indicate significant quality control issues that need to be addressed to reduce risk and improve customer satisfaction.
    """

    # Pie Chart of Risk Factors
    fig_pie_risk_factors = px.pie(
        names=list(risk_contributions.keys()), 
        values=list(risk_contributions.values()),
        title='Proportional Contribution of Risk Factors',
        template='plotly_dark'
    )
    fig_pie_risk_factors.update_layout(
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", x=1),
        margin=dict(t=100, b=60, l=40, r=40)
    )
    pie_chart_description = """
    This pie chart illustrates the proportional contribution of each risk factor to the overall supply chain risk. It provides a clear visual representation of which factors are dominating the risk landscape. Larger slices indicate areas where risk mitigation efforts might yield the greatest impact on overall supply chain performance and stability.
    """

    # Major Risk Factor Bar Chart
    fig_major_risk = px.bar(
        x=list(risk_contributions.keys()), 
        y=list(risk_contributions.values()),
        title="Major Risk Factor Analysis",
        labels={'x': 'Risk Factor', 'y': 'Impact (₹)'},
        template='plotly_dark'
    )
    fig_major_risk.update_layout(
        xaxis=dict(title='Risk Factor'),
        yaxis=dict(title='Impact (₹)', range=[0, max(risk_contributions.values()) * 1.1]),
        bargap=0.2,
        showlegend=False
    )
    major_risk_factor_description = """
    This bar chart provides a detailed analysis of major risk factors in the supply chain. It allows for a direct comparison of the impact of different risk factors. Taller bars represent risk factors with higher impact, indicating areas where management should focus their attention and resources for risk mitigation. This visualization is crucial for strategic decision-making in supply chain risk management.
    """

    risk_factors = ['Lead Time', 'Defect Rate', 'Shipping Cost', 'Manufacturing Cost']
    impact = [std_lead_time_cost, std_defect_rate, std_shipping_cost, std_manufacturing_cost]
    likelihood = [mean_lead_time_cost, mean_defect_rate, mean_shipping_cost, mean_manufacturing_cost]

    fig_heatmap = go.Figure(data=go.Heatmap(
        z=[impact, likelihood],
        x=risk_factors,
        y=['Impact', 'Likelihood'],
        colorscale='Viridis'
    ))
    fig_heatmap.update_layout(
        title='Risk Heatmap',
        xaxis_title='Risk Factors',
        yaxis_title='Risk Dimensions',
        template='plotly_dark'
    )
    heatmap_description = """
    The Risk Heatmap visualizes the relationship between the impact and likelihood of different risk factors. 
    Darker colors indicate higher risk areas, helping to quickly identify which factors pose the greatest threat to the supply chain.
    This visualization allows for easy prioritization of risk management efforts.
    """

    baseline = mean_revenue
    variations = [std_lead_time_cost, std_defect_rate, std_shipping_cost, std_manufacturing_cost]
    fig_tornado = go.Figure()
    for i, factor in enumerate(risk_factors):
        fig_tornado.add_trace(go.Bar(
            y=[factor],
            x=[variations[i]],
            orientation='h',
            name=f'+1 Std Dev',
            marker_color='red'
        ))
        fig_tornado.add_trace(go.Bar(
            y=[factor],
            x=[-variations[i]],
            orientation='h',
            name=f'-1 Std Dev',
            marker_color='blue'
        ))
    fig_tornado.update_layout(
        title='Tornado Diagram: Impact of Risk Factors on Revenue',
        xaxis_title='Change in Revenue (₹)',
        yaxis_title='Risk Factors',
        barmode='overlay',
        bargap=0.1,
        template='plotly_dark'
    )
    tornado_description = """
    The Tornado Diagram shows how each risk factor impacts revenue when varied by one standard deviation.
    Longer bars indicate factors with a larger potential impact on revenue.
    This helps in understanding which factors have the most significant influence on financial outcomes.
    """

    # Waterfall Chart
    fig_waterfall = go.Figure(go.Waterfall(
        name="20", orientation="v",
        measure=["relative", "relative", "relative", "relative", "total"],
        x=["Lead Time", "Defect Rate", "Shipping Cost", "Manufacturing Cost", "Net Impact"],
        textposition="outside",
        text=[f"₹{-x:,.0f}" for x in variations] + [f"₹{-sum(variations):,.0f}"],
        y=[-x for x in variations] + [-sum(variations)],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
    ))
    fig_waterfall.update_layout(
        title="Waterfall Chart: Cumulative Impact of Risk Factors",
        showlegend=False,
        template='plotly_dark'
    )
    waterfall_description = """
    The Waterfall Chart illustrates how each risk factor contributes to the overall impact on revenue.
    Each bar represents a risk factor, and the final bar shows the cumulative effect.
    This visualization helps in understanding both individual and combined impacts of different risks.
    """

    # Bubble Chart
    fig_bubble = px.scatter(
        x=[mean_lead_time_cost, mean_defect_rate, mean_shipping_cost, mean_manufacturing_cost],
        y=[std_lead_time_cost, std_defect_rate, std_shipping_cost, std_manufacturing_cost],
        size=[mean_lead_time_cost, mean_defect_rate, mean_shipping_cost, mean_manufacturing_cost],
        text=risk_factors,
        labels={'x': 'Mean Cost', 'y': 'Standard Deviation'},
        title='Risk Factor Bubble Chart'
    )
    fig_bubble.update_traces(textposition='top center')
    fig_bubble.update_layout(template='plotly_dark')
    bubble_description = """
    The Bubble Chart plots risk factors based on their mean cost (x-axis) and variability (y-axis).
    The size of each bubble represents the relative impact of the risk factor.
    This visualization helps in identifying which risks are both costly and unpredictable, requiring immediate attention.
    """

    # Scenario Analysis
    scenarios = ['Best Case', 'Expected Case', 'Worst Case']
    best_case = mean_revenue + std_revenue
    worst_case = mean_revenue - std_revenue
    fig_scenario = go.Figure(data=[
        go.Bar(name='Revenue', x=scenarios, y=[best_case, mean_revenue, worst_case])
    ])
    fig_scenario.update_layout(
        title='Scenario Analysis: Revenue Projections',
        xaxis_title='Scenarios',
        yaxis_title='Revenue (₹)',
        template='plotly_dark'
    )
    scenario_description = """
    The Scenario Analysis chart shows revenue projections under different scenarios:
    Best Case (mean + 1 std dev), Expected Case (mean), and Worst Case (mean - 1 std dev).
    This helps in understanding the range of possible outcomes and preparing for different eventualities.
    """

    # Risk Radar Chart
    categories = risk_factors
    fig_radar = go.Figure()

    fig_radar.add_trace(go.Scatterpolar(
        r=[mean_lead_time_cost, mean_defect_rate, mean_shipping_cost, mean_manufacturing_cost],
        theta=categories,
        fill='toself',
        name='Mean'
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=[mean_lead_time_cost + std_lead_time_cost, 
           mean_defect_rate + std_defect_rate, 
           mean_shipping_cost + std_shipping_cost, 
           mean_manufacturing_cost + std_manufacturing_cost],
        theta=categories,
        fill='toself',
        name='Mean + Std Dev'
    ))

    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max([mean_lead_time_cost + std_lead_time_cost, 
                               mean_defect_rate + std_defect_rate, 
                               mean_shipping_cost + std_shipping_cost, 
                               mean_manufacturing_cost + std_manufacturing_cost])]
            )),
        showlegend=True,
        title='Risk Radar Chart',
        template='plotly_dark'
    )
    radar_description = """
    The Risk Radar Chart provides a multi-dimensional view of different risk factors.
    The chart shows both the mean (inner area) and the mean plus one standard deviation (outer area) for each factor.
    This allows for a quick comparison of different risks across multiple dimensions, helping to identify which areas require the most attention.
    """

    # Key Insights
    key_insights = html.Div([
        html.P("1. Highest Risk Factor: Manufacturing cost shows the highest variability and impact on overall risk."),
        html.P("2. Revenue Volatility: There's significant variation in potential revenue outcomes, indicating a need for robust financial planning."),
        html.P("3. Defect Rate Impact: While not the largest cost, defect rates show considerable variability, suggesting a focus area for quality control."),
        html.P("4. Lead Time Consideration: Lead times contribute to overall risk and may need optimization to reduce supply chain uncertainties."),
        html.P("5. Risk Mitigation Priority: Based on the analysis, prioritize efforts on manufacturing cost control, followed by reducing defect rates and optimizing lead times.")
    ])

    return [
        simulation_results_text,
        fig_revenue,
        revenue_description,
        fig_lead_time,
        lead_time_description,
        fig_defect_rate,
        defect_rate_description,
        fig_shipping_cost,
        shipping_cost_description,
        fig_manufacturing_cost,
        manufacturing_cost_description,
        fig_box_plot,
        box_plot_description,
        fig_cdf,
        cdf_description,
        fig_bar_chart,
        bar_chart_description,
        fig_defect_rate_hist,
        defect_rate_histogram_description,
        fig_pie_risk_factors,
        pie_chart_description,
        fig_major_risk,
        major_risk_factor_description,
        fig_heatmap,
        heatmap_description,
        fig_tornado, tornado_description,
            fig_waterfall, waterfall_description, fig_bubble, bubble_description,
            fig_scenario, scenario_description, fig_radar, radar_description, key_insights
    ]

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
