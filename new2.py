#app.py
import numpy as np
import pandas as pd
from dash import Dash, html, dcc, Input, Output
import plotly.express as px
import plotly.graph_objs as go
from scipy import stats

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
app = Dash(_name_)

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
            "Advanced Supply Chain Risk Assessment Dashboard",
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
                    dcc.Graph(id="risk-heatmap", style=graph_style),
                    html.Div(id="risk-heatmap-description", style=description_style)
                ], style=graph_container_style),
                html.Div([
                    dcc.Graph(id="tornado-diagram", style=graph_style),
                    html.Div(id="tornado-diagram-description", style=description_style)
                ], style=graph_container_style)
            ],
            style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'space-between'}
        ),
        html.Div(
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
        # Conclusion Section
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
        )
    ]
)

@app.callback(
    [Output('simulation-results', 'children'),
     Output('risk-heatmap', 'figure'),
     Output('risk-heatmap-description', 'children'),
     Output('tornado-diagram', 'figure'),
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

    # Risk Heatmap
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

    # Tornado Diagram
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

    return (simulation_results_text, fig_heatmap, heatmap_description, fig_tornado, tornado_description,
            fig_waterfall, waterfall_description, fig_bubble, bubble_description,
            fig_scenario, scenario_description, fig_radar, radar_description, key_insights)

# Run the Dash app
if _name_ == '_main_':
    app.run_server(debug=True)