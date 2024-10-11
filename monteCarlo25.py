import numpy as np
import pandas as pd
from dash import Dash, html, dcc, Input, Output, State
import plotly.express as px
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Load dataset
data = pd.read_csv('supply_chain_data.csv')

# Key variables for risk analysis
price = data['Price']
products_sold = data['Number of products sold']
defect_rate = data['Defect rates']
shipping_costs = data['Shipping costs']
manufacturing_costs = data['Manufacturing costs']
lead_time = data['Lead times']

# Function to perform Monte Carlo simulation
def monte_carlo_simulation(num_simulations=5000):
    def get_lognorm_params(data_series):
        data_series = data_series[data_series > 0]
        mu = np.mean(np.log(data_series))
        sigma = np.std(np.log(data_series))
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
    revenues = np.maximum(
        0,
        (simulated_prices * adjusted_sales) - 
        (simulated_shipping_costs + simulated_manufacturing_costs + simulated_lead_times * 0.05)
    )

    return (
        revenues,
        simulated_lead_times * 0.05,
        simulated_defect_rates,
        simulated_shipping_costs,
        simulated_manufacturing_costs
    )

# Create the Dash app
app = Dash(__name__)
app.title = "Monte Carlo Simulation Dashboard"

# App layout
app.layout = html.Div([
    html.H1("Monte Carlo Simulation Dashboard", style={'textAlign': 'center'}),
    html.Div([
        html.Label("Number of Simulations:", style={'fontWeight': 'bold'}),
        dcc.Slider(
            id="sim-slider",
            min=1000,
            max=20000,
            step=1000,
            value=5000,
            marks={i: f'{i}' for i in range(1000, 20001, 3000)},
            tooltip={"placement": "bottom", "always_visible": True},
        ),
    ], style={'width': '80%', 'margin': 'auto'}),

    html.Div([
        html.Button('Submit', id='submit-query', n_clicks=0, style={'marginTop': '20px'}),
    ], style={'textAlign': 'center'}),

    html.Div([
        dcc.Textarea(
            id='user-query',
            placeholder='Ask a question...',
            style={'width': '80%', 'height': 100, 'margin': 'auto', 'display': 'block', 'marginTop': '20px'}
        ),
    ]),

    html.Div(
        id='ai-response',
        style={
            'whiteSpace': 'pre-wrap',
            'marginTop': '20px',
            'width': '80%',
            'margin': 'auto',
            'padding': '10px',
            'border': '1px solid #ccc',
            'borderRadius': '5px',
            'backgroundColor': '#f9f9f9'
        }
    ),

    dcc.Graph(id='simulation-graph', style={'marginTop': '40px'}),
], style={'fontFamily': 'Arial, sans-serif', 'padding': '20px'})

# Callback function
@app.callback(
    [Output('ai-response', 'children'),
     Output('simulation-graph', 'figure')],
    [Input('submit-query', 'n_clicks'),
     Input('sim-slider', 'value')],
    [State('user-query', 'value')]
)
def update_ai_response(n_clicks, num_simulations, query):
    # Initialize the AI response
    ai_reply = "Type your question and click submit to get a response."
    
    # Generate AI response if the button is clicked
    if n_clicks > 0:
        if query and query.strip():
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(query)
            ai_reply = response.text if response.text else "No response generated."
        else:
            ai_reply = "Please enter a question before submitting."

    # Monte Carlo simulation
    revenues, lead_times, defect_rates, shipping_costs_sim, manufacturing_costs_sim = monte_carlo_simulation(num_simulations)

    # Create histogram for Revenue Distribution
    fig_revenue = px.histogram(
        revenues,
        nbins=50,
        title='Revenue Distribution',
        labels={'value': 'Revenue'},
        template='plotly_dark',
        color_discrete_sequence=['#2ca02c']
    )

    fig_revenue.update_layout(
        title={
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title='Revenue ($)',
        yaxis_title='Frequency',
        bargap=0.1
    )

    return ai_reply, fig_revenue

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
