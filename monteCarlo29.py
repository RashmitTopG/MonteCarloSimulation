import numpy as np
import pandas as pd
from dash import Dash, html, dcc, Input, Output, State
import plotly.express as px
import os
import base64
from dotenv import load_dotenv
import google.generativeai as genai
from dash.exceptions import PreventUpdate

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

# Initialize the Dash app
app = Dash(__name__)

# Monte Carlo simulation placeholder function (use your existing simulation logic)
def update_simulation(num_simulations):
    revenues = np.random.normal(loc=100000, scale=15000, size=num_simulations)
    lead_times = np.random.normal(loc=30, scale=5, size=num_simulations)
    defect_rates = np.random.normal(loc=0.05, scale=0.01, size=num_simulations)
    shipping_costs_sim = np.random.normal(loc=5000, scale=1000, size=num_simulations)
    manufacturing_costs_sim = np.random.normal(loc=20000, scale=3000, size=num_simulations)

    fig_revenue = px.histogram(revenues, nbins=50, title='Revenue Distribution')
    fig_lead_time = px.histogram(lead_times, nbins=50, title='Lead Time Distribution')
    fig_defect_rate = px.histogram(defect_rates, nbins=50, title='Defect Rate Distribution')
    fig_shipping_cost = px.histogram(shipping_costs_sim, nbins=50, title='Shipping Cost Distribution')
    fig_manufacturing_cost = px.histogram(manufacturing_costs_sim, nbins=50, title='Manufacturing Cost Distribution')

    return ("Simulation completed.", fig_revenue, fig_lead_time, fig_defect_rate, fig_shipping_cost, fig_manufacturing_cost)

# AI assistant function that generates responses
def update_ai_response(n_clicks, num_simulations, query, image_contents):
    ai_reply = "Type your question and click submit to get a response."
    image_url = None

    if n_clicks > 0:
        if query and query.strip():
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(query)
            ai_reply = response.text if response.text else "No response generated."
        elif image_contents:
            ai_reply = "Image received. Analyzing the image..."
        else:
            ai_reply = "Please enter a question or upload an image before submitting."

    (simulation_results_text, fig_revenue, fig_lead_time, fig_defect_rate, fig_shipping_cost, fig_manufacturing_cost) = update_simulation(num_simulations)

    return (simulation_results_text, fig_revenue, fig_lead_time, fig_defect_rate, fig_shipping_cost, fig_manufacturing_cost, ai_reply, image_url)

# Function to decode the uploaded image to a base64 string
def parse_uploaded_image(contents):
    content_type, content_string = contents.split(',')
    decoded_image = base64.b64decode(content_string)
    return html.Img(src=contents, style={'maxWidth': '100%', 'borderRadius': '10px'})

# Dash layout
app.layout = html.Div(
    style={'padding': '20px', 'fontFamily': 'Arial'},
    children=[
        html.H1("Supply Chain Simulation with AI Assistant", style={'textAlign': 'center', 'marginBottom': '40px'}),

        # Text area for user to ask the AI assistant
        html.Div(
            style={
                'backgroundColor': '#f1f1f1',
                'padding': '20px',
                'borderRadius': '5px',
                'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)',
                'marginBottom': '20px'
            },
            children=[
                html.H2("Ask AI Assistant", style={'textAlign': 'center', 'marginBottom': '20px'}),
                dcc.Textarea(
                    id='user-query',
                    placeholder='Type your question here...',
                    style={
                        'width': '100%',
                        'height': '100px',
                        'marginBottom': '20px',
                        'padding': '10px',
                        'fontSize': '1em',
                        'borderRadius': '5px',
                        'border': '1px solid #ccc'
                    }
                ),
                html.Button(
                    'Submit',
                    id='submit-query',
                    n_clicks=0,
                    style={
                        'display': 'block',
                        'margin': '0 auto 20px auto',
                        'padding': '10px 20px',
                        'fontSize': '1em',
                        'borderRadius': '5px',
                        'border': 'none',
                        'backgroundColor': '#17a2b8',
                        'color': '#fff',
                        'cursor': 'pointer'
                    }
                ),
                html.Div(
                    id='ai-response',
                    style={
                        'backgroundColor': '#ffffff',
                        'padding': '20px',
                        'border': '1px solid #ccc',
                        'borderRadius': '5px',
                        'minHeight': '100px',
                        'boxShadow': '0 2px 4px rgba(0, 0, 0, 0.1)',
                        'fontSize': '1em'
                    }
                ),
                
                # Image upload area
                dcc.Upload(
                    id='upload-image',
                    children=html.Div([
                        'Drag and Drop or ',
                        html.A('Select or Paste an Image Here')
                    ]),
                    style={
                        'width': '100%',
                        'height': '60px',
                        'lineHeight': '60px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'marginBottom': '20px'
                    },
                    multiple=False  # Single image upload
                ),
                
                # Display uploaded image
                html.Div(id='uploaded-image')
            ]
        )
    ]
)

# Dash callback to handle AI query and display results, including uploaded image
@app.callback(
    [Output('ai-response', 'children'),
     Output('uploaded-image', 'children')],
    [Input('submit-query', 'n_clicks'),
     Input('upload-image', 'contents')],
    [State('user-query', 'value')]
)
def handle_query(n_clicks, image_contents, query):
    num_simulations = 100  # You can change this or make it dynamic

    # If no clicks or query, raise PreventUpdate
    if n_clicks == 0 and not image_contents:
        raise PreventUpdate

    # Call update_ai_response to get AI reply
    (simulation_results_text, fig_revenue, fig_lead_time, fig_defect_rate, fig_shipping_cost, fig_manufacturing_cost, ai_reply, image_url) = update_ai_response(n_clicks, num_simulations, query, image_contents)

    # Handle AI text response
    response_text = html.P(ai_reply)
    
    # Handle uploaded image
    if image_contents:
        uploaded_image_component = parse_uploaded_image(image_contents)
    else:
        uploaded_image_component = None
    
    return response_text, uploaded_image_component

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
