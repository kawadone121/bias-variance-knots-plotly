import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Function to compute the weighting matrix for given knots
def get_weighting_matrix(knots, save_csv=""):
  n_knots = len(knots)
  W = np.zeros((T, n_knots))  # Initialize the weighting matrix with zeros

  # Compute weights for each time step between knots
  for smaller_neighbour, larger_neighbour in zip(knots[:-1], knots[1:]):
    for t in np.arange(smaller_neighbour, larger_neighbour + 1):
      w = (larger_neighbour - t) / (larger_neighbour - smaller_neighbour)
      W[t, knots.index(smaller_neighbour)] = w
      W[t, knots.index(larger_neighbour)] = 1 - w

  # Save the weighting matrix to a CSV file if specified
  if save_csv:
    pd.DataFrame(W).to_csv('./csv/W_{}knots.csv'.format(n_knots), header=['knot_{}'.format(knot) for knot in knots])

  return W

# Function to compute knot values using the weighting matrix and observed data
def get_knot_values(W, y):
  return np.linalg.inv(W.T @ W) @ W.T @ y

# Set random seed for reproducibility
np.random.seed(0)

# Define the total number of time steps
T = 1001
t = np.arange(T)  # Time steps

# Generate synthetic data: seasonality, trend, and noise
seasonality = 2 * np.sin(2 * np.pi * t / 365) + 1 * np.cos(2 * np.pi * t / 365)
trend = 0.1 * t ** 0.6
noise = np.random.normal(0, 1, T)
mu = seasonality + trend  # True underlying signal
y = mu + noise  # Observed data with noise

# Define different sets of knots
knots_4 = [0, 333, 666, 1000]
knots_6 = [0, 200, 400, 600, 800, 1000]
knots_11 = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
knots_51 = np.linspace(0, 1000, 51).astype('int').tolist()
knots_101 = np.linspace(0, 1000, 101).astype('int').tolist()
knots_201 = np.linspace(0, 1000, 201).astype('int').tolist()

# Compute weighting matrices for each set of knots
W_4 = get_weighting_matrix(knots_4, save_csv=True)
W_6 = get_weighting_matrix(knots_6, save_csv=True)
W_11 = get_weighting_matrix(knots_11, save_csv=True)
W_51 = get_weighting_matrix(knots_51, save_csv=True)
W_101 = get_weighting_matrix(knots_101, save_csv=True)
W_201 = get_weighting_matrix(knots_201, save_csv=True)

# Compute knot values for each set of knots
knot_values_4 = get_knot_values(W_4, y)
knot_values_6 = get_knot_values(W_6, y)
knot_values_11 = get_knot_values(W_11, y)
knot_values_51 = get_knot_values(W_51, y)
knot_values_101 = get_knot_values(W_101, y)
knot_values_201 = get_knot_values(W_201, y)

# Create an interactive plot using Plotly
fig = go.Figure()

# Add traces for observed data and true signal
fig.add_trace(go.Scatter(x=t, y=y, mode='lines', name='y', line=dict(color='grey')))
fig.add_trace(go.Scatter(x=t, y=mu, mode='lines', name='mu', line=dict(color='black')))

# Add traces for approximations with different numbers of knots
fig.add_trace(go.Scatter(x=t, y=W_4 @ knot_values_4, mode='lines', name='4 knots', visible='legendonly', line=dict(color='orange')))
fig.add_trace(go.Scatter(x=t, y=W_6 @ knot_values_6, mode='lines', name='6 knots', visible='legendonly', line=dict(color='yellow')))
fig.add_trace(go.Scatter(x=t, y=W_11 @ knot_values_11, mode='lines', name='11 knots', visible='legendonly', line=dict(color='green')))
fig.add_trace(go.Scatter(x=t, y=W_51 @ knot_values_51, mode='lines', name='51 knots', visible='legendonly', line=dict(color='purple')))
fig.add_trace(go.Scatter(x=t, y=W_101 @ knot_values_101, mode='lines', name='101 knots', visible='legendonly', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=t, y=W_201 @ knot_values_201, mode='lines', name='201 knots', visible='legendonly', line=dict(color='red')))

# Add interactive buttons to toggle visibility of traces
fig.update_layout(
  title='Bias-Variance Trade-off in Number of Knots',
  updatemenus=[
    dict(
      active=0,
      buttons=list([
        dict(label='All',
           method='update',
           args=[{'visible': [True, True, True, True, True, True, True, True]}]),
        dict(label='4 knots',
           method='update',
           args=[{'visible': [True, True, True, False, False, False, False, False]}]),
        dict(label='6 knots',
           method='update',
           args=[{'visible': [True, True, False, True, False, False, False, False]}]),
        dict(label='11 knots',
           method='update',
           args=[{'visible': [True, True, False, False, True, False, False, False]}]),
        dict(label='51 knots',
           method='update',
           args=[{'visible': [True, True, False, False, False, True, False, False]}]),
        dict(label='101 knots',
           method='update',
           args=[{'visible': [True, True, False, False, False, False, True, False]}]),
        dict(label='201 knots',
           method='update',
           args=[{'visible': [True, True, False, False, False, False, False, True]}]),
      ]),
    )
  ])

# Display the plot
fig.show()

# Save the plot as an HTML file
fig.write_html("./html/Bias-Variance Trade-off in Number of Knots.html")
