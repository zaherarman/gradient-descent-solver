import dash
from dash import dcc, html
import plotly.graph_objs as go
import numpy as np

# Functions for supported objective functions
def rosenbrock(x, y):
    return 100 * (y - x**2)**2 + (1 - x)**2

def quadratic(x, y):
    return x**2 + y**2

def ackley(x, y):
    return (
        -20.0 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2)))
        - np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)))
        + np.e + 20
    )

def rastrigin(x, y):
    A = 10
    return A * 2 + (x**2 - A * np.cos(2 * np.pi * x)) + (y**2 - A * np.cos(2 * np.pi * y))

# Function to extract data from output.txt
def extract_data(file_path):
    x_values = []
    objective_values = []
    function_name = None

    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if "Objective Function:" in line:
                function_name = line.split(":")[1].strip()
            elif "x-values" in line:
                values = list(map(float, line.strip().split(":")[1].split()))
                x_values.append(values)
            elif "Objective Function Value" in line:
                value = float(line.strip().split(":")[1])
                objective_values.append(value)

    return function_name, np.array(x_values), np.array(objective_values)

# Create 2D contour plot
def create_2d_contour_figure(x_values, function_name):
    if function_name == 'Quadratic':
        x_min, x_max = -5, 5
        y_min, y_max = -5, 5
        func = quadratic
    elif function_name == 'Rosenbrock':
        x_min, x_max = -5, 5
        y_min, y_max = -5, 5
        func = rosenbrock
    elif function_name == 'Ackley':
        x_min, x_max = -5, 5
        y_min, y_max = -5, 5
        func = ackley
    elif function_name == 'Rastrigin':
        x_min, x_max = -5.12, 5.12
        y_min, y_max = -5.12, 5.12
        func = rastrigin
    else:
        raise ValueError(f"Unknown function name: {function_name}")

    resolution = 100
    x_lin = np.linspace(x_min, x_max, resolution)
    y_lin = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x_lin, y_lin)
    Z = func(X, Y)

    x_path = x_values[:, 0]
    y_path = x_values[:, 1]

    contour = go.Contour(
        x=x_lin,
        y=y_lin,
        z=Z,
        colorscale='Viridis',
        contours=dict(showlines=False),
        showscale=True
    )

    path = go.Scatter(
        x=x_path,
        y=y_path,
        mode='markers+lines',
        marker=dict(color='red', size=8, symbol='x'),
        line=dict(color='blue', width=2),
        name='Optimization Path'
    )

    layout = go.Layout(
        title=dict(text=f'2D Contour Plot of {function_name} Function', x=0.5),
        xaxis=dict(title='X', showgrid=True, zeroline=True),
        yaxis=dict(title='Y', showgrid=True, zeroline=True),
        height=600,
        margin=dict(l=50, r=50, t=50, b=50),
        template='plotly_dark'
    )

    return go.Figure(data=[contour, path], layout=layout)

# Create 3D surface plot
def create_3d_surface_figure(x_values, objective_values, function_name):
    if function_name == 'Quadratic':
        x_min, x_max = -5, 5
        y_min, y_max = -5, 5
        func = quadratic
    elif function_name == 'Rosenbrock':
        x_min, x_max = -5, 5
        y_min, y_max = -5, 5
        func = rosenbrock
    elif function_name == 'Ackley':
        x_min, x_max = -5, 5
        y_min, y_max = -5, 5
        func = ackley
    elif function_name == 'Rastrigin':
        x_min, x_max = -5.12, 5.12
        y_min, y_max = -5.12, 5.12
        func = rastrigin
    else:
        raise ValueError(f"Unknown function name: {function_name}")

    resolution = 100
    x_lin = np.linspace(x_min, x_max, resolution)
    y_lin = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x_lin, y_lin)
    Z = func(X, Y)

    surface = go.Surface(
        x=X,
        y=Y,
        z=Z,
        colorscale='Viridis',
        opacity=0.8
    )

    scatter_points = go.Scatter3d(
        x=x_values[:, 0],
        y=x_values[:, 1],
        z=objective_values,
        mode='markers+lines',
        marker=dict(color='red', size=4, symbol='x'),
        line=dict(color='blue', width=5),
        name='Optimization Path'
    )

    layout = go.Layout(
        title=dict(text=f'3D Surface Plot of {function_name} Function', x=0.5),
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Objective Function Value')
        ),
        height=700,
        margin=dict(l=50, r=50, t=50, b=50),
        template='plotly_dark'
    )

    return go.Figure(data=[surface, scatter_points], layout=layout)

# Main function
def main():
    file_path = 'data/output.txt'
    function_name, x_values, objective_values = extract_data(file_path)

    fig_2d = create_2d_contour_figure(x_values, function_name)
    fig_3d = create_3d_surface_figure(x_values, objective_values, function_name)

    app = dash.Dash(__name__)
    app.layout = html.Div(
        style={"fontFamily": "Times New Roman, Times New Roman", "backgroundColor": "#1f1f1f", "color": "#e0e0e0", "padding": "20px"},
        children=[
            html.H1(
                'Function Optimization Visualization',
                style={"textAlign": "center", "marginBottom": "30px"}
            ),
            html.Div(
                [
                    html.H2('2D Contour', style={"textAlign": "center", "marginBottom": "20px"}),
                    dcc.Graph(id='2d-contour-graph', figure=fig_2d, style={"margin": "auto", "width": "80%"})
                ],
                style={"marginBottom": "40px"}
            ),
            html.Div(
                [
                    html.H2('3D Surface', style={"textAlign": "center", "marginBottom": "20px"}),
                    dcc.Graph(id='3d-surface-graph', figure=fig_3d, style={"margin": "auto", "width": "80%"})
                ]
            )
        ]
    )

    app.run(debug=True)

if __name__ == '__main__':
    main()
