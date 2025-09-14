# Gradient Descent Optimizer

This repository is designed to help users understand and implement gradient-based optimization algorithms for unconstrained optimization problems. By combining practical implementations with an intuitive visualizer, the project aims to be a comprehensive educational tool for exploring optimization concepts.

## Project Organization

```
multivariate-descent-solver/
├── .gitignore
├── README.md
│
├── optimizer/            
│   ├── functions/        
│   │   ├── ObjectiveFunction.java
│   │   ├── QuadraticFunction.java
│   │   ├── RosenbrockFunction.java
│   │   ├── AckleyFunction.java
│   │   └── RastriginFunction.java
│   │
│   ├── GradientDescentOptimizer.java
│   └── Main.java
│
├── visualizer/           
│   ├── requirements.txt
│   └── app.py
│
└── data/                 
    ├── input.txt        
    └── output.txt       
```

## Test Functions

The repository includes several standard test functions for evaluating optimization algorithms:
- **Quadratic Function**: A simple convex function useful for testing basic behavior.
- **Rosenbrock Function**: A non-convex function challenging optimization algorithms.
- **Ackley Function**: A high-dimensional multimodal function, commonly used to evaluate convergence in rugged error surfaces.
- **Rastrigin Function**: Characterized by numerous local minima, testing an algorithm's robustness.

Each function is fully customizable in terms of dimensionality, parameters, and constraints, enabling flexible experimentation.

## Visualizer

The **Visualizer** offers a dynamic way to observe how the solution progresses over time, giving insights into convergence and search direction. It provides:
- **2D Contour Plots**: The visualizer shows updates to the contour maps as the algorithm progresses, with the trajectory of the algorithm being displayed as it moves toward the optimal solution.
  
![image](https://github.com/user-attachments/assets/f7efb428-ab07-4ea0-839e-562a7705971d)

- **3D Surface Plots**: The visualizer shows updates as 3D surfaces. These plots showcase the function’s behavior in three dimensions, helping to see how the optimization algorithm navigates the landscape.

![image](https://github.com/user-attachments/assets/6152b0d3-7ce3-4ad6-8d99-87ec8707e114)

## Features
1. **Objective Functions:**
   - Quadratic
   - Rosenbrock
   - Ackley
   - Rastirigin
2. **Algorithms:**
   - **Steepest Descent**: Standard gradient descent
   - **Momentum Descent**: Gradient descent with a momentum of $$β$$.
     - Momentum is an enhancement to the gradient descent algorithm that helps the search maintain inertia in a direction, allowing it to smooth out oscillations caused by noisy gradients and glide past flat areas in the search space.
3. **Input Methods:**
   - **Manual input** via console prompts
   - **File inputs** via a configuration file
4. **Output Methods:**
   - **Console**: Iteration-by-iteration logs displayed in the terminal
   - **File**: Logs written to a specified file path
5. **Comprehensive Error Handling**
6. **Visualizer**:
   - A **2D/3D plotting tool** (depending on the function) to visualize each iteration’s position relative to the function’s contour or surface.
   - Shows how the solution progresses over time, giving insights into convergence and search direction.
  
## Usage
Inputs should follow the following order:
```
1) objective function (quadratic, rosenbrock, ackley, rastrigin)
2) algorithm (steepest or momentum)
3) dimensionality (integer)
4) number of iterations (integer)
5) tolerance (double)
6) step size (double)
7) momentum beta (double) - only if algorithm is momentum
8) initial point (space-separated doubles)
```

## Detailed Outline of Gradient Calculations

### 1. Objective Functions
 
**1.1 Quadratic Function:**

- **Definition:**

$$
f(x) = \sum_{i=1}^{n} x_i^2
$$

- **Gradient Calculation:**

$$
\nabla f(x) = (2x_1, 2x_2, \dots, 2x_n)
$$


**1.2 Rosenbrock Function**

- **Definition:**

$$
f(x) = \sum_{i=1}^{n-1} \left[ 100 (x_{i+1} - x_i^2)^2 + (1 - x_i)^2 \right]
$$

- **Gradient Calculation:**

  - For each variable $$\(x_i\)$$ and $$\(x_{i+1}\)$$ respectively:

$$
\frac{\partial f}{\partial x_i} = -400 x_i (x_{i+1} - x_i^2) - 2(1 - x_i)
$$

$$
\frac{\partial f}{\partial x_{i+1}} = 200 (x_{i+1} - x_i^2)
$$

**1.3 Ackley Function**

- **Definition:**

$$
f(x) = -20 \exp \left(-0.2 \sqrt{\frac{1}{n} \sum_{i=1}^{n} x_i^2}\right) - \exp \left(\frac{1}{n}\sum_{i=1}^{n} \cos(2\pi x_i)\right) + 20 + e
$$

- **Gradient Calculation:** 
For each variable $$\(x_j\)$$ and $$r$$ respectively:

$$
\frac{\partial f}{\partial x_j} = -20 \exp \left(-0.2 r\right) \times (-0.2) \frac{x_j}{n r} - \exp \left(\frac{1}{n} \sum_{i=1}^{n} \cos(2\pi x_i)\right) \times \frac{1}{n} \times (-2\pi \sin(2\pi x_j))
$$

$$
r = \sqrt{\frac{1}{n} \sum_{i=1}^{n} x_i^2}
$$

**1.4 Rastrigin Function**

- **Definition:**

$$
f(x) = 10n + \sum_{i=1}^{n} \left[ x_i^2 - 10 \cos(2\pi x_i) \right]
$$

- **Gradient Calculation:**

$$
\frac{\partial f}{\partial x_j} = 2x_j + 20\pi \sin(2\pi x_j)
$$


### 2. Algorithms

**2.1 Steepest Descent**

- **Compute Gradient:**

$$
g = \nabla f(x_{\text{current}})
$$

- **Update Rule** (where $$\(\alpha\)$$ is the step size):

$$
x_{\text{next}} = x_{\text{current}} - \alpha g
$$

- **Check Convergence:** By the norm of the gradient $$\(\|g\|\)$$ against the tolerance.

**2.2 Momentum Descent**

- **Velocity Update** (where $$\(\beta \in [0,1]\)$$ is the momentum parameter):

$$
v_{t+1} = \beta v_t + (1 - \beta) \nabla f(x_t)
$$

- **Position Update:**

$$
x_{t+1} = x_t - \alpha v_{t+1}
$$

- **Check Convergence:** Similarly by gradient norm or iteration limit.




