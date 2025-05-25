package optimizer.functions;

import optimizer.GradientDescentOptimizer;

public class QuadraticFunction extends ObjectiveFunction {
    @Override
    public double compute(double[] variables) {
        double sum = 0;
        for (double x : variables) {
            sum += x * x;  // sum of squares
        }
        return sum;
    }

    @Override
    public double[] computeGradient(double[] variables) {
        double[] gradient = new double[variables.length];
        for (int i = 0; i < variables.length; i++) {
            gradient[i] = 2 * variables[i];  // derivative of x^2 is 2x
        }
        return gradient;
    }

    @Override
    public double[] getBounds() {
        return GradientDescentOptimizer.BOUNDS;  // Uses default [-5, 5]
    }

    @Override
    public String getName() {
        return "Quadratic";
    }
}
