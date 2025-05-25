package optimizer.functions;

import optimizer.GradientDescentOptimizer;

public class RastriginFunction extends ObjectiveFunction {
    @Override
    public double compute(double[] variables) {
        double A = 10.0;
        double sum = A * variables.length; // A times dimension
        for (double x : variables) {
            sum += (x * x - A * Math.cos(2 * Math.PI * x));
        }
        return sum;
    }

    @Override
    public double[] computeGradient(double[] variables) {
        double A = 10.0;
        double[] gradient = new double[variables.length];
        for (int i = 0; i < variables.length; i++) {
            gradient[i] = 2 * variables[i] + 2 * Math.PI * A * Math.sin(2 * Math.PI * variables[i]);
        }
        return gradient;
    }

    @Override
    public double[] getBounds() {
        return new double[]{-5.12, 5.12};  // Specific bounds for Rastrigin
    }

    @Override
    public String getName() {
        return "Rastrigin";
    }
}