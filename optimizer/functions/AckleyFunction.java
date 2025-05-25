package optimizer.functions;

import optimizer.GradientDescentOptimizer;

public class AckleyFunction extends ObjectiveFunction {
    @Override
    public double compute(double[] variables) {
        int n = variables.length;
        double sumSq = 0.0;
        double sumCos = 0.0;
        for (int i = 0; i < n; i++) {
            sumSq += variables[i] * variables[i];
            sumCos += Math.cos(2 * Math.PI * variables[i]);
        }
        double term1 = -20.0 * Math.exp(-0.2 * Math.sqrt(sumSq / n));
        double term2 = -Math.exp(sumCos / n);
        return term1 + term2 + 20.0 + Math.E;
    }

    @Override
    public double[] computeGradient(double[] variables) {
        int n = variables.length;
        double[] grad = new double[n];

        double sumSq = 0.0;
        double sumCos = 0.0;
        for (int i = 0; i < n; i++) {
            sumSq += variables[i] * variables[i];
            sumCos += Math.cos(2 * Math.PI * variables[i]);
        }

        double sqrtPart = Math.sqrt(sumSq / n);
        if (sqrtPart < 1e-14) {
            sqrtPart = 1e-14; 
        }
        double expPart1 = Math.exp(-0.2 * sqrtPart);
        double expPart2 = Math.exp(sumCos / n);

        for (int i = 0; i < n; i++) {
            double x_i = variables[i];
            double part1 = -20.0 * expPart1 * (-0.2) * (x_i / (n * sqrtPart));
            double dsumCos_dxi = -2.0 * Math.PI * Math.sin(2.0 * Math.PI * x_i);
            double part2 = -expPart2 * (1.0 / n) * dsumCos_dxi;
            grad[i] = part1 + part2;
        }

        return grad;
    }

    @Override
    public double[] getBounds() {
        return GradientDescentOptimizer.BOUNDS;  // same [-5, 5] default
    }

    @Override
    public String getName() {
        return "Ackley";
    }
}