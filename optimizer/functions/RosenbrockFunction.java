package optimizer.functions;

import optimizer.GradientDescentOptimizer;

public class RosenbrockFunction extends ObjectiveFunction {
    @Override
    public double compute(double[] variables){
        double sum = 0;
        for(int i = 0; i < variables.length -1; i++){
            sum += 100 * Math.pow(variables[i+1] - Math.pow(variables[i], 2), 2) + Math.pow(1 - variables[i], 2);
        }
        return sum;
    }

    @Override
    public double[] computeGradient(double[] variables) {
        double[] gradient = new double[variables.length];
        for (int i = 0; i < variables.length - 1; i++) {
            // derivative wrt x_i
            gradient[i] = -400 * variables[i] * (variables[i + 1] - Math.pow(variables[i], 2)) - 2 * (1 - variables[i]);
            // derivative wrt x_{i+1}
            gradient[i + 1] = 200 * (variables[i + 1] - Math.pow(variables[i], 2)); 
        }
        return gradient;
    }

    @Override
    public double[] getBounds() {
        return GradientDescentOptimizer.BOUNDS;  // Uses default [-5, 5]
    }

    @Override
    public String getName(){
        return "Rosenbrock";
    }
}