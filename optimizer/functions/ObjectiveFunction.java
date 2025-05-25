package optimizer.functions;

public abstract class ObjectiveFunction {
    // Computes the value of the objective function.
    public abstract double compute(double[] variables);

    // Computes the gradient of the objective function.
    public abstract double[] computeGradient(double[] variables);

    // Returns the bounds for the variables.
    public abstract double[] getBounds();

    // Returns the name of the objective function.
    public abstract String getName();
}
