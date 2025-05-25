package optimizer;

import optimizer.functions.*;
import java.util.Scanner;
import java.io.*; 
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.InputMismatchException;

public class GradientDescentOptimizer {
    public static final double[] BOUNDS = {-5.0, 5.0};  // default bounds

    static String objectiveFunctionChoice;
    static String algorithmChoice;
    static int dimensionality;
    static int iterations;
    static double tolerance;
    static double stepSize;
    static double[] variables;

    // Momentum parameter for momentum-based descent
    static double momentumBeta = 0.9; // default

    // For menu prompt toggles
    static int manualInput = -1;
    static int manualOutput = -1;

    // Displays a menu prompt to enter the program or exit
    public static void menuPrompt(Scanner scanner){                
        if (getValidatedInput(scanner, "Press 0 to exit or 1 to enter the program:", Integer.class) == 1){ 
            enteringProgram(scanner);
        } else {
            System.out.println("Exiting program...");
        }
    }

    // This method drives input reading (manual or file) and starts optimization
    public static void enteringProgram(Scanner scanner){
        manualInput = getValidatedInput(scanner, "Press 0 for .txt input or 1 for manual input:", Integer.class);
        manualOutput = getValidatedInput(scanner, "Press 0 for .txt output or 1 for console output:", Integer.class);

        if(manualInput == 1){
            getManualInput(scanner);
        } else {
            getFileInput(scanner);
        }

        ObjectiveFunction objectiveFunction = null;
        if(objectiveFunctionChoice.equals("quadratic")){
            objectiveFunction = new QuadraticFunction();
        } else if (objectiveFunctionChoice.equals("rosenbrock")){
            objectiveFunction = new RosenbrockFunction();
        } else if (objectiveFunctionChoice.equals("ackley")){
            objectiveFunction = new AckleyFunction();
        } else if (objectiveFunctionChoice.equals("rastrigin")) {
            objectiveFunction = new RastriginFunction();
        }

        if (objectiveFunction == null) { 
            System.out.println("No recognized objective function. Exiting...");
            return; 
        }

        if (algorithmChoice.equals("steepest")) {
            optimizeSteepestDescent(objectiveFunction, variables, iterations, tolerance, stepSize, dimensionality, scanner);
        } else if (algorithmChoice.equals("momentum")) {
            optimizeMomentumDescent(objectiveFunction, variables, iterations, tolerance, stepSize, momentumBeta, dimensionality, scanner);
        }
    }

    // Steepest Descent optimization method
    public static void optimizeSteepestDescent(ObjectiveFunction objectiveFunction, double[] variables, int iterations, double tolerance, double stepSize, int dimensionality, Scanner scanner) {
        int iteration = 1;
        double[] changingVariables = variables.clone();  // copy of the initial point

        if (manualOutput == 1) {
            // console output
            consoleOutputIterationLoop(objectiveFunction, changingVariables, iterations, tolerance, stepSize, dimensionality);
        } else if (manualOutput == 0) {
            // file output
            fileOutputIterationLoop(objectiveFunction, changingVariables, iterations, tolerance, stepSize, dimensionality, scanner);
        }
    }

    // Momentum-based gradient descent optimization method
    public static void optimizeMomentumDescent(ObjectiveFunction objectiveFunction, double[] variables, int iterations, double tolerance, double stepSize, double beta, int dimensionality, Scanner scanner) {
        int iteration = 1;
        double[] changingVariables = variables.clone();
        double[] velocity = new double[dimensionality];  // initialize velocity to zero
        for (int i = 0; i < dimensionality; i++) {
            velocity[i] = 0.0;
        }

        if (manualOutput == 1) {
            // console output
            System.out.println(String.format("Objective Function: %s", objectiveFunction.getName()));
            System.out.println("Algorithm: Momentum Descent (beta=" + beta + ")");
            System.out.println("Dimensionality: " + dimensionality);

            System.out.print("Initial Point: ");
            for (double v : variables){
                System.out.print(v + " ");
            }

            System.out.println(String.format("\nIterations: %d\nTolerance: %.5f\nStep Size: %.5f\n", iterations, tolerance, stepSize));
            System.out.println("Optimization process:");    
            System.out.println("Iteration 1:");
            System.out.println(String.format("Objective Function Value: %.5f", objectiveFunction.compute(changingVariables)));
            System.out.println(String.format("x-values: %s\n", xValuesToString(changingVariables)));

            while (iteration < iterations) {
                double[] gradient = objectiveFunction.computeGradient(changingVariables);
                double norm = computeNorm(gradient);

                // momentum update: velocity = beta * velocity + (1 - beta)*gradient
                for (int i = 0; i < dimensionality; i++){
                    velocity[i] = beta * velocity[i] + (1 - beta) * gradient[i];
                    changingVariables[i] = floorTo5Decimals(changingVariables[i] - (stepSize * velocity[i]));
                }

                double objectiveValue = floorTo5Decimals(objectiveFunction.compute(changingVariables));

                System.out.println(String.format("Iteration %d:", iteration + 1));
                System.out.println(String.format("Objective Function Value: %.5f", objectiveValue));
                System.out.println(String.format("x-values: %s", xValuesToString(changingVariables)));
                System.out.println(String.format("Current Tolerance: %.5f\n", norm));

                // check convergence
                if (norm < tolerance) {
                    System.out.println(String.format("Convergence reached after %d iterations.\n", iteration + 1));
                    break;
                }
                iteration++;
            }

            if (iteration == iterations){
                System.out.println("Maximum iterations reached without satisfying the tolerance.\n");
            }
            System.out.println("Optimization process completed.");

        } else if (manualOutput == 0) {
            // file output
            BufferedWriter writer = null;
            System.out.println("Please provide the path for the output file:");
            String filepath = scanner.nextLine();

            try {
                writer = new BufferedWriter(new FileWriter(filepath));

                writer.write(String.format("Objective Function: %s\n", objectiveFunction.getName()));
                writer.write(String.format("Algorithm: Momentum Descent (beta=%f)\n", beta));
                writer.write(String.format("Dimensionality: %d\n", dimensionality));
                writer.write("Initial Point: ");
                for (double variable : variables) {
                    writer.write(variable + " ");
                }
                writer.newLine();
                writer.write(String.format("Iterations: %d\nTolerance: %.5f\nStep Size: %.5f\n\n", iterations, tolerance, stepSize));
                writer.write("Optimization process:\n");
                writer.write("Iteration 1:\n");
                writer.write(String.format("Objective Function Value: %.5f\n", objectiveFunction.compute(changingVariables)));
                writer.write(String.format("x-values: %s\n\n", xValuesToString(changingVariables)));

                while (iteration < iterations) {
                    double[] gradient = objectiveFunction.computeGradient(changingVariables);
                    double norm = computeNorm(gradient);

                    // momentum update
                    for (int i = 0; i < dimensionality; i++){
                        velocity[i] = beta * velocity[i] + (1 - beta) * gradient[i];
                        changingVariables[i] = floorTo5Decimals(changingVariables[i] - (stepSize * velocity[i]));
                    }

                    double objectiveValue = floorTo5Decimals(objectiveFunction.compute(changingVariables));

                    writer.write(String.format("Iteration %d:\n", iteration + 1));
                    writer.write(String.format("Objective Function Value: %.5f\n", objectiveValue));
                    writer.write(String.format("x-values: %s\n", xValuesToString(changingVariables)));
                    writer.write(String.format("Current Tolerance: %.5f\n\n", norm));

                    // check convergence
                    if (norm < tolerance) {
                        writer.write(String.format("Convergence reached after %d iterations.\n\n", iteration + 1));
                        break;
                    }
                    iteration++;
                }

                if (iteration == iterations){
                    writer.write("Maximum iterations reached without satisfying the tolerance.\n\n");
                }
                writer.write("Optimization process completed.");

            } catch(IOException e){
                return;
            } finally {
                if (writer != null){
                    try{
                        writer.close();
                    } catch (IOException e){
                        return;
                    }
                }
            }
        }
    }

    // Console-based loop that iterates for steepest descent
    private static void consoleOutputIterationLoop(ObjectiveFunction objectiveFunction, double[] changingVariables, int iterations, double tolerance, double stepSize, int dimensionality) {
        int iteration = 1;

        System.out.println(String.format("Objective Function: %s", objectiveFunction.getName()));
        System.out.println("Algorithm: Steepest Descent");
        System.out.println("Dimensionality: " + dimensionality);

        System.out.printf("Initial Point: ");
        for (int i = 0; i < changingVariables.length; i++){
            System.out.printf(changingVariables[i] + " ");
        }

        System.out.println(String.format("\nIterations: %d\nTolerance: %.5f\nStep Size: %.5f\n", iterations, tolerance, stepSize));
        System.out.println("Optimization process:");    
        System.out.println("Iteration 1:");
        System.out.println(String.format("Objective Function Value: %.5f", objectiveFunction.compute(changingVariables)));
        System.out.println(String.format("x-values: %s\n", xValuesToString(changingVariables)));

        for (; iteration < iterations; iteration++) {
            double[] gradient = objectiveFunction.computeGradient(changingVariables);
            double norm = computeNorm(gradient);

            for (int i = 0; i < changingVariables.length; i++) {
                changingVariables[i] = floorTo5Decimals(changingVariables[i] - (stepSize * gradient[i]));
            }

            double objectiveValue = floorTo5Decimals(objectiveFunction.compute(changingVariables));

            System.out.println(String.format("Iteration %d:", iteration + 1));
            System.out.println(String.format("Objective Function Value: %.5f", objectiveValue));
            System.out.println(String.format("x-values: %s", xValuesToString(changingVariables)));
            System.out.println(String.format("Current Tolerance: %.5f\n", norm));

            if (norm < tolerance) {
                System.out.println(String.format("Convergence reached after %d iterations.\n", iteration + 1));
                break;
            }
        }

        if (iteration == iterations){
            System.out.println("Maximum iterations reached without satisfying the tolerance.\n");
        }

        System.out.println("Optimization process completed.");
    }

    // File-based loop that iterates for steepest descent
    private static void fileOutputIterationLoop(ObjectiveFunction objectiveFunction, double[] changingVariables, int iterations, double tolerance, double stepSize,int dimensionality, Scanner scanner) {
        int iteration = 1;
        BufferedWriter writer = null;
        System.out.println("Please provide the path for the output file:");
        String filepath = scanner.nextLine();

        try{
            writer = new BufferedWriter(new FileWriter(filepath));
            writer.write(String.format("Objective Function: %s\n", objectiveFunction.getName()));
            writer.write("Algorithm: Steepest Descent\n");
            writer.write(String.format("Dimensionality: %d\n", dimensionality));
            writer.write("Initial Point: ");
            for (double variable : changingVariables) {
                writer.write(variable + " ");
            }
            writer.newLine();
            writer.write(String.format("Iterations: %d\nTolerance: %.5f\nStep Size: %.5f\n\n", iterations, tolerance, stepSize));
            writer.write("Optimization process:\n");
            writer.write("Iteration 1:\n");
            writer.write(String.format("Objective Function Value: %.5f\n", objectiveFunction.compute(changingVariables)));
            writer.write(String.format("x-values: %s\n\n", xValuesToString(changingVariables)));

            for (; iteration < iterations; iteration++) {
                double[] gradient = objectiveFunction.computeGradient(changingVariables);
                double norm = computeNorm(gradient);

                for (int i = 0; i < changingVariables.length; i++) {
                    changingVariables[i] = floorTo5Decimals(changingVariables[i] - (stepSize * gradient[i]));
                }

                double objectiveValue = floorTo5Decimals(objectiveFunction.compute(changingVariables));

                writer.write(String.format("Iteration %d:\n", iteration + 1));
                writer.write(String.format("Objective Function Value: %.5f\n", objectiveValue));
                writer.write(String.format("x-values: %s\n", xValuesToString(changingVariables)));
                writer.write(String.format("Current Tolerance: %.5f\n\n", norm));

                if (norm < tolerance) {
                    writer.write(String.format("Convergence reached after %d iterations.\n\n", iteration + 1));
                    break;
                }
            }

            if (iteration == iterations){
                writer.write("Maximum iterations reached without satisfying the tolerance.\n\n");
            }
            writer.write("Optimization process completed.");

        } catch(IOException e){
            return;
        } finally {
            if (writer != null){
                try{
                    writer.close();
                } catch (IOException e){
                    return;
                }
            }
        }
    }

    // Utility method to round a double value to 5 decimals using FLOOR
    private static double floorTo5Decimals(double value) {
        return new BigDecimal(value).setScale(5, RoundingMode.FLOOR).doubleValue();
    }

    // Computes the norm of a gradient vector
    public static double computeNorm(double[] gradient) {
        double sum = 0;
        for (double g : gradient) {
            sum += g*g;
        }
        return floorTo5Decimals(Math.sqrt(sum));
    }

    // Helper method for printing x-values in a string format
    public static String xValuesToString(double[] array) {
        StringBuilder stringbuilder = new StringBuilder();
        for (double value : array) {
            stringbuilder.append(String.format("%.5f ", value));
        }
        return stringbuilder.toString().trim();
    }

    // Generic method to handle user input validation
    public static <T> T getValidatedInput(Scanner scanner, String prompt, Class<T> type){
        System.out.println(prompt);

        if (prompt.equals("Press 0 to exit or 1 to enter the program:") ||
            prompt.equals("Press 0 for .txt input or 1 for manual input:") ||
            prompt.equals("Press 0 for .txt output or 1 for console output:")) {
            while (true) {
                try {
                    int validIntInput = scanner.nextInt();
                    if (validIntInput == 1 || validIntInput == 0){
                        return type.cast(validIntInput);
                    } else {
                        System.out.println("Please enter a valid input (0 or 1).");
                        System.out.println(prompt);
                    }
                } catch(InputMismatchException e){
                    System.out.println("Please enter a valid input (0 or 1).");
                    System.out.println(prompt);
                    scanner.next();
                }
            }
        }
        else if (prompt.equals("Enter the choice of objective function (quadratic, rosenbrock or ackley):")) {
            String validStringInput = scanner.next().trim().toLowerCase();
            if (validStringInput.equals("quadratic") ||
                validStringInput.equals("rosenbrock") ||
                validStringInput.equals("ackley") ||
                validStringInput.equals("rastrigin")) {
                return type.cast(validStringInput);
            } else {
                return type.cast("Error: Unknown objective function.");
            }
        }
        else if (prompt.equals("Enter the choice of algorithm (steepest or momentum):")) {
            String algo = scanner.next().trim().toLowerCase();
            if (algo.equals("steepest") || algo.equals("momentum")) {
                return type.cast(algo);
            } else {
                return type.cast("Error: Unknown algorithm.");
            }
        }
        else if (prompt.equals("Enter the dimensionality of the problem:") ||
                 prompt.equals("Enter the number of iterations:")) {
            double validDoubleInput = scanner.nextDouble();
            int validIntInput = (int) validDoubleInput;
            return type.cast(validIntInput);
        }
        else if (prompt.equals("Enter the tolerance:") ||
                 prompt.equals("Enter the step size:")) {
            double validDoubleInput = scanner.nextDouble();
            return type.cast(validDoubleInput);
        }
        else if (prompt.equals("Enter the momentum parameter beta (0 to 1):")) {
            double validDoubleInput = scanner.nextDouble();
            return type.cast(validDoubleInput);
        }
        else {
            return type.cast(null);
        }
    }

    // Reads input manually from the console
    public static void getManualInput(Scanner scanner) {
        objectiveFunctionChoice = getValidatedInput(scanner, 
            "Enter the choice of objective function (quadratic, rosenbrock, ackley or rastrigin):", String.class);
        algorithmChoice = getValidatedInput(scanner, 
            "Enter the choice of algorithm (steepest or momentum):", String.class);
        dimensionality = getValidatedInput(scanner, 
            "Enter the dimensionality of the problem:", Integer.class);
        iterations = getValidatedInput(scanner, 
            "Enter the number of iterations:", Integer.class);
        tolerance = getValidatedInput(scanner, 
            "Enter the tolerance:", Double.class);
        stepSize = getValidatedInput(scanner, 
            "Enter the step size:", Double.class);

        if (algorithmChoice.equals("momentum")) {
            momentumBeta = getValidatedInput(scanner, "Enter the momentum parameter beta (0 to 1):", Double.class);
        }

        if (objectiveFunctionChoice.equals("Error: Unknown objective function.")) {
            System.out.print(objectiveFunctionChoice);
            return;
        }
        if (algorithmChoice.equals("Error: Unknown algorithm.")) {
            System.out.print(algorithmChoice);
            return;
        }

        variables = new double[dimensionality];
        System.out.println(String.format("Enter the initial point as %d space-separated values:", dimensionality));
        String x = scanner.nextLine();
        if (x.trim().length() == 0) {
            x = scanner.nextLine();
        }
        String[] variablesStrings = x.trim().split("\\s+");
        if (variablesStrings.length != dimensionality){
            System.out.println("Error: Initial point dimensionality mismatch.");
            System.exit(1);
        } else {
            for(int i = 0; i < variablesStrings.length; i++){
                variables[i] = Double.parseDouble(variablesStrings[i]);
            }
            checkBounds(variables, BOUNDS);
        }
    }

    // Reads input from a .txt config file
    public static void getFileInput(Scanner scanner) {
        BufferedReader reader = null;
        scanner.nextLine();
        System.out.println("Please provide the path to the config file:");
        String pathfile = scanner.nextLine();

        try {
            reader = new BufferedReader(new FileReader(pathfile));
            String line; 
            int count = 0;
            while((line = reader.readLine()) != null){
                line = line.trim();
                count++;
                switch(count){
                    case 1:
                        if(line.equals("quadratic") || line.equals("rosenbrock") || line.equals("ackley") || line.equals("rastrigin")){
                            objectiveFunctionChoice = line;
                        } else {
                            System.out.println("Error: Unknown objective function.");
                            System.exit(1);
                        }
                        break;
                    case 2:
                        if(line.equals("steepest") || line.equals("momentum")){
                            algorithmChoice = line;
                        } else {
                            System.out.println("Error: Unknown algorithm.");
                            System.exit(1);
                        }
                        break;
                    case 3:
                        dimensionality = Integer.parseInt(line);
                        break;
                    case 4:
                        iterations = Integer.parseInt(line);
                        break;
                    case 5:
                        tolerance = Double.parseDouble(line);
                        break;
                    case 6:
                        stepSize = Double.parseDouble(line);
                        break;
                    case 7:
                        if(algorithmChoice.equals("momentum")) {
                            momentumBeta = Double.parseDouble(line);
                        }
                        break;
                    case 8:
                        variables = new double[dimensionality];
                        String[] variablesStrings = line.split("\\s+");
                        if (variablesStrings.length != dimensionality){
                            System.out.println("Error: Initial point dimensionality mismatch.");
                            System.exit(1);
                        } else {
                            for(int i = 0; i < variablesStrings.length; i++){
                                variables[i] = Double.parseDouble(variablesStrings[i]);
                            }
                            checkBounds(variables, BOUNDS);
                        }
                        break;
                }
            }
            reader.close();
        } catch (IOException e){
            System.out.println("Error reading the file.");
            System.exit(1);
        } finally {
            if (reader != null){
                try{
                    reader.close();
                } catch(IOException e){
                    return;
                }
            }
        }
    }

    // Checks that the initial points are within the given bounds
    public static void checkBounds(double[] variables, double[] BOUNDS){
        for(int i = 0; i < variables.length; i++){
            if(variables[i] < BOUNDS[0] || variables[i] > BOUNDS[1]){
                System.out.println(String.format(
                    "Error: Initial point %s is outside the bounds [-5.0, 5.0].", variables[i]));
                System.exit(1);
            }
        } 
    }
}
