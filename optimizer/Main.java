package optimizer;

import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in); 
        GradientDescentOptimizer.menuPrompt(scanner); 
        scanner.close();
    }
}
