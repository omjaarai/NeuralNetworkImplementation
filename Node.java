import java.util.*;

/**
 * Class for internal organization of a Neural Network. There are 5 types of nodes. Check the type
 * attribute of the node for details. Feel free to modify the provided function signatures to fit
 * your own implementation
 */

public class Node {
    private int type = 0; // 0=input,1=biasToHidden,2=hidden,3=biasToOutput,4=Output
    public ArrayList<NodeWeightPair> parents = null; // Array List that will contain the parents
                                                     // (including the bias node) with weights if
                                                     // applicable

    private double inputValue = 0.0;
    private double outputValue = 0.0;
    private double outputGradient = 0.0;
    private double delta = 0.0; // input gradient

    // Create a node with a specific type
    Node(int type) {
        if (type > 4 || type < 0) {
            System.out.println("Incorrect value for node type");
            System.exit(1);

        } else {
            this.type = type;
        }

        if (type == 2 || type == 4) {
            parents = new ArrayList<>();
        }
    }

    // For an input node sets the input value which will be the value of a particular attribute
    public void setInput(double inputValue) {
        if (type == 0) { // If input node
            this.inputValue = inputValue;
        }
    }

    /**
     * Calculate the output of a node. You can get this value by using getOutput()
     */
    public void calculateOutput(double denominator) {
        if (type == 2 || type == 4) { // Not an input or bias node
            // TODO: add code here
            double z = 0;
            for (NodeWeightPair nwp : parents) {
                z = z + nwp.node.getOutput() * nwp.weight;
            }
            if (type == 2) {
                outputValue = Math.max(0, z);
            } else {
                outputValue = java.lang.Math.exp(z) / denominator;
            }
        }
    }

    public double recLUder() {
        double z = 0;
        for (NodeWeightPair nwp : parents) {
            z = z + nwp.node.getOutput() * nwp.weight;
        }
        if (z <= 0)
            return 0;

        return 1;
    }

    // Gets the output value
    public double getOutput() {

        if (type == 0) { // Input node
            return inputValue;
        } else if (type == 1 || type == 3) { // Bias node
            return 1.00;
        } else {
            return outputValue;
        }

    }

    public double softMaxNum() {
        double z = 0;
        for (NodeWeightPair nwp : parents) {
            z += nwp.node.getOutput() * nwp.weight;
        }
        return java.lang.Math.exp(z);
    }

    // Calculate the delta value of a node.
    public void calculateDelta(double sum, double target) {
        if (type == 2 || type == 4) {
            // TODO: add code here
            if (type == 2) {
                delta = recLUder() * sum;
            } else if (type == 4) {
                delta = target - outputValue;
            }
        }
    }

    public double getDelta() {
        return delta;
    }

    // Update the weights between parents node and current node
    public void updateWeight(double learningRate) {
        if (type == 2 || type == 4) {
            for (NodeWeightPair nwp : parents) {
                nwp.weight += learningRate * nwp.node.getOutput() * delta;
            }
        }
    }

    public int getType() {
        return type;
    }
}


