
public abstract class DecisionTreeNode {
	
	private double previousAttributeValue;
	
	public DecisionTreeNode(double previousAttributeValue) {
		this.previousAttributeValue = previousAttributeValue;
	}

	public double getPreviousAttributeValue() {
		return previousAttributeValue;
	}

}
