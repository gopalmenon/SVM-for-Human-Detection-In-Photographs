
public abstract class DecisionTreeNode {
	
	boolean lessThanPreviousThreshold;
	private double previousThresholdValue;
	
	public DecisionTreeNode(boolean lessThanPreviousThreshold, double previousThresholdValue) {
		this.lessThanPreviousThreshold = lessThanPreviousThreshold;
		this.previousThresholdValue = previousThresholdValue;
	}

	public boolean isLessThanPreviousThreshold() {
		return lessThanPreviousThreshold;
	}

	public double getPreviousThresholdValue() {
		return previousThresholdValue;
	}

}
