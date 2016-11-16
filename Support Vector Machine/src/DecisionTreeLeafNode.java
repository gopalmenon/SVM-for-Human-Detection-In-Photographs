
public class DecisionTreeLeafNode extends DecisionTreeNode {

	private BinaryDataLabel label;
	private boolean ignoreThreshold;
	
	public DecisionTreeLeafNode(boolean lessThanPreviousThreshold, double previousThresholdValue, boolean ignoreThreshold, BinaryDataLabel label) {
		super(lessThanPreviousThreshold, previousThresholdValue);
		this.label = label;
		this.ignoreThreshold = ignoreThreshold;
	}

	public BinaryDataLabel getLabel() {
		return label;
	}

	public boolean isIgnoreThreshold() {
		return ignoreThreshold;
	}

}
