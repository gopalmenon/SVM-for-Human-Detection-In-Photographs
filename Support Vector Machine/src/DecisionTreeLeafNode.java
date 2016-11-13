
public class DecisionTreeLeafNode extends DecisionTreeNode {

	private BinaryDataLabel label;
	
	public DecisionTreeLeafNode(double previousAttributeValue, BinaryDataLabel label) {
		super(previousAttributeValue);
		this.label = label;
	}

	public BinaryDataLabel getLabel() {
		return label;
	}

}
