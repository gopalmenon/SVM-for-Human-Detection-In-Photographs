import java.util.Comparator;


public class DataAndLabelComparator implements Comparator<DataAndLabel> {
	
	public static final int LESS_THAN = -1;
	public static final int GREATER_THAN = 1;
	public static final int EQUAL_TO = 0;

	private int featureNumberToSortOn;
	
	public DataAndLabelComparator(int featureNumberToSortOn) {
		this.featureNumberToSortOn = featureNumberToSortOn;
	}

	@Override
	public int compare(DataAndLabel dataAndLabel1, DataAndLabel dataAndLabel2) {
		
		if (dataAndLabel1.getData().get(this.featureNumberToSortOn) < dataAndLabel2.getData().get(this.featureNumberToSortOn)) {
			return LESS_THAN;
		} else if (dataAndLabel1.getData().get(this.featureNumberToSortOn) > dataAndLabel2.getData().get(this.featureNumberToSortOn)) {
			return GREATER_THAN;
		} else {
			return EQUAL_TO;
		}

	}

}
