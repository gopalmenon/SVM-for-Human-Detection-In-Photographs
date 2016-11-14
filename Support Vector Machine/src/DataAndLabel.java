import java.util.ArrayList;
import java.util.List;


public class DataAndLabel {
	
	private List<Double> data;
	private BinaryDataLabel label;
	
	public DataAndLabel(List<Double> data, BinaryDataLabel label) {
		this.data = data;
		this.label = label;
	}
	
	public List<Double> getData() {
		return data;
	}

	public BinaryDataLabel getLabel() {
		return label;
	}

	/**
	 * @param data
	 * @param labels
	 * @return combined list
	 */
	public static List<DataAndLabel> getCombinDataAndLabels(List<List<Double>> data, List<BinaryDataLabel> labels) {
		
		int dataCounter = 0;
		List<DataAndLabel> combinedList = new ArrayList<DataAndLabel>();
		DataAndLabel combinedEntry = null;
		
		for (List<Double> datavector : data) {
		
			combinedEntry = new DataAndLabel(datavector, labels.get(dataCounter));
			combinedList.add(combinedEntry);
			++dataCounter;
			
		}
		
		return combinedList;
		
	}
	
	/**
	 * @param datandLabels
	 * @return data list
	 */
	public static List<List<Double>> getData(List<DataAndLabel> datandLabels) {
		
		List<List<Double>> data = new ArrayList<List<Double>>(datandLabels.size());
		
		for (DataAndLabel dataAndLabel : datandLabels) {
			data.add(dataAndLabel.getData());
		}
		
		return data;
		
	}

	/**
	 * @param datandLabels
	 * @return labels list
	 */
	public static List<BinaryDataLabel> getLabels(List<DataAndLabel> datandLabels) {
		
		List<BinaryDataLabel> labels = new ArrayList<BinaryDataLabel>(datandLabels.size());
		
		for (DataAndLabel dataAndLabel : datandLabels) {
			labels.add(dataAndLabel.getLabel());
		}
		
		return labels;
		
	}
}
