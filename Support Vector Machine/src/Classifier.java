import java.util.List;

public interface Classifier {
	
	/**
	 * Use the training data to construct a model for the features
	 * @param trainingData
	 */
	public void train(List<List<Double>> trainingData, List<BinaryDataLabel> trainingDataLabels);
	
	/**
	 * @param testData
	 * @return prediction labels using the model constructed in the train method
	 */
	public List<BinaryDataLabel> predict(List<List<Double>> testData);
	
}
