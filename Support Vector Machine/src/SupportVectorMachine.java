import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * Support Vector Machine implementation
 */
public class SupportVectorMachine {

	public static final int DEFAULT_NUMBER_OF_EPOCHS = 5, DEFAULT_CROSS_VALIDATION_SPLITS = 6;
	public static final List<Double> DEFAULT_LEARNING_RATES = Arrays.asList(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0);
	public static final List<Double> DEFAULT_TRADEOFF_VALUES = Arrays.asList(Math.pow(2.0, 0.0), Math.pow(2.0, 1.0), Math.pow(2.0, 2.0), Math.pow(2.0, 3.0), Math.pow(2.0, 4.0), Math.pow(2.0, 5.0));
	public static final int MINIMUM_SHUFFLES = 10;
	
	private int numberOfEpochsForTraining;
	private int crossValidationSplits;
	private List<Double> learningRatesForTraining, tradeoffValuesForTraining;
	private Kernel kernel;
	private List<Double> weightVector;
	private Random randomNumberGenerator;
	
	/**
	 * Constructor using default values
	 */
	public SupportVectorMachine() {
		this(DEFAULT_NUMBER_OF_EPOCHS, DEFAULT_CROSS_VALIDATION_SPLITS, DEFAULT_LEARNING_RATES, DEFAULT_TRADEOFF_VALUES, new IdentityKernel());
	}
	
	/**
	 * Constructor
	 * 
	 * @param numberOfEpochsForTraining
	 * @param learningRatesForTraining
	 * @param tradeoffValuesForTraining
	 * @param kernel
	 */
	public SupportVectorMachine(int numberOfEpochsForTraining, int crossValidationSplits, List<Double> learningRatesForTraining, List<Double> tradeoffValuesForTraining, Kernel kernel) {
		
		this.numberOfEpochsForTraining = numberOfEpochsForTraining;
		this.crossValidationSplits = crossValidationSplits;
		this.learningRatesForTraining = learningRatesForTraining;
		this.tradeoffValuesForTraining = tradeoffValuesForTraining;
		this.kernel = kernel;
		this.weightVector = new ArrayList<Double>();
		
	}
	
	public void fit(List<List<Double>> featureVectors, List<BinaryDataLabel> trainingDataLabels) {
		
		boolean firstTime = true;
		for (int epochCounter = 0; epochCounter < this.numberOfEpochsForTraining; ++ epochCounter) {
			
			//Shuffle the training data for each subsequent epoch
			if (firstTime) {
				firstTime = false;
			} else {
				shuffleTrainingData(featureVectors, trainingDataLabels);
			}
		
		
		
		}
		
	}
	
	/**
	 * @param testingData
	 * @return prediction labels
	 */
	public List<BinaryDataLabel> getPredictions(List<List<Double>> testingData) {
		
		List<BinaryDataLabel> predictionLabels = new ArrayList<BinaryDataLabel>(testingData.size());
		
		for (List<Double> testVector : testingData) {			
			predictionLabels.add(getPrediction(testVector));
		}
		
		return predictionLabels;
		
	}
	
	/**
	 * @param testVector
	 * @return prediction label
	 */
	public BinaryDataLabel getPrediction(List<Double> testVector) {
		
		if (Kernel.getDotProduct(this.weightVector, adjustForBias(testVector)) >= 0) {
			return BinaryDataLabel.POSITIVE_LABEL;
		} else {
			return BinaryDataLabel.NEGATIVE_LABEL;
		}
		
	}
	
	/**
	 * @param inputVector
	 * @return vector with first term as 1 to account for the bias
	 */
	private List<Double> adjustForBias(List<Double> inputVector) {
		
		List<Double> vectorAdjustedForBias = new ArrayList<Double>(inputVector.size() + 1);
		
		vectorAdjustedForBias.add(1.0);
		for (Double feature : inputVector) {
			vectorAdjustedForBias.add(feature);
		}
		
		return vectorAdjustedForBias;
		
	}
	
	/**
	 * Shuffle the labels and features together
	 * @param featureVectors
	 * @param trainingDataLabels
	 */
	private void shuffleTrainingData(List<List<Double>> featureVectors, List<BinaryDataLabel> trainingDataLabels) {
		
		//Generate a random number for the number of times to shuffle the data  
		int numberOfTimesToSuffle = this.randomNumberGenerator.nextInt(MINIMUM_SHUFFLES + trainingDataLabels.size() / 2), swapContentsWith1 = 0, swapContentsWith2 = 0;
		BinaryDataLabel tempLabel;
		List<Double> tempFeatureVector = null;
		
		//Shuffle the data
		for (int shuffleCounter = 0; shuffleCounter < numberOfTimesToSuffle; ++shuffleCounter) {
			
			//Randomly generate the row numbers to shuffle
			swapContentsWith1 = this.randomNumberGenerator.nextInt(trainingDataLabels.size());
			swapContentsWith2 = this.randomNumberGenerator.nextInt(trainingDataLabels.size());
			
			//Swap the contents
			if (swapContentsWith1 != swapContentsWith2) {
				
				tempLabel = trainingDataLabels.get(swapContentsWith1);
				tempFeatureVector = featureVectors.get(swapContentsWith1);
				
				trainingDataLabels.set(swapContentsWith1, trainingDataLabels.get(swapContentsWith2));
				featureVectors.set(swapContentsWith1, featureVectors.get(swapContentsWith2));
				
				trainingDataLabels.set(swapContentsWith2, tempLabel);
				featureVectors.set(swapContentsWith2, tempFeatureVector);
				
			}
			
		}
		
	}
	
}
