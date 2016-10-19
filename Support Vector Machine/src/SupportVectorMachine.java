import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * Support Vector Machine implementation
 */
public class SupportVectorMachine {

	public static final int DEFAULT_NUMBER_OF_EPOCHS = 20, DEFAULT_CROSS_VALIDATION_SPLITS = 5, NUMBER_OF_CROSS_VALIDATION_FOLDS = 6, MINIMUM_SHUFFLES = 100;
	public static final List<Double> DEFAULT_LEARNING_RATES = Arrays.asList(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0);
	public static final List<Double> DEFAULT_TRADEOFF_VALUES = Arrays.asList(Math.pow(2.0, 0.0), Math.pow(2.0, 1.0), Math.pow(2.0, 2.0), Math.pow(2.0, 3.0), Math.pow(2.0, 4.0), Math.pow(2.0, 5.0));
	
	private int numberOfEpochsForTraining;
	private int crossValidationSplits;
	private List<Double> learningRatesForTraining, tradeoffValuesForTraining;
	private Kernel kernel;
	private List<Double> weightVector;
	private Random randomNumberGenerator;
	private int stochasticGradientDescentCounter;
	
	/**
	 * Constructor using default values
	 */
	public SupportVectorMachine() {
		this(DEFAULT_NUMBER_OF_EPOCHS, DEFAULT_CROSS_VALIDATION_SPLITS, DEFAULT_LEARNING_RATES, DEFAULT_TRADEOFF_VALUES, new IdentityKernel());
	}
	
	/**
	 * Constructor
	 * @param numberOfEpochsForTraining
	 * @param crossValidationSplits
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
		this.randomNumberGenerator = new Random(0);
	}
	
	/**
	 * Train the SVM
	 * @param featureVectors
	 * @param trainingDataLabels
	 */
	public void fit(List<List<Double>> featureVectors, List<BinaryDataLabel> trainingDataLabels) {
		
		boolean firstTime = true;
		double currentAccuracy = 0.0, maximumAccuracy = Double.MIN_VALUE;
				
		//Run through multiple learning rates
		for (Double learningRate : this.learningRatesForTraining) {
			List<Double> weightVector = null;
			
			//Run through multiple tradeoff values
			for (Double tradeoffValue : this.tradeoffValuesForTraining) {
		
				//Run k-fold cross validation
				double averageAccuracy = 0.0;
				
				List<FeaturesAndLabels> crossValidationData = getCrossValidationData(this.crossValidationSplits, featureVectors, trainingDataLabels);

				for (int crossValidationCounter = 0; crossValidationCounter < this.crossValidationSplits; ++crossValidationCounter) {
				
					//Load training and testing data
					List<List<Double>> trainingDataSubsetFeatures = new ArrayList<List<Double>>();
					List<BinaryDataLabel> trainingDataSubsetLabels = new ArrayList<BinaryDataLabel>();
					
					List<List<Double>> testingDataSubsetFeatures = new ArrayList<List<Double>>();
					List<BinaryDataLabel> testingDataSubsetLabels = new ArrayList<BinaryDataLabel>();
					
					int splitCounter = 0;
					for (FeaturesAndLabels featuresAndLabels : crossValidationData) {
					
						if (splitCounter == crossValidationCounter) {
							testingDataSubsetFeatures.addAll(featuresAndLabels.getFeatureVectors());
							testingDataSubsetLabels.addAll(featuresAndLabels.getLabels());
						} else {
							trainingDataSubsetFeatures.addAll(featuresAndLabels.getFeatureVectors());
							trainingDataSubsetLabels.addAll(featuresAndLabels.getLabels());
						}
					
						++splitCounter;
						
					}
					
					weightVector = getZeroWeightVector(trainingDataSubsetFeatures.get(1));
					
					//Run through multiple epochs
					this.stochasticGradientDescentCounter = 0;
					for (int epochCounter = 0; epochCounter < this.numberOfEpochsForTraining; ++ epochCounter) {

						//Shuffle the training data for each subsequent epoch
						if (firstTime) {
							firstTime = false;
						} else {
							shuffleTrainingData(trainingDataSubsetFeatures, trainingDataSubsetLabels);
						}
						
						//Find the optimum weights by running stochastic gradient descent
						weightVector = runStochasticGradientDescent(trainingDataSubsetFeatures, trainingDataSubsetLabels, learningRate.doubleValue(), tradeoffValue.doubleValue(), weightVector);
					
					}
					
					//Use the weight vector to run predictions
					List<BinaryDataLabel> predictions = getPredictions(testingDataSubsetFeatures, weightVector);
					
					//Get accuracy for current settings
					currentAccuracy = new ClassifierMetrics(testingDataSubsetLabels, predictions).getAccuracy();
					averageAccuracy += currentAccuracy;
					
				}
				
				//If this is the most accurate classification save the weight vector
				averageAccuracy /= this.crossValidationSplits;
				if (averageAccuracy > maximumAccuracy) {
					maximumAccuracy = averageAccuracy;
					this.weightVector = weightVector;
				}
			}
		
		}
		
	}
	
	/**
	 * @param stochasticGradientDescentCounter
	 * @param originalLearningRate
	 * @param tradeoffValue
	 * @return next learning rate
	 */
	private double getNextLearningRate(int stochasticGraientDescentCounter, double originalLearningRate, double tradeoffValue) {
		return originalLearningRate / (1 + (originalLearningRate * stochasticGraientDescentCounter / tradeoffValue));
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
	 * This method is used while doing cross-validation
	 * @param testingData
	 * @param weightVector
	 * @return predictions
	 */
	public List<BinaryDataLabel> getPredictions(List<List<Double>> testingData, List<Double> weightVector) {
		
		List<BinaryDataLabel> predictionLabels = new ArrayList<BinaryDataLabel>(testingData.size());
		
		for (List<Double> testVector : testingData) {
			if (kernel.getDotProductInFeatureSpace(weightVector, adjustForBias(testVector)) >= 0) {
				predictionLabels.add(BinaryDataLabel.POSITIVE_LABEL);
			} else {
				predictionLabels.add(BinaryDataLabel.NEGATIVE_LABEL);
			}			
		}
		
		return predictionLabels;
		
	}	

	/**
	 * @param testVector
	 * @return prediction label
	 */
	public BinaryDataLabel getPrediction(List<Double> testVector) {
		
		if (kernel.getDotProductInFeatureSpace(this.weightVector, adjustForBias(testVector)) >= 0) {
			return BinaryDataLabel.POSITIVE_LABEL;
		} else {
			return BinaryDataLabel.NEGATIVE_LABEL;
		}
		
	}
	
	/**
	 * @param featureVector
	 * @return a zero weight vector
	 */
	private List<Double>  getZeroWeightVector(List<Double> featureVector) {
		
		List<Double> weightVector = new ArrayList<Double>();
		
		for (int weightVectorIndex = 0; weightVectorIndex <= featureVector.size(); ++weightVectorIndex) {
			weightVector.add(Double.valueOf(0.0));
		}
		
		return weightVector;
		
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
	
	/**
	 * @param numberOfCrossValidationFolds
	 * @param featuresVector
	 * @param labels
	 * @return list of features and labels lists
	 */
	private List<FeaturesAndLabels> getCrossValidationData(int numberOfCrossValidationFolds, List<List<Double>> featureVectors, List<BinaryDataLabel> labels) {
				
		List<FeaturesAndLabels> crossValidationData = new ArrayList<FeaturesAndLabels>(numberOfCrossValidationFolds);
		
		List<List<Double>> featuresVectorCopy = new ArrayList<List<Double>>(featureVectors);
		List<BinaryDataLabel> labelsCopy = new ArrayList<BinaryDataLabel>(labels);
		
		int numberOfCrossValidationDataRecords = labels.size() / numberOfCrossValidationFolds, randomRecordNumber = 0;
		
		//Create one less than the required number of splits
		for (int splitCounter = 0; splitCounter < numberOfCrossValidationFolds - 1; ++splitCounter) {

			List<List<Double>> featureVectorsSubset = new ArrayList<List<Double>>(numberOfCrossValidationDataRecords);
			List<BinaryDataLabel> labelsSubset = new ArrayList<BinaryDataLabel>(numberOfCrossValidationDataRecords);

			//Fill data required for cross validation split
			for (int recordCounter = 0; recordCounter < numberOfCrossValidationDataRecords; ++recordCounter) {
				
				randomRecordNumber = randomNumberGenerator.nextInt(labelsCopy.size());
				
				featureVectorsSubset.add(featuresVectorCopy.get(randomRecordNumber));
				featuresVectorCopy.remove(randomRecordNumber);
				
				labelsSubset.add(labelsCopy.get(randomRecordNumber));
				labelsCopy.remove(randomRecordNumber);

			}
			
			crossValidationData.add(new FeaturesAndLabels(featureVectorsSubset, labelsSubset));
			
		}
		
		//Add the remaining labels and features to the last split
		crossValidationData.add(new FeaturesAndLabels(featuresVectorCopy, labelsCopy));
		
		//Return the data
		return crossValidationData;
		
	} 
	
	/**
	 * @param trainingDataSubsetFeatures
	 * @param trainingDataSubsetLabels
	 * @param learningRate
	 * @param tradeoffValue
	 * @param weightVector
	 * @return
	 */
	private List<Double> runStochasticGradientDescent(List<List<Double>> trainingDataSubsetFeatures, List<BinaryDataLabel> trainingDataSubsetLabels, double learningRate, double tradeoffValue, List<Double> weightVector) {
		
		//Loop through each training record sample
		int featureVectorCounter = 0;
		double nextLearningRate = 0.0;
		for (List<Double> featureVector : trainingDataSubsetFeatures) {
		
			nextLearningRate = getNextLearningRate(this.stochasticGradientDescentCounter++, learningRate, tradeoffValue);
			
			if (trainingDataSubsetLabels.get(featureVectorCounter).getValue() * this.kernel.getDotProductInFeatureSpace(weightVector, adjustForBias(featureVector)) <= 1) {
				weightVector = getSumOfVectors(multiplyWithVector(1 - nextLearningRate, weightVector), multiplyWithVector(nextLearningRate * tradeoffValue * trainingDataSubsetLabels.get(featureVectorCounter).getValue(), adjustForBias(featureVector)));
			} else {
				weightVector = multiplyWithVector(1 - nextLearningRate, weightVector);
			}
			
			++featureVectorCounter;
			
		}
		
		return weightVector;
		
	}
	
	/**
	 * @param multiplyWith
	 * @param weightVector
	 * @return a vector that is the result of multiplying the input vector with a factor
	 */
	private List<Double> multiplyWithVector(double multiplyWith, List<Double> weightVector) {
		
		List<Double> newWeightVector = new ArrayList<Double>(weightVector.size());
		
		for (Double feature : weightVector) {
			newWeightVector.add(Double.valueOf(feature.doubleValue() * multiplyWith));
		}
		
		return newWeightVector;
		
	}
	
	/**
	 * @param vector1
	 * @param vector2
	 * @return the sum of two vectors
	 */
	private List<Double> getSumOfVectors(List<Double> vector1, List<Double> vector2) {
		
		assert vector1.size() == vector2.size();
		List<Double> sumVector = new ArrayList<Double>(weightVector.size());
		
		int vectorIndex = 0;
		for (Double vectorElement : vector1) {
			sumVector.add(Double.valueOf(vectorElement.doubleValue() + vector2.get(vectorIndex++).doubleValue()));
		}
		
		return sumVector;
		
	}
	
}
