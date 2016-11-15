import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;


/**
 * Run the SVM Classifier and report accuracy measures
 */
public class SvmClient {

	public static final String HANDWRITING_CLASSIFICATION = "Handwriting";
	public static final String MADELON_CLASSIFICATION = "Madelon";
	
	public static final String HANDWRITING_TRAINING_DATA_FILE = "handwriting/train.data";
	public static final String HANDWRITING_TRAINING_LABELS_FILE = "handwriting/train.labels";
	public static final String HANDWRITING_TEST_DATA_FILE = "handwriting/test.data";
	public static final String HANDWRITING_TEST_LABELS_FILE = "handwriting/test.labels";

	public static final String MADELON_TRAINING_DATA_FILE = "madelon/madelon_train.data";
	public static final String MADELON_TRAINING_LABELS_FILE = "madelon/madelon_train.labels";
	public static final String MADELON_TEST_DATA_FILE = "madelon/madelon_test.data";
	public static final String MADELON_TEST_LABELS_FILE = "madelon/madelon_test.labels";
	
	public static final int NUMBER_OF_HANDWRITING_CLASSIFICATION_DECISION_TREES = 5;
	public static final List<Integer> NUMBER_OF_MADELON_CLASSIFICATION_DECISION_TREES = Arrays.asList(10, 30, 100);
	public static final double MINIMUM_FRACTION_OF_DATA_TO_USE = 0.20;
	
	public static final String SVM_RUN_LOG_FILE = "SvmRunLogFile.txt";

	public static List<Double> BASIC_TRADEOFF_HYPERPARAMETER = Arrays.asList(Double.valueOf(1.0));
	public static List<Double> BASIC_LEARNING_RATE_HYPERPARAMETER = Arrays.asList(Double.valueOf(0.01));
	
	private PrintWriter out;
	private DecimalFormat decimalFormat;
	private Random randomNumberGenerator;
	
	/**
	 * Constructor
	 */
	public SvmClient() {
		try{
			this.out = new PrintWriter(new FileWriter(SVM_RUN_LOG_FILE));
		} catch (IOException e) {
			System.err.println("IOException while opening file ");
			e.printStackTrace();
			System.exit(0);
		}
		
		this.decimalFormat = new DecimalFormat("0.0000");
		this.randomNumberGenerator = new Random(0);
	}
	
	/**
	 * Main method
	 * @param args
	 */
	public static void main(String[] args) {
		
		SvmClient svmClient = new SvmClient();
		svmClient.runHandwritingClassification();
		svmClient.runMadelonClassification();
		svmClient.runHandwritingEnsembleClassification();
		svmClient.runMadelonEnsembleClassification();
		svmClient.closeRunLog();
		
	}
	
	/**
	 * Close the run log file
	 */
	private void closeRunLog() {
		this.out.close();
	}
	
	/**
	 * Call handwriting classification to report accuracy measures
	 */
	private void runHandwritingClassification() {
		runClassification(HANDWRITING_CLASSIFICATION, HANDWRITING_TRAINING_DATA_FILE, HANDWRITING_TRAINING_LABELS_FILE, HANDWRITING_TEST_DATA_FILE, HANDWRITING_TEST_LABELS_FILE, true);
	}
	
	/**
	 * Call madelon classification to report accuracy measures
	 */
	private void runMadelonClassification() {
		runClassification(MADELON_CLASSIFICATION, MADELON_TRAINING_DATA_FILE, MADELON_TRAINING_LABELS_FILE, MADELON_TEST_DATA_FILE, MADELON_TEST_LABELS_FILE, false);
	}
	
	/**
	 * Run classification and report accuracy measures
	 * @param classificationType
	 * @param trainingDataFile
	 * @param trainingLabelsFile
	 * @param testDataFile
	 * @param testLabelsFile
	 */
	private void runClassification(String classificationType, String trainingDataFile, String trainingLabelsFile, String testDataFile, String testLabelsFile, boolean useBasicHyperParameters) {
			
		//Get training data and labels
		List<List<Double>> trainingData = DataFileReader.getData(trainingDataFile);
		List<BinaryDataLabel> trainingDataLabels = DataFileReader.getLabels(trainingLabelsFile);
		
		//Train the classifier
		SupportVectorMachine svmClassifier = new SupportVectorMachine(SupportVectorMachine.DEFAULT_NUMBER_OF_EPOCHS, SupportVectorMachine.DEFAULT_CROSS_VALIDATION_SPLITS, useBasicHyperParameters ? BASIC_LEARNING_RATE_HYPERPARAMETER : SupportVectorMachine.DEFAULT_LEARNING_RATES, useBasicHyperParameters ? BASIC_TRADEOFF_HYPERPARAMETER : SupportVectorMachine.DEFAULT_TRADEOFF_VALUES, new IdentityKernel(), true);
		svmClassifier.fit(trainingData, trainingDataLabels);

		//Get test data and labels
		List<List<Double>> testData = DataFileReader.getData(testDataFile);
		List<BinaryDataLabel> testDataLabels = DataFileReader.getLabels(testLabelsFile);
		
		//Get predictions for test data
		List<BinaryDataLabel> testDataPredictions = svmClassifier.getPredictions(testData);
		
		//Compute accuracy measures on test data
		ClassifierMetrics classifierMetrics = new ClassifierMetrics(testDataLabels, testDataPredictions);
				
		this.out.println("\n" + classificationType + " using SVM");
		this.out.println("Accuracy on test set: " + this.decimalFormat.format(classifierMetrics.getAccuracy()));
		this.out.println("Precision on test set: " + this.decimalFormat.format(classifierMetrics.getPrecision()));
		this.out.println("Recall on test set: " + this.decimalFormat.format(classifierMetrics.getRecall()));
		this.out.println("F1 Score on test set: " + this.decimalFormat.format(classifierMetrics.getF1Score()));
		
		//Get predictions for training data
		List<BinaryDataLabel> trainingDataPredictions = svmClassifier.getPredictions(trainingData);
		
		//Compute accuracy measures on test data
		classifierMetrics = new ClassifierMetrics(trainingDataLabels, trainingDataPredictions);
				
		this.out.println("\n" + classificationType + " using SVM");
		this.out.println("Accuracy on training set: " + this.decimalFormat.format(classifierMetrics.getAccuracy()));
		this.out.println("Precision on training set: " + this.decimalFormat.format(classifierMetrics.getPrecision()));
		this.out.println("Recall on training set: " + this.decimalFormat.format(classifierMetrics.getRecall()));
		this.out.println("F1 Score on training set: " + this.decimalFormat.format(classifierMetrics.getF1Score()));
		
		svmClassifier.closeLogFile();
		
	}

	/**
	 * Run ensemble classification on the handwriting data set
	 */
	private void runHandwritingEnsembleClassification() {
		runEnsembleClassification(HANDWRITING_CLASSIFICATION, HANDWRITING_TRAINING_DATA_FILE, HANDWRITING_TRAINING_LABELS_FILE, HANDWRITING_TEST_DATA_FILE, HANDWRITING_TEST_LABELS_FILE, true, NUMBER_OF_HANDWRITING_CLASSIFICATION_DECISION_TREES, false);
	}

	/**
	 * Run ensemble classification on the madelon data set
	 */
	private void runMadelonEnsembleClassification() {
		
		for (Integer numberOfDecisionTrees : NUMBER_OF_MADELON_CLASSIFICATION_DECISION_TREES) {
		
			runEnsembleClassification(MADELON_CLASSIFICATION, MADELON_TRAINING_DATA_FILE, MADELON_TRAINING_LABELS_FILE, MADELON_TEST_DATA_FILE, MADELON_TEST_LABELS_FILE, false, numberOfDecisionTrees, true);

		}
	}
	
	/**
	 * Run SVM classification based on outputs from decision trees
	 * 
	 * @param classificationType
	 * @param trainingDataFile
	 * @param trainingLabelsFile
	 * @param testDataFile
	 * @param testLabelsFile
	 * @param useBasicHyperParameters
	 * @param numberOfDecisionTrees
	 * @param continuousFeatures
	 */
	private void runEnsembleClassification(String classificationType, String trainingDataFile, String trainingLabelsFile, String testDataFile, String testLabelsFile, boolean useBasicHyperParameters, int numberOfDecisionTrees, boolean continuousFeatures) {
		
		
		//Get training data and labels
		List<List<Double>> trainingData = DataFileReader.getData(trainingDataFile);
		List<BinaryDataLabel> trainingDataLabels = DataFileReader.getLabels(trainingLabelsFile);

		//Get test data and labels
		List<List<Double>> testData = DataFileReader.getData(testDataFile);
		List<BinaryDataLabel> testDataLabels = DataFileReader.getLabels(testLabelsFile);
	
		List<DataAndLabel> combinedDataAndLabels = DataAndLabel.getCombinDataAndLabels(trainingData, trainingDataLabels);
		List<DecisionTreeId3BinaryClassifier> decisionTreeList = new ArrayList<DecisionTreeId3BinaryClassifier>(numberOfDecisionTrees);
		List<List<BinaryDataLabel>> predictions = new ArrayList<List<BinaryDataLabel>>(numberOfDecisionTrees);
		
		//Create decision trees, train them and make predictions
		for (int treeCounter = 0; treeCounter < numberOfDecisionTrees; ++treeCounter) {
			
			decisionTreeList.add(new DecisionTreeId3BinaryClassifier(Integer.MAX_VALUE, continuousFeatures));
			List<DataAndLabel> trainingDataCopy = new ArrayList<DataAndLabel>(combinedDataAndLabels);
			List<DataAndLabel> trainingDataSamples = getTrainingDataSamples(trainingDataCopy);
			decisionTreeList.get(treeCounter).train(DataAndLabel.getData(trainingDataSamples), DataAndLabel.getLabels(trainingDataSamples));
			predictions.add(decisionTreeList.get(treeCounter).predict(trainingData));
			
		}
		
		List<List<Double>> transformedFeatures = new ArrayList<List<Double>>();
		List<BinaryDataLabel> firstTransformedFeature = predictions.get(0);
		
		//Create transformed feature vectors that are obtained from the forest of the decision trees
		int transformedFeatureVectorNumber = 0;
		for (BinaryDataLabel binaryDataLabel : firstTransformedFeature) {
		
			List<Double> transformedFeatureVector = new ArrayList<Double>();
			transformedFeatureVector.add(Double.valueOf(binaryDataLabel.getValue()));
			
			for (int transformedFeatureVectorElementCounter = 1; transformedFeatureVectorElementCounter < predictions.size(); ++transformedFeatureVectorElementCounter) {
			
				transformedFeatureVector.add(Double.valueOf(predictions.get(transformedFeatureVectorElementCounter).get(transformedFeatureVectorNumber).getValue()));
				
			}
			
			transformedFeatures.add(transformedFeatureVector);
			++transformedFeatureVectorNumber;
		}
		
		//Train an SVM on the transformed features
		SupportVectorMachine svmClassifier = new SupportVectorMachine(SupportVectorMachine.DEFAULT_NUMBER_OF_EPOCHS, SupportVectorMachine.DEFAULT_CROSS_VALIDATION_SPLITS, useBasicHyperParameters ? BASIC_LEARNING_RATE_HYPERPARAMETER : SupportVectorMachine.DEFAULT_LEARNING_RATES, useBasicHyperParameters ? BASIC_TRADEOFF_HYPERPARAMETER : SupportVectorMachine.DEFAULT_TRADEOFF_VALUES, new IdentityKernel(), false);
		svmClassifier.fit(transformedFeatures, trainingDataLabels);
		
		//Use the tree forest to make predictions based on the test data
		predictions.clear();
		for (int treeCounter = 0; treeCounter < numberOfDecisionTrees; ++treeCounter) {
			predictions.add(decisionTreeList.get(treeCounter).predict(testData));
		}
		
		//Create transformed feature vectors that are obtained from the forest of the decision trees
		transformedFeatureVectorNumber = 0;
		transformedFeatures.clear();
		firstTransformedFeature = predictions.get(0);
		for (BinaryDataLabel binaryDataLabel : firstTransformedFeature) {
		
			List<Double> transformedFeatureVector = new ArrayList<Double>();
			transformedFeatureVector.add(Double.valueOf(binaryDataLabel.getValue()));
			
			for (int transformedFeatureVectorElementCounter = 1; transformedFeatureVectorElementCounter < predictions.size(); ++transformedFeatureVectorElementCounter) {
			
				transformedFeatureVector.add(Double.valueOf(predictions.get(transformedFeatureVectorElementCounter).get(transformedFeatureVectorNumber).getValue()));
				
			}
			
			transformedFeatures.add(transformedFeatureVector);
			++transformedFeatureVectorNumber;
		}
		
		//Predict using the trained SVM Classifier
		List<BinaryDataLabel> finalPredictions = svmClassifier.getPredictions(testData);
		ClassifierMetrics classifierMetrics = new ClassifierMetrics(testDataLabels, finalPredictions);
		
		this.out.println("\n" + classificationType + " using forest of " + numberOfDecisionTrees + " trees");
		this.out.println("Accuracy on test set: " + this.decimalFormat.format(classifierMetrics.getAccuracy()));
		this.out.println("Precision on test set: " + this.decimalFormat.format(classifierMetrics.getPrecision()));
		this.out.println("Recall on test set: " + this.decimalFormat.format(classifierMetrics.getRecall()));
		this.out.println("F1 Score on test set: " + this.decimalFormat.format(classifierMetrics.getF1Score()));
		
		svmClassifier.closeLogFile();		
		
	}
	
	/**
	 * @param combinedDataAndLabels
	 * @return training data samples
	 */
	private List<DataAndLabel> getTrainingDataSamples(List<DataAndLabel> combinedDataAndLabels) {
		
		int minimumNumberOfTrainingRecordsToUse = (int) (MINIMUM_FRACTION_OF_DATA_TO_USE * combinedDataAndLabels.size());
		int numberOfTrainingDataSamplesToDraw = minimumNumberOfTrainingRecordsToUse + this.randomNumberGenerator.nextInt(combinedDataAndLabels.size() - minimumNumberOfTrainingRecordsToUse);
		
		List<DataAndLabel> trainingDataSamples = new ArrayList<DataAndLabel>();
		
		//Sample training data
		int recordCounter = 0, dataIndex = 0;
		while (recordCounter < numberOfTrainingDataSamplesToDraw) {
			
			dataIndex = this.randomNumberGenerator.nextInt(combinedDataAndLabels.size());
			trainingDataSamples.add(combinedDataAndLabels.get(dataIndex));
			combinedDataAndLabels.remove(dataIndex);
			++recordCounter;
		}
		
		return trainingDataSamples;
		
	}
	

}
