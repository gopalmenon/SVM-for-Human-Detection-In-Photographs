import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.text.DecimalFormat;
import java.util.Arrays;
import java.util.List;


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
	
	public static final String SVM_RUN_LOG_FILE = "SvmRunLogFile.txt";

	public static List<Double> BASIC_TRADEOFF_HYPERPARAMETER = Arrays.asList(Double.valueOf(1.0));
	public static List<Double> BASIC_LEARNING_RATE_HYPERPARAMETER = Arrays.asList(Double.valueOf(0.01));
	
	private PrintWriter out;
	private DecimalFormat decimalFormat;
	
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
	}
	
	/**
	 * Main method
	 * @param args
	 */
	public static void main(String[] args) {
		
		SvmClient svmClient = new SvmClient();
		svmClient.runHandwritingClassification();
		svmClient.runMadelonClassification();
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
				
		this.out.println("\n" + classificationType + " Accuracy on test set: " + this.decimalFormat.format(classifierMetrics.getAccuracy()));
		this.out.println(classificationType + " Precision on test set: " + this.decimalFormat.format(classifierMetrics.getPrecision()));
		this.out.println(classificationType + " Recall on test set: " + this.decimalFormat.format(classifierMetrics.getRecall()));
		this.out.println(classificationType + " F1 Score on test set: " + this.decimalFormat.format(classifierMetrics.getF1Score()));
		
		//Get predictions for training data
		List<BinaryDataLabel> trainingDataPredictions = svmClassifier.getPredictions(trainingData);
		
		//Compute accuracy measures on test data
		classifierMetrics = new ClassifierMetrics(trainingDataLabels, trainingDataPredictions);
				
		this.out.println("\n" + classificationType + " Accuracy on training set: " + this.decimalFormat.format(classifierMetrics.getAccuracy()));
		this.out.println(classificationType + " Precision on training set: " + this.decimalFormat.format(classifierMetrics.getPrecision()));
		this.out.println(classificationType + " Recall on training set: " + this.decimalFormat.format(classifierMetrics.getRecall()));
		this.out.println(classificationType + " F1 Score on training set: " + this.decimalFormat.format(classifierMetrics.getF1Score()));
		
	}

}
