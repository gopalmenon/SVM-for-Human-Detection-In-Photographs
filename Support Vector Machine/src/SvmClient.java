import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.sql.Timestamp;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;

/**
 * Run the SVM Classifier and report accuracy measures
 */
public class SvmClient {
	
	public static final double TRAINING_DATA_FRACTION = 0.8;
	
	public static final String HUMANS_PRESENT_DATA_FOLDER = "Humans/present";
	public static final String HUMANS_ABSENT_DATA_FOLDER = "Humans/absent";
	
	public static final String HUMANS_PRESENT_RESIZED_DATA_FOLDER = "Resized/present";
	public static final String HUMANS_ABSENT_RESIZED_DATA_FOLDER = "Resized/absent";

	public static final String SVM_RUN_LOG_FILE = "SvmRunLogFile.txt";
	
	private List<List<Double>> trainingData;
	private List<BinaryDataLabel> trainingDataLabels;
	private List<List<Double>> testingData;
	private List<BinaryDataLabel> testingDataLabels;
	
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
		svmClient.retrieveDataFromImageFiles();
		svmClient.runClassifier();
		svmClient.closeRunLog();
		
	}
	
	/**
	 * Retrieve image data from image files
	 */
	@SuppressWarnings("unchecked")
	private void retrieveDataFromImageFiles() {
		
		System.out.println(new Timestamp(System.currentTimeMillis()) + ": Starting data retrieval.");
		
		//Get image data and labels
		List<List<Double>> humansPresentData = DataFileReader.getGrayScaleImageArrays(new File(HUMANS_PRESENT_DATA_FOLDER), new File(HUMANS_PRESENT_RESIZED_DATA_FOLDER));
		List<BinaryDataLabel> humansPresentDataLabels = DataFileReader.getLabelsList(humansPresentData.size(), true);
		List<List<Double>> humansAbsentData = DataFileReader.getGrayScaleImageArrays(new File(HUMANS_ABSENT_DATA_FOLDER), new File(HUMANS_ABSENT_RESIZED_DATA_FOLDER));
		List<BinaryDataLabel> humansAbsentDataLabels = DataFileReader.getLabelsList(humansAbsentData.size(), true);
		
		//Split images and labels data into training and test sets
		Map<String, Object> splitHumansPresentDataAndLabels = DataFileReader.partitionDataAndLabels(humansPresentData, humansPresentDataLabels, TRAINING_DATA_FRACTION);
		Map<String, Object> splitHumansAbsentDataAndLabels = DataFileReader.partitionDataAndLabels(humansAbsentData, humansAbsentDataLabels, TRAINING_DATA_FRACTION);
		
		//Save training and test data
		int numberOfTrainingRecords = (int) (TRAINING_DATA_FRACTION * (humansPresentData.size() + humansAbsentData.size())), numberOfTestingRecords = humansPresentData.size() + humansAbsentData.size() - numberOfTrainingRecords;
		
		this.trainingData = new ArrayList<List<Double>>(numberOfTrainingRecords);
		this.trainingData.addAll((Collection<? extends List<Double>>) splitHumansPresentDataAndLabels.get(DataFileReader.TRAINING_DATA));
		this.trainingData.addAll((Collection<? extends List<Double>>) splitHumansAbsentDataAndLabels.get(DataFileReader.TRAINING_DATA));
		
		this.trainingDataLabels = new ArrayList<BinaryDataLabel>(numberOfTrainingRecords);
		this.trainingDataLabels.addAll((Collection<? extends BinaryDataLabel>) splitHumansPresentDataAndLabels.get(DataFileReader.TRAINING_DATA_LABELS));
		this.trainingDataLabels.addAll((Collection<? extends BinaryDataLabel>) splitHumansAbsentDataAndLabels.get(DataFileReader.TRAINING_DATA_LABELS));
		
		this.testingData = new ArrayList<List<Double>>(numberOfTestingRecords);
		this.testingData.addAll((Collection<? extends List<Double>>) splitHumansPresentDataAndLabels.get(DataFileReader.TESTING_DATA));
		this.testingData.addAll((Collection<? extends List<Double>>) splitHumansAbsentDataAndLabels.get(DataFileReader.TESTING_DATA));
		
		this.testingDataLabels = new ArrayList<BinaryDataLabel>(numberOfTestingRecords);
		this.testingDataLabels.addAll((Collection<? extends BinaryDataLabel>) splitHumansPresentDataAndLabels.get(DataFileReader.TESTING_DATA_LABELS));
		this.testingDataLabels.addAll((Collection<? extends BinaryDataLabel>) splitHumansAbsentDataAndLabels.get(DataFileReader.TESTING_DATA_LABELS));
				
		System.out.println(new Timestamp(System.currentTimeMillis()) + ": Finished with data retrieval.");
		
	}
	
	/**
	 * Run the classifier
	 */
	private void runClassifier() {
		
		
		//Instantiate classifier
		SupportVectorMachine classifier = new SupportVectorMachine();
		
		System.out.println(new Timestamp(System.currentTimeMillis()) + ": Starting training.");
		
		//Train the classifier
		classifier.fit(this.testingData, this.testingDataLabels);
		
		System.out.println(new Timestamp(System.currentTimeMillis()) + ": Starting predictions.");
		
		//Run predictions
		List<BinaryDataLabel> predictions = classifier.getPredictions(this.testingData);
		
		//Write the SVM logs
		classifier.closeLogFile();
		
		//Print predictions
		printPredictionAccuracyMetrics(predictions);

	}
	
	/**
	 * Print prediction accuracy metrics
	 * @param predictions
	 */
	private void printPredictionAccuracyMetrics(List<BinaryDataLabel> predictions) {
		
		ClassifierMetrics classifierMetrics = new ClassifierMetrics(this.testingDataLabels, predictions);
		this.out.println("\nNumber of pictures used in prediction: " + predictions.size());
		this.out.println("Accuracy on test set: " + this.decimalFormat.format(classifierMetrics.getAccuracy()));
		this.out.println("Precision on test set: " + this.decimalFormat.format(classifierMetrics.getPrecision()));
		this.out.println("Recall on test set: " + this.decimalFormat.format(classifierMetrics.getRecall()));
		this.out.println("F1 Score on test set: " + this.decimalFormat.format(classifierMetrics.getF1Score()));	

	}
	
	/**
	 * Close the run log file
	 */
	private void closeRunLog() {
		this.out.close();
	}

	
}
