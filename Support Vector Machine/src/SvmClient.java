import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.sql.Timestamp;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Random;


/**
 * Run the SVM Classifier and report accuracy measures
 */
public class SvmClient {
	
	public static final double TRAINING_DATA_FRACTION = 0.8;
	
	public static final String HUMANS_PRESENT_DATA_FILE = "Humans/present";
	public static final String HUMANS_ABSENT_DATA_FILE = "Humans/absent";
	
	public static final String SVM_RUN_LOG_FILE = "SvmRunLogFile.txt";
	
	private List<List<Double>> trainingData;
	private List<BinaryDataLabel> trainingDataLabels;
	private List<List<Double>> testingData;
	private List<BinaryDataLabel> testingDataLabels;
	
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
		svmClient.retrieveDataFromImageFiles();
		svmClient.closeRunLog();
		
	}
	
	/**
	 * Retrieve image data from image files
	 */
	@SuppressWarnings("unchecked")
	private void retrieveDataFromImageFiles() {
		
		//Get image data and labels
		List<List<Double>> humansPresentData = DataFileReader.getGrayScaleImageArrays(new File(HUMANS_PRESENT_DATA_FILE));
		List<BinaryDataLabel> humansPresentDataLabels = DataFileReader.getLabelsList(humansPresentData.size(), true);
		List<List<Double>> humansAbsentData = DataFileReader.getGrayScaleImageArrays(new File(HUMANS_ABSENT_DATA_FILE));
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
		
	}
	
	/**
	 * Close the run log file
	 */
	private void closeRunLog() {
		this.out.close();
	}

	
}
