import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


public class ClassifierClient {

	public static void main(String[] args) {
		
		ClassifierClient classifierClient = new ClassifierClient();
		classifierClient.runWith1dFeatures();
		//classifierClient.runWith2dFeatures();
		
	}

	private void runWith1dFeatures() {
		
		List<List<Double>> features = new ArrayList<List<Double>>();
		features.add(Arrays.asList(Double.valueOf(1.0)));
		features.add(Arrays.asList(Double.valueOf(2.0)));
		features.add(Arrays.asList(Double.valueOf(3.0)));
		features.add(Arrays.asList(Double.valueOf(4.0)));
		features.add(Arrays.asList(Double.valueOf(5.0)));
		features.add(Arrays.asList(Double.valueOf(6.0)));
		features.add(Arrays.asList(Double.valueOf(7.0)));
		features.add(Arrays.asList(Double.valueOf(8.0)));
		features.add(Arrays.asList(Double.valueOf(9.0)));
		features.add(Arrays.asList(Double.valueOf(10.0)));
		features.add(Arrays.asList(Double.valueOf(11.0)));
		features.add(Arrays.asList(Double.valueOf(12.0)));
		features.add(Arrays.asList(Double.valueOf(13.0)));
		features.add(Arrays.asList(Double.valueOf(14.0)));
		features.add(Arrays.asList(Double.valueOf(15.0)));
		features.add(Arrays.asList(Double.valueOf(16.0)));
		features.add(Arrays.asList(Double.valueOf(17.0)));
		features.add(Arrays.asList(Double.valueOf(18.0)));
		features.add(Arrays.asList(Double.valueOf(19.0)));
		features.add(Arrays.asList(Double.valueOf(20.0)));
		features.add(Arrays.asList(Double.valueOf(21.0)));
		features.add(Arrays.asList(Double.valueOf(22.0)));
		features.add(Arrays.asList(Double.valueOf(23.0)));
		features.add(Arrays.asList(Double.valueOf(24.0)));
		features.add(Arrays.asList(Double.valueOf(25.0)));
		features.add(Arrays.asList(Double.valueOf(26.0)));
		features.add(Arrays.asList(Double.valueOf(27.0)));
		features.add(Arrays.asList(Double.valueOf(28.0)));
		features.add(Arrays.asList(Double.valueOf(29.0)));
		features.add(Arrays.asList(Double.valueOf(30.0)));
		
		features.add(Arrays.asList(Double.valueOf(50.0)));
		features.add(Arrays.asList(Double.valueOf(51.0)));
		features.add(Arrays.asList(Double.valueOf(52.0)));
		features.add(Arrays.asList(Double.valueOf(53.0)));
		features.add(Arrays.asList(Double.valueOf(54.0)));
		features.add(Arrays.asList(Double.valueOf(55.0)));
		features.add(Arrays.asList(Double.valueOf(56.0)));
		features.add(Arrays.asList(Double.valueOf(57.0)));
		features.add(Arrays.asList(Double.valueOf(58.0)));
		features.add(Arrays.asList(Double.valueOf(59.0)));
		features.add(Arrays.asList(Double.valueOf(60.0)));
		features.add(Arrays.asList(Double.valueOf(61.0)));
		features.add(Arrays.asList(Double.valueOf(62.0)));
		features.add(Arrays.asList(Double.valueOf(63.0)));
		features.add(Arrays.asList(Double.valueOf(64.0)));
		features.add(Arrays.asList(Double.valueOf(65.0)));
		features.add(Arrays.asList(Double.valueOf(66.0)));
		features.add(Arrays.asList(Double.valueOf(67.0)));
		features.add(Arrays.asList(Double.valueOf(68.0)));
		features.add(Arrays.asList(Double.valueOf(69.0)));
		features.add(Arrays.asList(Double.valueOf(70.0)));
		features.add(Arrays.asList(Double.valueOf(71.0)));
		features.add(Arrays.asList(Double.valueOf(72.0)));
		features.add(Arrays.asList(Double.valueOf(73.0)));
		features.add(Arrays.asList(Double.valueOf(74.0)));
		features.add(Arrays.asList(Double.valueOf(75.0)));
		features.add(Arrays.asList(Double.valueOf(76.0)));
		features.add(Arrays.asList(Double.valueOf(77.0)));
		features.add(Arrays.asList(Double.valueOf(78.0)));
		features.add(Arrays.asList(Double.valueOf(79.0)));
		
		List<BinaryDataLabel> labels = new ArrayList<BinaryDataLabel>();
		labels.add(BinaryDataLabel.POSITIVE_LABEL);
		labels.add(BinaryDataLabel.POSITIVE_LABEL);
		labels.add(BinaryDataLabel.POSITIVE_LABEL);
		labels.add(BinaryDataLabel.POSITIVE_LABEL);
		labels.add(BinaryDataLabel.POSITIVE_LABEL);
		labels.add(BinaryDataLabel.POSITIVE_LABEL);
		labels.add(BinaryDataLabel.POSITIVE_LABEL);
		labels.add(BinaryDataLabel.POSITIVE_LABEL);
		labels.add(BinaryDataLabel.POSITIVE_LABEL);
		labels.add(BinaryDataLabel.POSITIVE_LABEL);
		labels.add(BinaryDataLabel.POSITIVE_LABEL);
		labels.add(BinaryDataLabel.POSITIVE_LABEL);
		labels.add(BinaryDataLabel.POSITIVE_LABEL);
		labels.add(BinaryDataLabel.POSITIVE_LABEL);
		labels.add(BinaryDataLabel.POSITIVE_LABEL);
		labels.add(BinaryDataLabel.POSITIVE_LABEL);
		labels.add(BinaryDataLabel.POSITIVE_LABEL);
		labels.add(BinaryDataLabel.POSITIVE_LABEL);
		labels.add(BinaryDataLabel.POSITIVE_LABEL);
		labels.add(BinaryDataLabel.POSITIVE_LABEL);
		labels.add(BinaryDataLabel.POSITIVE_LABEL);
		labels.add(BinaryDataLabel.POSITIVE_LABEL);
		labels.add(BinaryDataLabel.POSITIVE_LABEL);
		labels.add(BinaryDataLabel.POSITIVE_LABEL);
		labels.add(BinaryDataLabel.POSITIVE_LABEL);
		labels.add(BinaryDataLabel.POSITIVE_LABEL);
		labels.add(BinaryDataLabel.POSITIVE_LABEL);
		labels.add(BinaryDataLabel.POSITIVE_LABEL);
		labels.add(BinaryDataLabel.POSITIVE_LABEL);
		labels.add(BinaryDataLabel.POSITIVE_LABEL);
		
		labels.add(BinaryDataLabel.NEGATIVE_LABEL);
		labels.add(BinaryDataLabel.NEGATIVE_LABEL);
		labels.add(BinaryDataLabel.NEGATIVE_LABEL);
		labels.add(BinaryDataLabel.NEGATIVE_LABEL);
		labels.add(BinaryDataLabel.NEGATIVE_LABEL);
		labels.add(BinaryDataLabel.NEGATIVE_LABEL);
		labels.add(BinaryDataLabel.NEGATIVE_LABEL);
		labels.add(BinaryDataLabel.NEGATIVE_LABEL);
		labels.add(BinaryDataLabel.NEGATIVE_LABEL);
		labels.add(BinaryDataLabel.NEGATIVE_LABEL);
		labels.add(BinaryDataLabel.NEGATIVE_LABEL);
		labels.add(BinaryDataLabel.NEGATIVE_LABEL);
		labels.add(BinaryDataLabel.NEGATIVE_LABEL);
		labels.add(BinaryDataLabel.NEGATIVE_LABEL);
		labels.add(BinaryDataLabel.NEGATIVE_LABEL);
		labels.add(BinaryDataLabel.NEGATIVE_LABEL);
		labels.add(BinaryDataLabel.NEGATIVE_LABEL);
		labels.add(BinaryDataLabel.NEGATIVE_LABEL);
		labels.add(BinaryDataLabel.NEGATIVE_LABEL);
		labels.add(BinaryDataLabel.NEGATIVE_LABEL);
		labels.add(BinaryDataLabel.NEGATIVE_LABEL);
		labels.add(BinaryDataLabel.NEGATIVE_LABEL);
		labels.add(BinaryDataLabel.NEGATIVE_LABEL);
		labels.add(BinaryDataLabel.NEGATIVE_LABEL);
		labels.add(BinaryDataLabel.NEGATIVE_LABEL);
		labels.add(BinaryDataLabel.NEGATIVE_LABEL);
		labels.add(BinaryDataLabel.NEGATIVE_LABEL);
		labels.add(BinaryDataLabel.NEGATIVE_LABEL);
		labels.add(BinaryDataLabel.NEGATIVE_LABEL);
		labels.add(BinaryDataLabel.NEGATIVE_LABEL);
		
		SupportVectorMachine svmClassifier = new SupportVectorMachine(SupportVectorMachine.DEFAULT_NUMBER_OF_EPOCHS, SupportVectorMachine.DEFAULT_CROSS_VALIDATION_SPLITS, SupportVectorMachine.DEFAULT_LEARNING_RATES, SupportVectorMachine.DEFAULT_TRADEOFF_VALUES, new IdentityKernel(), true);
		svmClassifier.fit(features, labels);

		System.out.println("Weight vector is " + svmClassifier.getWeightVector() + ".");
		
		List<Double> testFeatures = Arrays.asList(Double.valueOf(12.5), Double.valueOf(11.5), Double.valueOf(102.5), Double.valueOf(-12.5), Double.valueOf(55.5), Double.valueOf(33.5), Double.valueOf(35.6), Double.valueOf(39.5), Double.valueOf(43.5));
		for (Double testFeature : testFeatures) {
			System.out.println("Feature " + testFeature.toString() + " has label " + svmClassifier.getPrediction(Arrays.asList(testFeature)));
		}
		
	}
	
	private void runWith2dFeatures() {
		
		List<List<Double>> features = new ArrayList<List<Double>>();
		features.add(Arrays.asList(Double.valueOf(4.0), Double.valueOf(5.0)));
		features.add(Arrays.asList(Double.valueOf(6.0), Double.valueOf(5.0)));
		features.add(Arrays.asList(Double.valueOf(5.0), Double.valueOf(3.0)));
		features.add(Arrays.asList(Double.valueOf(5.0), Double.valueOf(6.0)));
		features.add(Arrays.asList(Double.valueOf(5.0), Double.valueOf(5.0)));

		features.add(Arrays.asList(Double.valueOf(1.0), Double.valueOf(1.0)));
		features.add(Arrays.asList(Double.valueOf(2.0), Double.valueOf(2.0)));
		features.add(Arrays.asList(Double.valueOf(1.0), Double.valueOf(2.0)));
		features.add(Arrays.asList(Double.valueOf(1.1), Double.valueOf(1.0)));
		features.add(Arrays.asList(Double.valueOf(2.5), Double.valueOf(1.0)));
		
		List<BinaryDataLabel> labels = new ArrayList<BinaryDataLabel>();
		labels.add(BinaryDataLabel.POSITIVE_LABEL);
		labels.add(BinaryDataLabel.POSITIVE_LABEL);
		labels.add(BinaryDataLabel.POSITIVE_LABEL);
		labels.add(BinaryDataLabel.POSITIVE_LABEL);
		labels.add(BinaryDataLabel.POSITIVE_LABEL);
		
		labels.add(BinaryDataLabel.NEGATIVE_LABEL);
		labels.add(BinaryDataLabel.NEGATIVE_LABEL);
		labels.add(BinaryDataLabel.NEGATIVE_LABEL);
		labels.add(BinaryDataLabel.NEGATIVE_LABEL);
		labels.add(BinaryDataLabel.NEGATIVE_LABEL);
		
		SupportVectorMachine svmClassifier = new SupportVectorMachine(SupportVectorMachine.DEFAULT_NUMBER_OF_EPOCHS, SupportVectorMachine.DEFAULT_CROSS_VALIDATION_SPLITS, SupportVectorMachine.DEFAULT_LEARNING_RATES, SupportVectorMachine.DEFAULT_TRADEOFF_VALUES, new IdentityKernel(), false);
		svmClassifier.fit(features, labels);
		
		List<Double> testFeature = Arrays.asList(Double.valueOf(4.5), Double.valueOf(4.5));
		System.out.println("Feature " + testFeature + " has label " + svmClassifier.getPrediction(testFeature));
		
		testFeature = Arrays.asList(Double.valueOf(1.0), Double.valueOf(1.1));
		System.out.println("Feature " + testFeature + " has label " + svmClassifier.getPrediction(testFeature));
	}
}
