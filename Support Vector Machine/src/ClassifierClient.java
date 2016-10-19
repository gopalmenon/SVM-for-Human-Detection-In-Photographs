import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


public class ClassifierClient {

	public static void main(String[] args) {
		
		ClassifierClient classifierClient = new ClassifierClient();
		classifierClient.runWith1dFeatures();
		classifierClient.runWith2dFeatures();
		
	}

	private void runWith1dFeatures() {
		
		List<List<Double>> features = new ArrayList<List<Double>>();
		features.add(Arrays.asList(Double.valueOf(5.0)));
		features.add(Arrays.asList(Double.valueOf(6.0)));
		features.add(Arrays.asList(Double.valueOf(9.0)));
		features.add(Arrays.asList(Double.valueOf(10.0)));
		features.add(Arrays.asList(Double.valueOf(12.0)));
		
		features.add(Arrays.asList(Double.valueOf(1.0)));
		features.add(Arrays.asList(Double.valueOf(1.2)));
		features.add(Arrays.asList(Double.valueOf(3.3)));
		features.add(Arrays.asList(Double.valueOf(3.7)));
		features.add(Arrays.asList(Double.valueOf(4.1)));
		
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
		
		SupportVectorMachine svmClassifier = new SupportVectorMachine(SupportVectorMachine.DEFAULT_NUMBER_OF_EPOCHS, SupportVectorMachine.DEFAULT_CROSS_VALIDATION_SPLITS, SupportVectorMachine.DEFAULT_LEARNING_RATES, SupportVectorMachine.DEFAULT_TRADEOFF_VALUES, new IdentityKernel());
		svmClassifier.fit(features, labels);
		
		List<Double> testFeature = Arrays.asList(Double.valueOf(12.5));
		System.out.println("Feature " + testFeature + " has label " + svmClassifier.getPrediction(testFeature));
		
		testFeature = Arrays.asList(Double.valueOf(0.11));
		System.out.println("Feature " + testFeature + " has label " + svmClassifier.getPrediction(testFeature));

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
		
		SupportVectorMachine svmClassifier = new SupportVectorMachine(SupportVectorMachine.DEFAULT_NUMBER_OF_EPOCHS, SupportVectorMachine.DEFAULT_CROSS_VALIDATION_SPLITS, SupportVectorMachine.DEFAULT_LEARNING_RATES, SupportVectorMachine.DEFAULT_TRADEOFF_VALUES, new IdentityKernel());
		svmClassifier.fit(features, labels);
		
		List<Double> testFeature = Arrays.asList(Double.valueOf(4.5), Double.valueOf(4.5));
		System.out.println("Feature " + testFeature + " has label " + svmClassifier.getPrediction(testFeature));
		
		testFeature = Arrays.asList(Double.valueOf(1.0), Double.valueOf(1.1));
		System.out.println("Feature " + testFeature + " has label " + svmClassifier.getPrediction(testFeature));
	}
}
