import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


public class ClassifierClient {

	public static void main(String[] args) {
		
		ClassifierClient classifierClient = new ClassifierClient();
		classifierClient.runWith2dFeatures();
		
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
		features.add(Arrays.asList(Double.valueOf(8.0), Double.valueOf(1.0)));
		features.add(Arrays.asList(Double.valueOf(8.0), Double.valueOf(8.0)));
		
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
		
		SupportVectorMachine svmClassifier = new SupportVectorMachine(SupportVectorMachine.DEFAULT_NUMBER_OF_EPOCHS, SupportVectorMachine.DEFAULT_CROSS_VALIDATION_SPLITS, SupportVectorMachine.DEFAULT_LEARNING_RATES, SupportVectorMachine.DEFAULT_TRADEOFF_VALUES, new GaussianRadialBasisFunctionKernel(1.1));
		svmClassifier.fit(features, labels);
		
		List<Double> testFeature = Arrays.asList(Double.valueOf(4.5), Double.valueOf(4.5));
		System.out.println("Feature " + testFeature + " has label " + svmClassifier.getPrediction(testFeature));
		
		testFeature = Arrays.asList(Double.valueOf(1.0), Double.valueOf(3.0));
		System.out.println("Feature " + testFeature + " has label " + svmClassifier.getPrediction(testFeature));
	}
}
