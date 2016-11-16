import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;


public class ClassifierClient {

	public static void main(String[] args) {
		
		ClassifierClient classifierClient = new ClassifierClient();
		classifierClient.runWith1dFeatures();
		classifierClient.runWith2dFeatures();
		
	}

	private void runWith1dFeatures() {
		
		
		int numberOfFeaturesPerLabel = 10000, numberOfTestFeaturesPerLabel = 20, maximumFeatureValue = 1000;
		Random randomNumberGenerator = new Random(0);
		double feature = 0.0;
				
		List<List<Double>> features = new ArrayList<List<Double>>();
		List<BinaryDataLabel> labels = new ArrayList<BinaryDataLabel>();
		
		for (int featureCounter = 0; featureCounter < numberOfFeaturesPerLabel; ++featureCounter) {
			
			feature = randomNumberGenerator.nextInt(maximumFeatureValue);
			features.add(Arrays.asList(Double.valueOf(feature)));
			if (feature>= 500.0) {
				labels.add(BinaryDataLabel.NEGATIVE_LABEL);
			} else {
				labels.add(BinaryDataLabel.POSITIVE_LABEL);
			}

		}
		
		SupportVectorMachine svmClassifier = new SupportVectorMachine(SupportVectorMachine.DEFAULT_NUMBER_OF_EPOCHS, SupportVectorMachine.DEFAULT_CROSS_VALIDATION_SPLITS, SupportVectorMachine.DEFAULT_LEARNING_RATES, SupportVectorMachine.DEFAULT_TRADEOFF_VALUES, new IdentityKernel(), false, SupportVectorMachine.LOG_FILE_NAME);
		svmClassifier.fit(features, labels);

		List<List<Double>> testFeatures = new ArrayList<List<Double>>();
		
		for (int featureCounter = 0; featureCounter < numberOfTestFeaturesPerLabel; ++featureCounter) {
			
			feature = randomNumberGenerator.nextInt(maximumFeatureValue);
			testFeatures.add(Arrays.asList(Double.valueOf(feature)));

		}
		
		System.out.println("Weight vector is " + svmClassifier.getWeightVector() + ".");

		for (List<Double> testFeature : testFeatures) {
			System.out.println("Feature " + testFeature.toString() + " has label " + svmClassifier.getPrediction(testFeature));
		}
		
		//svmClassifier.printBestSvmObjectiveTrend();

	}
	
	private void runWith2dFeatures() {
		
		int numberOfFeaturesPerLabel = 500, numberOfTestFeaturesPerLabel = 20,maximumFeatureValue = 1000;
		Random randomNumberGenerator = new Random(0);
		int feature11 = 0, feature12 = 0, feature21 = 0, feature22 = 0;
				
		List<List<Double>> features = new ArrayList<List<Double>>();
		List<BinaryDataLabel> labels = new ArrayList<BinaryDataLabel>();
		
		for (int featureCounter = 0; featureCounter < numberOfFeaturesPerLabel; ++featureCounter) {
			
			feature11 = randomNumberGenerator.nextInt(maximumFeatureValue);
			feature12 = randomNumberGenerator.nextInt(feature11);
			features.add(Arrays.asList(Double.valueOf(feature11), Double.valueOf(feature12)));
			labels.add(BinaryDataLabel.NEGATIVE_LABEL);
			
			feature21 = randomNumberGenerator.nextInt(maximumFeatureValue);
			feature22 = randomNumberGenerator.nextInt(maximumFeatureValue- feature21) + feature21;
			features.add(Arrays.asList(Double.valueOf(feature21), Double.valueOf(feature22)));
			labels.add(BinaryDataLabel.POSITIVE_LABEL);

		}

		SupportVectorMachine svmClassifier = new SupportVectorMachine(SupportVectorMachine.DEFAULT_NUMBER_OF_EPOCHS, SupportVectorMachine.DEFAULT_CROSS_VALIDATION_SPLITS, SupportVectorMachine.DEFAULT_LEARNING_RATES, SupportVectorMachine.DEFAULT_TRADEOFF_VALUES, new IdentityKernel(), false, SupportVectorMachine.LOG_FILE_NAME);
		svmClassifier.fit(features, labels);
		
		List<List<Double>> testFeatures = new ArrayList<List<Double>>();
		
		for (int featureCounter = 0; featureCounter < numberOfTestFeaturesPerLabel; ++featureCounter) {
			
			feature11 = randomNumberGenerator.nextInt(maximumFeatureValue);
			feature12 = randomNumberGenerator.nextInt(feature11);
			testFeatures.add(Arrays.asList(Double.valueOf(feature11), Double.valueOf(feature12)));
			
			feature21 = randomNumberGenerator.nextInt(maximumFeatureValue);
			feature22 = randomNumberGenerator.nextInt(maximumFeatureValue- feature21) + feature21;
			testFeatures.add(Arrays.asList(Double.valueOf(feature21), Double.valueOf(feature22)));

		}		
		
		testFeatures.add(Arrays.asList(Double.valueOf(750), Double.valueOf(751)));

		for (List<Double> testFeature : testFeatures) {
			
			System.out.println("Feature " + testFeature + " has label " + svmClassifier.getPrediction(testFeature));
			
		}
		
		//svmClassifier.printBestSvmObjectiveTrend();

	}

}
