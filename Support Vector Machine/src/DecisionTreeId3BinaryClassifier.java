import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Random;
import java.util.Set;

/**
 * Decision Tree classifier that uses the ID3 algorithm
 */
public class DecisionTreeId3BinaryClassifier implements Classifier {
	
	private DecisionTreeNode decisionTreeRootNode;
	private Random randomNumberGenerator;
	private int maximumTreeDepth;
	private int limitingTreeDepth;
	private boolean continuousFeatures;
	private int numberOfRandomFeaturesToChooseFrom;

	public DecisionTreeId3BinaryClassifier(int limitingTreeDepth, boolean continuousFeatures) {
		this.limitingTreeDepth = limitingTreeDepth;
		this.continuousFeatures = continuousFeatures;
		this.randomNumberGenerator = new Random(0);
		this.maximumTreeDepth = 0;
	}
	
	/* (non-Javadoc)
	 * @see Classifier#train(java.util.List)
	 */
	@Override
	public void train(List<List<Double>> trainingData, List<BinaryDataLabel> trainingDataLabels) {
		
		//Training data should be present and duplicates in training data are not allowed
		assert trainingData.size() > 0 && trainingDataLabels.size()> 0 && !isDuplicatesInTrainingData(trainingData);

		this.numberOfRandomFeaturesToChooseFrom = (int) (Math.log(trainingDataLabels.size()) / Math.log(2));
		
		this.decisionTreeRootNode = buildDecisionTree(trainingData, trainingDataLabels, getAttributesVector(trainingData.get(0).size()), 0.0, this.maximumTreeDepth);

	}

	/* (non-Javadoc)
	 * @see Classifier#predict(java.util.List)
	 */
	@Override
	public List<BinaryDataLabel> predict(List<List<Double>> testData) {

		List<BinaryDataLabel> prediction = new ArrayList<BinaryDataLabel>();
		for (List<Double> testDataRecord : testData) {
			prediction.add(getPrediction(testDataRecord));
		}
		
		return prediction;
	}
	
	/**
	 * @param testDataRecord
	 * @return the prediction based on the decision tree constructed during the training phase
	 */
	private BinaryDataLabel getPrediction(List<Double> testDataRecord) {
		
		DecisionTreeNode decisionTreeNode = this.decisionTreeRootNode;
		int attributeSplitOn = 0;
		double attribute = 0;
		outerLoop:
		while (decisionTreeNode instanceof DecisionTreeInternalNode) {
			
			attributeSplitOn = ((DecisionTreeInternalNode) decisionTreeNode).getAttributeSplitOn();
			attribute = testDataRecord.get(attributeSplitOn);
			
			for (DecisionTreeNode decisionTreeChildNode : ((DecisionTreeInternalNode) decisionTreeNode).getChildNodes()) {
				if (decisionTreeChildNode instanceof DecisionTreeInternalNode) {
					if (decisionTreeChildNode.getPreviousAttributeValue() == attribute) {
						decisionTreeNode = decisionTreeChildNode;
						break;
					}
				} else {
					if (decisionTreeChildNode.getPreviousAttributeValue() == attribute) {
						decisionTreeNode = decisionTreeChildNode;
						break outerLoop;
					}
				}
			}
			
			
		}
		
		return ((DecisionTreeLeafNode) decisionTreeNode).getLabel();
		
	}
	
	/**
	 * @param trainingData
	 * @return true of training data has duplicate rows
	 */
	private boolean isDuplicatesInTrainingData(List<List<Double>> trainingData) {
		
		Set<List<Double>> trainingDataSet = new HashSet<List<Double>>(trainingData);
		if (trainingDataSet.size() < trainingData.size()) {
			return true;
		} else {
			return false;
		}
		
	}
	
	/**
	 * @param trainingDataLabels
	 * @return entropy of the collection of training data records
	 */
	private double getCollectionEntropy(List<BinaryDataLabel> trainingDataLabels) {
		
		Iterator<BinaryDataLabel> trainingDataLabelsIterator = trainingDataLabels.iterator();
		BinaryDataLabel trainingDataLabel = null;
		int positiveLabelCount = 0, negativeLabelCount = 0;
		
		while (trainingDataLabelsIterator.hasNext()) {
			
			trainingDataLabel = trainingDataLabelsIterator.next();
			
			if (trainingDataLabel == BinaryDataLabel.POSITIVE_LABEL) {
				++positiveLabelCount;
			} else {
				++negativeLabelCount;	
			}
			
		}

		double totalCount = positiveLabelCount + negativeLabelCount;
		double positiveLabelFraction = positiveLabelCount/totalCount, negativeLabelFraction = negativeLabelCount/totalCount;
		return -1 * positiveLabelFraction * logBase2(positiveLabelFraction) - negativeLabelFraction * logBase2(negativeLabelFraction);

	}
	
	

	/**
	 * @param trainingData
	 * @param trainingDataLabels
	 * @param featureToPartitionOn
	 * @return the information gain or expected reduction in entropy by partitioning on an attribute 
	 */
	private double getInformationGain(List<List<Double>> trainingData, List<BinaryDataLabel> trainingDataLabels, int featureToPartitionOn) {
		
		//Feature to partition on must be one of the columns
		assert featureToPartitionOn <= trainingData.get(0).size();
		
		if (this.continuousFeatures) {
			getInformationGainForContinuousFeatures(trainingData, trainingDataLabels, featureToPartitionOn);		
		} else {
			getInformationGainForBinaryFeatures(trainingData, trainingDataLabels, featureToPartitionOn);		
		}
		
		Map<Character, FeatureValueCounts> counts = new HashMap<Character, FeatureValueCounts>();
		Iterator<List<Character>> trainingDataIterator = trainingData.iterator();
		
		List<Character> trainingDataRecord = null;
		char trainingDataRecordLabel = ' ';
		Character featureValue = null;
		FeatureValueCounts existingFeatureValueCounts = null;
				
		while (trainingDataIterator.hasNext()) {
			
			trainingDataRecord = trainingDataIterator.next();
			featureValue = trainingDataRecord.get(featureToPartitionOn);
			trainingDataRecordLabel = trainingDataRecord.get(this.labelOffset).charValue();
			
			if (counts.containsKey(featureValue)) {
				existingFeatureValueCounts = counts.get(featureValue);
				if (trainingDataRecordLabel == this.firstLabel) {
					existingFeatureValueCounts.incrementFirstLabelCount();
				} else {
					existingFeatureValueCounts.incrementSecondLabelCount();
				}
				counts.put(featureValue, existingFeatureValueCounts);
			} else {
				if (trainingDataRecordLabel == this.firstLabel) {
					counts.put(featureValue, this.new FeatureValueCounts(1, 0));
				} else {
					counts.put(featureValue, this.new FeatureValueCounts(0, 1));
				}
			}
			
		}
		
		return getCollectionEntropy(trainingDataLabels) - getWeightedFeatureValuesEntropy(counts);
		
	}
	
	/**
	 * @param trainingData
	 * @param trainingDataLabels
	 * @param featureToPartitionOn
	 * @return the information gain or expected reduction in entropy by partitioning on an attribute
	 */
	private EntropyGainAndFeatureValue getInformationGainForContinuousFeatures(List<List<Double>> trainingData, List<BinaryDataLabel> trainingDataLabels, int featureToPartitionOn) {
		
		//Combine the data and labels into a single list so that they can be sorted together 
		List<DataAndLabel> combinedDataAndLabels = DataAndLabel.getCombinDataAndLabels(trainingData, trainingDataLabels);
		
		//Create a comparator that can be used for sorting the combined data and labels list
		DataAndLabelComparator dataAndLabelComparator = new DataAndLabelComparator(featureToPartitionOn);
		
		//Sort the combined list on the feature
		Collections.sort(combinedDataAndLabels, dataAndLabelComparator);
		
		//Run through the sorted feature and find potential splitting points
		boolean firstTime = true;
		double previousAttributeValue = 0.0;
		BinaryDataLabel previousLabel = null;
		Set<Double> potentialAttributesToSplitOn = new HashSet<Double>();
		
		for (DataAndLabel dataAndLabel : combinedDataAndLabels) {
			
			if (firstTime) {
				firstTime = false;
			} else {
				if (dataAndLabel.getLabel() != previousLabel) {
					potentialAttributesToSplitOn.add(previousAttributeValue);
				}
			}
			previousAttributeValue = dataAndLabel.getData().get(featureToPartitionOn);
			previousLabel = dataAndLabel.getLabel();
		}
		
		//Find the point where splitting will result in maximum entropy gain
		double maximumEntropyGain = Double.MIN_VALUE, splittingPointForMaximumEntropyGain = 0.0, currentEntropyGain = 0.0;
		for (Double splittingPoint : potentialAttributesToSplitOn) {
		
			currentEntropyGain = getCollectionEntropy(trainingDataLabels) - getEntropyOnSplit(combinedDataAndLabels, splittingPoint, featureToPartitionOn);
			if (currentEntropyGain > maximumEntropyGain) {
				
				maximumEntropyGain = currentEntropyGain;
				splittingPointForMaximumEntropyGain = splittingPoint;
				
			}

		}
		
		return this.new EntropyGainAndFeatureValue(maximumEntropyGain, splittingPointForMaximumEntropyGain);

	}
	
	/**
	 * Class to store entropy gain and feature value that resulted in the gain
	 *
	 */
	private class EntropyGainAndFeatureValue {
		private double entropyGain;
		private double featureValue;
		public EntropyGainAndFeatureValue(double entropyGain, double featureValue) {
			this.entropyGain = entropyGain;
			this.featureValue = featureValue;
		}
		public double getEntropyGain() {
			return entropyGain;
		}
		public double getFeatureValue() {
			return featureValue;
		}
	}
	
	private double getInformationGainForBinaryFeatures(List<List<Double>> trainingData, List<BinaryDataLabel> trainingDataLabels, int featureToPartitionOn) {
		return 0.0;
	}
	
	/**
	 * @param combinedDataAndLabels
	 * @param splittingPoint
	 * @return entropy on splitting 
	 */
	private double getEntropyOnSplit(List<DataAndLabel> combinedDataAndLabels, double splittingPoint, int featureToPartitionOn) {
		
		int belowSplitPositivesCount = 0, belowSplitNegativesCount = 0, aboveSplitPositivesCount = 0, aboveSplitNegativesCount = 0;
		
		for (DataAndLabel dataAndLabel : combinedDataAndLabels) {
		
			if (dataAndLabel.getData().get(featureToPartitionOn) < splittingPoint) {
				if (dataAndLabel.getLabel() == BinaryDataLabel.POSITIVE_LABEL) {
					++belowSplitPositivesCount;
				} else {
					++belowSplitNegativesCount;
				}
			} else {
				if (dataAndLabel.getLabel() == BinaryDataLabel.POSITIVE_LABEL) {
					++aboveSplitPositivesCount;
				} else {
					++aboveSplitNegativesCount;
				}
			}
			
		}
		
		int totalCount =  belowSplitPositivesCount + belowSplitNegativesCount + aboveSplitPositivesCount + aboveSplitNegativesCount;
		
		return (double) ((belowSplitPositivesCount + belowSplitNegativesCount) / totalCount) * getWeightedSubsetEntropy(belowSplitPositivesCount, belowSplitNegativesCount) +
			   (double) ((aboveSplitPositivesCount + aboveSplitNegativesCount) / totalCount) * getWeightedSubsetEntropy(aboveSplitPositivesCount, aboveSplitNegativesCount);
	}

	
	/**
	 * @param firstLabelCount
	 * @param secondLabelCount
	 * @return the weighted entropy of a specific value of an attribute
	 */
	private double getWeightedSubsetEntropy(int positiveLabelCount, int negativeLabelCount) {
		
		double firstLabelFraction = (double) positiveLabelCount / (positiveLabelCount + negativeLabelCount);
		double secondLabelFraction = (double) negativeLabelCount / (positiveLabelCount + negativeLabelCount);
		
		return -1 * firstLabelFraction * logBase2(firstLabelFraction) - secondLabelFraction * logBase2(secondLabelFraction);
		
	}
	
	/**
	 * Store label counts for each value of an attribute
	 *
	 */
	private class FeatureValueCounts {
		private int firstLabelCount;
		private int secondLabelCount;
		public FeatureValueCounts(int firstLabelCount, int secondLabelCount) {
			this.firstLabelCount = firstLabelCount;
			this.secondLabelCount = secondLabelCount;
		}
		public int getFirstLabelCount() {
			return firstLabelCount;
		}
		public void incrementFirstLabelCount() {
			++this.firstLabelCount;
		}
		public void incrementSecondLabelCount() {
			++this.secondLabelCount;
		}
		public int getSecondLabelCount() {
			return secondLabelCount;
		}
	}

	/**
	 * Build a decision tree to classify the training data based on the ID3 algorithm 
	 * @param trainingData
	 * @param labels
	 * @param attributesVector
	 * @return a root node for the tree (subtree for the recursive case)
	 */
	private DecisionTreeNode buildDecisionTree(List<List<Double>> examples, List<BinaryDataLabel> labels, Set<Integer> attributesVector, double previousAttributeValue, int currentDepth) {
		
		//Keep track of maximum tree depth
		if (currentDepth > this.maximumTreeDepth) {
			this.maximumTreeDepth = currentDepth;
		}
		
		//Check if limiting tree depth has been reached. If so return a leaf node with most common label
		if (currentDepth == this.limitingTreeDepth) {
			return new DecisionTreeLeafNode(previousAttributeValue, getMostCommonTargetAttribute(labels));
		}
		
		//If all examples have the same label, return a leaf node marked with the common label 
		AllExamples allExamples = isAllExamplesTheSame(labels);
		if (allExamples.isAllExamplesSame()) {
			return new DecisionTreeLeafNode(previousAttributeValue, allExamples.getAllExamplesLabel());
		}
		
		//If there are no attributes left, return a leaf node with most common target label
		if (attributesVector.size() == 0) {
			return new DecisionTreeLeafNode(previousAttributeValue, getMostCommonTargetAttribute(labels));
		}
		
		//Choose best attribute to split on from a subset of all the attributes left
		int numberOfAttributesToUse = Math.min(attributesVector.size(), this.randomNumberGenerator.nextInt(this.numberOfRandomFeaturesToChooseFrom) + 1);
		Set<Integer> attributesSubset = getAttributesSubset(attributesVector, numberOfAttributesToUse);
		
		int bestAttribute = getBestClassifyingAttribute(examples, labels, attributesSubset);
		
		//Create child nodes for each possible value of the attribute to split on
		List<DecisionTreeNode> childNodes = new ArrayList<DecisionTreeNode>();
		Set<Character> allPossibleAttributeValues = getAllPossibleAttributeValues(bestAttribute);
		
		//For each possible attribute create a new sub tree
		for (Character attributeValue : allPossibleAttributeValues) {
		
			List<List<Character>> trainingDataSubset = getTrainingDataSubset(examples, bestAttribute, attributeValue.charValue());
			if (trainingDataSubset.size() == 0) {
				childNodes.add(new DecisionTreeLeafNode(attributeValue.charValue(), getMostCommonTargetAttribute(examples)));
			} else {
				Set<Integer> reducedAttributesVector = new HashSet<Integer>(attributesVector);
				reducedAttributesVector.remove(Integer.valueOf(bestAttribute));
				childNodes.add(buildDecisionTree(trainingDataSubset, reducedAttributesVector, attributeValue, currentDepth + 1));
			}
			
		}
		
		return new DecisionTreeInternalNode(previousAttributeValue, bestAttribute, childNodes);
		
	}
	
	/**
	 * @param attributesVector
	 * @param numberOfAttributesToUse
	 * @return a subset of the attributes remaining
	 */
	private Set<Integer> getAttributesSubset(Set<Integer> attributesVector, int numberOfAttributesToUse) {
		
		if (numberOfAttributesToUse == attributesVector.size()) {
			return attributesVector;
		}
		
		Set<Integer> attributesSubset = new HashSet<Integer>();
		int attributeToUse = 0;
		
		while (attributesVector.size() < numberOfAttributesToUse) {
			
			attributeToUse = this.randomNumberGenerator.nextInt(attributesVector.size());
			if (!attributesSubset.contains(Integer.valueOf(attributeToUse))) {
				attributesSubset.add(Integer.valueOf(attributeToUse));
			}
		
		}
		
		return attributesSubset;
		
	}
	
	/**
	 * @param examples
	 * @param attributesVector
	 * @param labels
	 * @return the attribute that best classifies the examples based on information gain computed from the 
	 * selection of the attribute that maximizes the reduction in entropy
	 */
	private int getBestClassifyingAttribute(List<List<Character>> examples, List<BinaryDataLabel> labels, Set<Integer> attributesVector) {
		
		int bestAttribute = Integer.MIN_VALUE;
		double bestInformationGainSoFar = Double.MIN_VALUE, informationGain = 0.0;
		for (Integer attribute : attributesVector) {
			
			informationGain = getInformationGain(examples, labels, attribute.intValue());
			if (informationGain > bestInformationGainSoFar) {
				bestInformationGainSoFar = informationGain;
				bestAttribute = attribute.intValue();
			}

		}
		
		return bestAttribute;
	}
	
	/**
	 * @param attribute
	 * @return a set of all possible values for the attribute
	 */
	private Set<Character> getAllPossibleAttributeValues(int attribute) {
		
		Set<Character> allPossibleAttributeValues = new HashSet<Character>();
		Properties featureProperties = new Properties();
		
		try {
		
			//Load attribute values from properties file
			InputStream inputStream = new FileInputStream(this.propertiesFileName);
			featureProperties.load(inputStream);
			
			String featureName = featureProperties.getProperty(Integer.valueOf(attribute).toString());
			String commaSeparatedAttributeValues = featureProperties.getProperty(featureName);
			
			String[] attributeValuesArray = commaSeparatedAttributeValues.split(ATTRIBUTE_VALUE_SEPARATOR);
			
			//Load attribute values into set to be returned
			for (String attributeValue : attributeValuesArray) {
				
				if (attributeValue.trim().length() > 0) {
					allPossibleAttributeValues.add(Character.valueOf(attributeValue.trim().charAt(0)));
				}
				
			}
		} catch (IOException e) {
			e.printStackTrace();
			System.exit(0);
		}
	
		return allPossibleAttributeValues;
	}
	
	/**
	 * @param trainingData
	 * @param featureToPartitionOn
	 * @param featureValue
	 * @return a subset of the training data containing only those records with matching feature values
	 */
	private List<List<Character>> getTrainingDataSubset(List<List<Character>> trainingData, int featureToPartitionOn, char featureValue) {
				
		List<List<Character>> trainingDataSubset = new ArrayList<List<Character>>();
		for (List<Character> trainingDataRecord : trainingData) {
			if (trainingDataRecord.get(featureToPartitionOn).charValue() == featureValue) {
				trainingDataSubset.add(new ArrayList<Character>(trainingDataRecord));
			}
		}
		return trainingDataSubset;
		
	}
	
	/**
	 * @param numberOfAttributes
	 * @return an attribute vector filled with attribute column numbers from zero to one less than number of attributes
	 */
	private Set<Integer> getAttributesVector(int numberOfAttributes) {
		
		Set<Integer> attributesVector = new HashSet<Integer>();
		for (int attributeCounter = 0; attributeCounter < numberOfAttributes; ++attributeCounter) {
			attributesVector.add(Integer.valueOf(attributeCounter));
		}
		return attributesVector;
		
	}
	
	/**
	 * @param label
	 * @return class containing boolean value saying whether all examples have the same label 
	 * and a character value giving the value of the common attribute
	 */
	private AllExamples isAllExamplesTheSame(List<BinaryDataLabel> labels) {
		
		boolean firstTime = true;
		BinaryDataLabel allExamples = null;
		for (BinaryDataLabel label : labels) {
			if (firstTime) {
				firstTime = false;
				allExamples = label;
			} else {
				if (label != allExamples) {
					return this.new AllExamples(false, allExamples);
				}
			}
		}
		return this.new AllExamples(true, allExamples); 
	}
	
	/**
	 * Class returned from check to see if all examples in training set have the same attribute

	 */
	private class AllExamples {
		private boolean isAllExamplesSame;
		private BinaryDataLabel allExamplesLabel;
		public AllExamples(boolean isAllExamplesSame, BinaryDataLabel allExamplesLabel) {
			this.isAllExamplesSame = isAllExamplesSame;
			this.allExamplesLabel = allExamplesLabel;
		}
		public boolean isAllExamplesSame() {
			return isAllExamplesSame;
		}
		public BinaryDataLabel getAllExamplesLabel() {
			return allExamplesLabel;
		}
	}
	
	/**
	 * @param labels
	 * @return the most common target label in the training set. If there is a tie, choose a random target value;
	 */
	private BinaryDataLabel getMostCommonTargetAttribute(List<BinaryDataLabel> labels) {
		
		//Count the number of occurrences for the labels
		int positiveLabelCount = 0, negativeLabelCount = 0;
		for (BinaryDataLabel label : labels) {
			if (label == BinaryDataLabel.POSITIVE_LABEL) {
				++positiveLabelCount;
			} else {
				++negativeLabelCount;
			}
		}
		
		//Return the label corresponding to the most frequent value
		if (positiveLabelCount > negativeLabelCount) {
			return BinaryDataLabel.POSITIVE_LABEL;
		} else if (negativeLabelCount > positiveLabelCount) {
			return BinaryDataLabel.NEGATIVE_LABEL;
		} else {
			//Break the tie randomly
			if (this.randomNumberGenerator.nextBoolean()) {
				return BinaryDataLabel.POSITIVE_LABEL;
			} else {
				return BinaryDataLabel.NEGATIVE_LABEL;
			}
		}
		
	}
	
	/**
	 * @param number
	 * @return log base 2. Log 0 will be defined as zero.
	 */
	private double logBase2(double number) {
		if (number == 0.0) {
			return 0.0;
		} else {
			return Math.log(number) / Math.log(2);
		}
		
	}
	
	/**
	 * @return the maximum tree depth reached while building the decision tree
	 */
	public int getMaximumTreeDepth() {
		return maximumTreeDepth;
	}
	
}
