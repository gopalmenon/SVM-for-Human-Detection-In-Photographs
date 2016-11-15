
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
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
		assert trainingData.size() > 0 && trainingDataLabels.size() > 0 && !isDuplicatesInTrainingData(trainingData);

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
		double attribute = 0.0;
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
		double positiveLabelFraction = (double) positiveLabelCount/totalCount, negativeLabelFraction = (double) negativeLabelCount/totalCount;
		return -1 * positiveLabelFraction * logBase2(positiveLabelFraction) - negativeLabelFraction * logBase2(negativeLabelFraction);

	}
	
	

	/**
	 * @param trainingData
	 * @param trainingDataLabels
	 * @param featureToPartitionOn
	 * @return the information gain or expected reduction in entropy by partitioning on an attribute 
	 */
	private InformationGainAndFeatureValue getInformationGain(List<List<Double>> trainingData, List<BinaryDataLabel> trainingDataLabels, int featureToPartitionOn) {
		
		//Feature to partition on must be one of the columns
		assert featureToPartitionOn <= trainingData.get(0).size();
		
		if (this.continuousFeatures) {
			return getInformationGainForContinuousFeatures(trainingData, trainingDataLabels, featureToPartitionOn);		
		} else {
			return getInformationGainForBinaryFeatures(trainingData, trainingDataLabels, featureToPartitionOn);		
		}
		
	}
	
	/**
	 * @param trainingData
	 * @param trainingDataLabels
	 * @param featureToPartitionOn
	 * @return the information gain or expected reduction in entropy by partitioning on an attribute
	 */
	private InformationGainAndFeatureValue getInformationGainForContinuousFeatures(List<List<Double>> trainingData, List<BinaryDataLabel> trainingDataLabels, int featureToPartitionOn) {
		
		//Combine the data and labels into a single list so that they can be sorted together 
		List<DataAndLabel> combinedDataAndLabels = DataAndLabel.getCombinDataAndLabels(trainingData, trainingDataLabels);
		
		//Create a comparator that can be used for sorting the combined data and labels list
		DataAndLabelComparator dataAndLabelComparator = new DataAndLabelComparator(featureToPartitionOn);
		
		//Sort the combined list on the feature
		Collections.sort(combinedDataAndLabels, dataAndLabelComparator);
		
		//Run through the sorted feature and find potential splitting points
		boolean firstTime = true;
		BinaryDataLabel previousLabel = null;
		Set<Double> potentialAttributesToSplitOn = new HashSet<Double>();
		
		for (DataAndLabel dataAndLabel : combinedDataAndLabels) {
			
			if (firstTime) {
				firstTime = false;
			} else {
				if (dataAndLabel.getLabel() != previousLabel) {
					potentialAttributesToSplitOn.add(dataAndLabel.getData().get(featureToPartitionOn));
				}
			}
			previousLabel = dataAndLabel.getLabel();
		}
		
		//Find the point where splitting will result in maximum entropy gain
		double maximumInformationGain = Double.MIN_VALUE, splittingPointForMaximumEntropyGain = 0.0, currentInformationGain = 0.0;
		for (Double splittingPoint : potentialAttributesToSplitOn) {
		
			currentInformationGain = getCollectionEntropy(trainingDataLabels) - getEntropyOnSplit(combinedDataAndLabels, splittingPoint, featureToPartitionOn);
			if (currentInformationGain > maximumInformationGain) {
				
				maximumInformationGain = currentInformationGain;
				splittingPointForMaximumEntropyGain = splittingPoint;
				
			}

		}
		
		return this.new InformationGainAndFeatureValue(maximumInformationGain, splittingPointForMaximumEntropyGain);

	}
	
	/**
	 * Class to store entropy gain and feature value that resulted in the gain
	 *
	 */
	private class InformationGainAndFeatureValue {
		private double informationGain;
		private double featureValue;
		public InformationGainAndFeatureValue(double informationGain, double featureValue) {
			this.informationGain = informationGain;
			this.featureValue = featureValue;
		}
		public double getInformationGain() {
			return informationGain;
		}
		public double getFeatureValue() {
			return featureValue;
		}
	}
	
	/**
	 * @param trainingData
	 * @param trainingDataLabels
	 * @param featureToPartitionOn
	 * @return the information gain or expected reduction in entropy by partitioning on an attribute
	 */
	private InformationGainAndFeatureValue getInformationGainForBinaryFeatures(List<List<Double>> trainingData, List<BinaryDataLabel> trainingDataLabels, int featureToPartitionOn) {
		
		return this.new InformationGainAndFeatureValue(getCollectionEntropy(trainingDataLabels) - getEntropyOnSplit(DataAndLabel.getCombinDataAndLabels(trainingData, trainingDataLabels), 1.0, featureToPartitionOn), 1.0);
		
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
		int numberOfAttributesToUse = Math.min(attributesVector.size(), this.numberOfRandomFeaturesToChooseFrom);
		Set<Integer> attributesSubset = getAttributesSubset(attributesVector, numberOfAttributesToUse);
		
		BestAttributeAndFeatureValue bestAttributeAndFeatureValue = getBestClassifyingAttribute(examples, labels, attributesSubset);
		
		//Create child nodes for cases on both sides of the split point
		List<DecisionTreeNode> childNodes = new ArrayList<DecisionTreeNode>();
		
		//Create a new sub tree for both sides of the split
		for (boolean lessThanSplitValue : Arrays.asList(true, false)){
			
			List<DataAndLabel> trainingDataSubset = getTrainingDataSubset(examples, labels, bestAttributeAndFeatureValue.getBestAttribute(), bestAttributeAndFeatureValue.getFeatureValue(), lessThanSplitValue);
			if (trainingDataSubset.size() == 0) {
				childNodes.add(new DecisionTreeLeafNode(bestAttributeAndFeatureValue.getFeatureValue(), getMostCommonTargetAttribute(labels)));
			} else {
				Set<Integer> reducedAttributesVector = new HashSet<Integer>(attributesVector);
				reducedAttributesVector.remove(Integer.valueOf(bestAttributeAndFeatureValue.getBestAttribute()));
				childNodes.add(buildDecisionTree(DataAndLabel.getData(trainingDataSubset), DataAndLabel.getLabels(trainingDataSubset), reducedAttributesVector, bestAttributeAndFeatureValue.getFeatureValue(), currentDepth + 1));
			}
			
		}
		
		return new DecisionTreeInternalNode(previousAttributeValue, bestAttributeAndFeatureValue.bestAttribute, childNodes);
		
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
	private BestAttributeAndFeatureValue getBestClassifyingAttribute(List<List<Double>> examples, List<BinaryDataLabel> labels, Set<Integer> attributesVector) {
		
		int bestAttribute = Integer.MIN_VALUE;
		double bestInformationGainSoFar = Double.MIN_VALUE;
		InformationGainAndFeatureValue informationGainAndFeatureValue = null;
		
		for (Integer attribute : attributesVector) {
			
			informationGainAndFeatureValue = getInformationGain(examples, labels, attribute.intValue());
			if (informationGainAndFeatureValue.getInformationGain() > bestInformationGainSoFar) {
				bestInformationGainSoFar = informationGainAndFeatureValue.getInformationGain();
				bestAttribute = attribute.intValue();
			}

		}
		
		return this.new BestAttributeAndFeatureValue(bestAttribute, informationGainAndFeatureValue.getFeatureValue());
	}
	
	/**
	 * Class to store best attribute and feature value 
	 *
	 */
	private class BestAttributeAndFeatureValue {
		private int bestAttribute;
		private double featureValue;
		public BestAttributeAndFeatureValue(int bestAttribute, double featureValue) {
			this.bestAttribute = bestAttribute;
			this.featureValue = featureValue;
		}
		public int getBestAttribute() {
			return bestAttribute;
		}
		public double getFeatureValue() {
			return featureValue;
		}
	}	
	
	/**
	 * @param trainingData
	 * @param labels
	 * @param featureToPartitionOn
	 * @param splitValue
	 * @param lessThanSplitValue
	 * @return a subset of the training data containing only those records with matching feature values
	 */
	private List<DataAndLabel> getTrainingDataSubset(List<List<Double>> trainingData, List<BinaryDataLabel> labels, int featureToPartitionOn, double splitValue, boolean lessThanSplitValue) {
				
		List<DataAndLabel> trainingDataSubset = new ArrayList<DataAndLabel>();
		List<DataAndLabel> combinDataAndLabels = DataAndLabel.getCombinDataAndLabels(trainingData, labels);

		for (DataAndLabel dataAndLabel : combinDataAndLabels) {
			if (lessThanSplitValue) {
				if (dataAndLabel.getData().get(featureToPartitionOn) < splitValue) {
					trainingDataSubset.add(dataAndLabel);
				}
			} else {
				if (dataAndLabel.getData().get(featureToPartitionOn) >= splitValue) {
					trainingDataSubset.add(dataAndLabel);
				}
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
