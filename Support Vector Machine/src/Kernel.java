import java.util.ArrayList;
import java.util.List;

/**
 * Interface that defines signature for a Kernel class that provides a method that implements a dot product in a higher
 * dimension feature space given two vectors in the input space
 */
public interface Kernel {
	
	public double getDotProductInFeatureSpace(List<Double> inputSpaceVector1, List<Double> inputSpaceVector2);

	/**
	 * @param vector1
	 * @param vector2
	 * @return the dot product of two vectors
	 */
	public static double getDotProduct(List<Double> vector1, List<Double> vector2) {
		
		//Both vectors need to have the same dimensions
		assert vector1.size() == vector2.size();
		
		//Compute the dot product
		double dotProduct = 0.0;
		int vectorIndex = 0;
		for (Double feature : vector1) {
			dotProduct += feature.doubleValue() * vector2.get(vectorIndex++).doubleValue();
		}
		return dotProduct;
		
	}
	
	/**
	 * @param vector1
	 * @param vector2
	 * @return the difference vector
	 */
	public static List<Double> getDifferenceVector(List<Double> vector1, List<Double> vector2) {
		
		//Both vectors need to have the same dimensions
		assert vector1.size() == vector2.size();
		
		List<Double> differenceVector = new ArrayList<Double>(vector1.size());
		
		//Compute difference vector
		int vectorIndex = 0;
		for (Double feature : vector1) {
			differenceVector.add(feature.doubleValue() - vector2.get(vectorIndex++).doubleValue());
		}
		
		return differenceVector;
		
	}
	
	/**
	 * @param vector
	 * @return square of the vector norm 2
	 */
	public static double getVectorNormSquared(List<Double> vector) {
		
		double vectorNormSquared = 0.0;
		for (Double feature : vector) {
			vectorNormSquared += Math.pow(feature, 2.0);
		}
		return vectorNormSquared;
		
	}
}
