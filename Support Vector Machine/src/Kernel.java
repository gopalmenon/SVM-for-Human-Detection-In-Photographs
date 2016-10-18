import java.util.List;

/**
 * Interface that defines signature for a Kernel class that provides a method that implements a dot product in a higher
 * dimension feature space given two vectors in the input space
 */
public interface Kernel {
	
	public double getDotProductInFeatureSpace(List<Double> inputSpaceVector1, List<Double> inputSpaceVector2);

}
