import java.util.List;

/**
 * The identity kernel returns an inner product of the input vectors
 *
 */
public class IdentityKernel implements Kernel {

	/* (non-Javadoc)
	 * @see Kernel#getDotProductInFeatureSpace(java.util.List, java.util.List)
	 */
	@Override
	public double getDotProductInFeatureSpace(List<Double> inputSpaceVector1, List<Double> inputSpaceVector2) {
		return Kernel.getDotProduct(inputSpaceVector1, inputSpaceVector2);
	}

}
