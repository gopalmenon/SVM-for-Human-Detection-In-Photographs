import java.util.List;

/**
 * The polynomial kernel returns inner product of the input vectors in a feature space that is a polynomial transform
 * 
 * phi(x1).phi(x2) = 1 + Sigma(x1i.x2i) + Sigma(x1i.x1j.x2i.x2j)
 *
 */
public class PolynomialKernel extends KernelImplementation {

	/* (non-Javadoc)
	 * @see Kernel#getDotProductInFeatureSpace(java.util.List, java.util.List)
	 */
	@Override
	public double getDotProductInFeatureSpace(List<Double> inputSpaceVector1, List<Double> inputSpaceVector2) {
		
		double innerProduct = getDotProduct(inputSpaceVector1, inputSpaceVector2);
		return 1.0 + innerProduct + Math.pow(innerProduct, 2.0);
		
	}

}
