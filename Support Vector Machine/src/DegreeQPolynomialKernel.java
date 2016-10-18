import java.util.List;

/**
 * The degree-Q polynomial kernel returns (zeta + gamma * x1.x2)^Q
 */
public class DegreeQPolynomialKernel extends KernelImplementation {

	private double zeta, gamma, degree;
	
	public DegreeQPolynomialKernel(double zeta, double gamma, int degree) {
		this.zeta = zeta;
		this.gamma = gamma;
		this.degree = degree;
	}
	
	/* (non-Javadoc)
	 * @see Kernel#getDotProductInFeatureSpace(java.util.List, java.util.List)
	 */
	@Override
	public double getDotProductInFeatureSpace(List<Double> inputSpaceVector1, List<Double> inputSpaceVector2) {
		return Math.pow(this.zeta + gamma * getDotProduct(inputSpaceVector1, inputSpaceVector2), degree);
	}

}
