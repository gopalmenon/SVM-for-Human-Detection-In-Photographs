import java.util.List;


/**
 * The Gaussian RBF kernel returns exp(-gamma * ||x1-x2||)
 *
 */
public class GaussianRadialBasisFunctionKernel implements Kernel {
	
	private double gaussianKernalWidth;
	
	/**
	 * Constructor
	 * @param gaussianKernalWidth controls the width of the kernel
	 */
	public GaussianRadialBasisFunctionKernel(double gaussianKernalWidth) {
		this.gaussianKernalWidth = gaussianKernalWidth;
	}

	/* (non-Javadoc)
	 * @see Kernel#getDotProductInFeatureSpace(java.util.List, java.util.List)
	 */
	@Override
	public double getDotProductInFeatureSpace(List<Double> inputSpaceVector1, List<Double> inputSpaceVector2) {
		
		return Math.pow(Math.E, -1 * this.gaussianKernalWidth * Kernel.getVectorNormSquared(Kernel.getDifferenceVector(inputSpaceVector1, inputSpaceVector2)));

	}
	
}
