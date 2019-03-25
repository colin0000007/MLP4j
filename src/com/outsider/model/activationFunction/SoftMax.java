package com.outsider.model.activationFunction;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

public class SoftMax implements ActivationFunction{
	public static final SoftMax SOFTMAX= new SoftMax();
	private SoftMax() {
	}
	
	@Override
	public double[] partialLoss2PartialZ(double[] z, double[] a, double[] y) {
		if(a != null) {
			// partial L / partial z = a - y;
			return new DoubleMatrix(a).subi(new DoubleMatrix(y)).toArray();
		}
		double[] as = functionValue(z);
		return new DoubleMatrix(as).subi(new DoubleMatrix(y)).toArray();
	}
	
	/**
	 * ע������������봫��layer��ȫ����z����Ȼ����������
	 */
	@Override
	public double[] functionValue(double[] z) {
		DoubleMatrix zMatrix = new DoubleMatrix(z);
		DoubleMatrix zEXPMatrix = MatrixFunctions.exp(zMatrix);
		double total = zEXPMatrix.sum();
		double[] result = new double[z.length];
		for(int i = 0; i < result.length; i++) {
			result[i] = zEXPMatrix.get(i) / total;
		}
		return result;
	}
	
	@Override
	public double[] firstDerivativeValue(double[] z) {
		System.err.println("Not Implemented!");
		return null;
	}
	
}
