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
	 * 注意这个函数必须传入layer中全部的z，不然计算会出问题
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
