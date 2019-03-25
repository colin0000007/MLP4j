package com.outsider.model.activationFunction;

import org.jblas.DoubleMatrix;

/**
 * RELU¼¤»îº¯Êý
 * @author outsider
 *
 */
public class Relu implements ActivationFunction{
	public static final Relu RELU = new Relu();
	private Relu() {
	}
	@Override
	public double[] functionValue(double[] z) {
		double[] sigmaZ = new double[z.length];
		for(int i = 0; i < sigmaZ.length; i++) {
			sigmaZ[i] = z[i] > 0 ? z[i] : 0;
		}
		return sigmaZ;
	}

	@Override
	public double[] firstDerivativeValue(double[] z) {
		double[] sigmaPZ = new double[z.length];
		for(int i = 0; i < sigmaPZ.length; i++) {
			sigmaPZ[i] = z[i] > 0 ? 1 : 0;
		}
		return sigmaPZ;
	}

	@Override
	public double[] partialLoss2PartialZ(double[] z, double[] a, double[] y) {
		System.err.println("Not Implemented!");
		return null;
	}

}
