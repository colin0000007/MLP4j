package com.outsider.model.activationFunction;

/**
 * sigmoid¼¤»îº¯Êý
 * @author outsider
 *
 */
public class Sigmoid implements ActivationFunction{
	public static final Sigmoid SIGMOID = new Sigmoid();
	
	private Sigmoid() {
	}
	
	@Override
	public double[] functionValue(double[] z) {
		double[] sigmaZ = new double[z.length];
		for(int i = 0; i < sigmaZ.length; i++) {
			sigmaZ[i] = 1 / (1+Math.exp(-z[i]));
		}
		return sigmaZ;
	}
	
	public double functionValue(double z) {
		return 1 / (1+Math.exp(-z));
	}
	
	
	@Override
	public double[] firstDerivativeValue(double[] z) {
		double[] sigmaPZ = new double[z.length];
		for(int i = 0; i < sigmaPZ.length; i++) {
			double fz = functionValue(z[i]);
			sigmaPZ[i] = fz * (1- fz);
		}
		return sigmaPZ;
	}
	

	@Override
	public double[] partialLoss2PartialZ(double[] z, double[] a, double[] y) {
		System.err.println("Not Implemented!");
		return null;
	}
	
}
