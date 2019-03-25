package com.outsider.model;

/**
 * 梯度对象
 * 只是作为保存梯度使用
 * @author outsider
 *
 */
public class Gradient {
	/**
	 * 权重梯度
	 */
	public double[][][] weightsGradient;
	/**
	 * 偏置梯度
	 */
	public double[][] biasesGradient;
	
	public Gradient() {}
	public Gradient(double[][][] weightsGradient, double[][] biasesGradient) {
		this.weightsGradient = weightsGradient;
		this.biasesGradient = biasesGradient;
	}
}
