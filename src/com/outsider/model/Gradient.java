package com.outsider.model;

/**
 * �ݶȶ���
 * ֻ����Ϊ�����ݶ�ʹ��
 * @author outsider
 *
 */
public class Gradient {
	/**
	 * Ȩ���ݶ�
	 */
	public double[][][] weightsGradient;
	/**
	 * ƫ���ݶ�
	 */
	public double[][] biasesGradient;
	
	public Gradient() {}
	public Gradient(double[][][] weightsGradient, double[][] biasesGradient) {
		this.weightsGradient = weightsGradient;
		this.biasesGradient = biasesGradient;
	}
}
