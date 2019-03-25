package com.outsider.model.layer;

import org.jblas.DoubleMatrix;

import com.outsider.model.activationFunction.ActivationFunction;
/**
 * ȫ���Ӳ�
 * @author outsider
 *
 */
public class Dense extends Layer{
	//wij ָ��ʱ��ǰ���i����Ԫ����һ���j����Ԫ���ӵ�Ȩ�ء�
	//����weights�������Ǳ���ĵ�Ԫ��������������һ�㵥Ԫ�ĸ���
	private double[][] weights;
	//ƫ��
	private double[] biases;
	//��ǰ��ĵ�Ԫ����
	private int units;
	//�����
	private ActivationFunction activationFunction;
	public Dense(int units, ActivationFunction activationFunction) {
		this.units = units;
		this.activationFunction  = activationFunction;
	}
	/**
	 * ��ʼ��Ȩ�ز���
	 * @param lastUnits
	 */
	public void init(int lastUnits) {
		this.biases = new double[units];
		//Ȩ�������ʼ��
		this.weights = DoubleMatrix.randn(units, lastUnits).divi(10).toArray2();
		DoubleMatrix doubleMatrix = new DoubleMatrix();
	}
	
	public int getUnits() {
		return units;
	}
	
	public double[][] getWeights() {
		return weights;
	}
	
	public double[] getBiases() {
		return biases;
	}
	
	public ActivationFunction getActivationFunction() {
		return activationFunction;
	}
	
	public void setWeights(double[][] weights) {
		this.weights = weights;
	}
	public void setBiases(double[] biases) {
		this.biases = biases;
	}
	
	public static void main(String[] args) {
		System.out.println(DoubleMatrix.randn(4, 4));
	}
}
