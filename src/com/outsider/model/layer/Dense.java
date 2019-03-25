package com.outsider.model.layer;

import org.jblas.DoubleMatrix;

import com.outsider.model.activationFunction.ActivationFunction;
/**
 * 全连接层
 * @author outsider
 *
 */
public class Dense extends Layer{
	//wij 指的时当前层第i个单元到上一层的j个单元连接的权重。
	//所以weights的行数是本层的单元个数，列数是上一层单元的个数
	private double[][] weights;
	//偏置
	private double[] biases;
	//当前层的单元个数
	private int units;
	//激活函数
	private ActivationFunction activationFunction;
	public Dense(int units, ActivationFunction activationFunction) {
		this.units = units;
		this.activationFunction  = activationFunction;
	}
	/**
	 * 初始化权重参数
	 * @param lastUnits
	 */
	public void init(int lastUnits) {
		this.biases = new double[units];
		//权重随机初始化
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
