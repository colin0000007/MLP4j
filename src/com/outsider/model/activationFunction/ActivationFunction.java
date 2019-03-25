package com.outsider.model.activationFunction;
/**
 * 激活函数接口
 * @author outsider
 *
 */
public interface ActivationFunction {
	public static Sigmoid SIGMOID = Sigmoid.SIGMOID;
	public static Relu RELU = Relu.RELU;
	public static SoftMax SOFTMAX = SoftMax.SOFTMAX;
	/**
	 * 激活函数的函数值
	 * @param z σ(z)的输入z,这里一维数组指代的是一层的z
	 * @return
	 */
	public double[] functionValue(double[] z);
	/**
	 * 激活函数的一阶导数值
	 * @param z σ'(z)的输入z，这里一维数组指代的是一层的z
	 * @return
	 */
	public double[] firstDerivativeValue(double[] z);
	
	/**
	 * 如果该激活函数作用在输出层，那么需要
	 * 计算输出层的partial L / partial z
	 * 参数a，z，y只是作为预选，但是必然最多用到这3个参数
	 * @param z 本层的z
	 * @param a 本层的a
	 * @param y 真实样本的y
	 * @return
	 */
	public double[] partialLoss2PartialZ(double[] z, double[] a, double[] y);
}
