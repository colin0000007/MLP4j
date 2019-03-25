package com.outsider.model.activationFunction;
/**
 * ������ӿ�
 * @author outsider
 *
 */
public interface ActivationFunction {
	public static Sigmoid SIGMOID = Sigmoid.SIGMOID;
	public static Relu RELU = Relu.RELU;
	public static SoftMax SOFTMAX = SoftMax.SOFTMAX;
	/**
	 * ������ĺ���ֵ
	 * @param z ��(z)������z,����һά����ָ������һ���z
	 * @return
	 */
	public double[] functionValue(double[] z);
	/**
	 * �������һ�׵���ֵ
	 * @param z ��'(z)������z������һά����ָ������һ���z
	 * @return
	 */
	public double[] firstDerivativeValue(double[] z);
	
	/**
	 * ����ü��������������㣬��ô��Ҫ
	 * ����������partial L / partial z
	 * ����a��z��yֻ����ΪԤѡ�����Ǳ�Ȼ����õ���3������
	 * @param z �����z
	 * @param a �����a
	 * @param y ��ʵ������y
	 * @return
	 */
	public double[] partialLoss2PartialZ(double[] z, double[] a, double[] y);
}
