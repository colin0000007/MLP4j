package com.outsider.model;

import org.junit.Test;

/**
 * ѧϰ��ָ��˥��
 * @author outsider
 *
 */
public class ExponentialLearningRateDecay {
	public static ExponentialLearningRateDecay defaultDecay = 
			new ExponentialLearningRateDecay(2, 0.2, 0.8, false);
	public static ExponentialLearningRateDecay noDecay = 
			new ExponentialLearningRateDecay(2, 0.2, 1, false);
	/**
	 * 
	 * ˥������:Խ��˥��Խ��
	 * ��staircaseΪtrueʱ��ÿһ��˥�����ڲ�˥��һ��
	 */
	public int decaySteps;
	/**
	 * ��ʼѧϰ��
	 */
	public double lr;
	/**
	 * ˥���ʣ�ע��0��1֮�䣬Խ��˥��Խ��
	 * �������˥����ֱ������Ϊ1
	 */
	public double decayRate;
	/**
	 * �Ƿ�ʽ�����˥�������Ϊtrue����ôÿ��һ��decaySteps˥��һ��
	 */
	public boolean staircase;
	
	public ExponentialLearningRateDecay(int decaySteps, double lr, double decayRate, boolean staircase) {
		this.decayRate = decayRate;
		this.decaySteps = decaySteps;
		this.lr = lr;
		this.staircase = staircase;
	}
	
	/**
	 * ����ĳ��globalStep�µ�lr
	 * @param globalStep ָ�ľ��ǵ������������������epoch��˥��lr��epoch=2ʱ��globalStep=2
	 * @return
	 */
	public double decayLR(int globalStep) {
		if (staircase) {
			return decayLRWithStaircase(globalStep);
		}
		return decayLRWithoutStaircase(globalStep);
	}
	
	private double decayLRWithoutStaircase(int globalStep) {
		return lr * Math.pow(decayRate,(globalStep*1.0/decaySteps));
	}
	@Test
	public void t3() {
		System.out.println(Math.pow(0.5, 0.5));
	}
	
	private double decayLRWithStaircase(int globalStep) {
		//globalStep/decaySteps����ֻ�����������֣������Ե�lr˥������ֱ������
		return lr * Math.pow(decayRate,(globalStep/decaySteps));
	}
	
	//����
	/*public static void main(String[] args) {
		ExponentialLearningRateDecay decay = new ExponentialLearningRateDecay(10, 0.5, 0.9, false);
		double[] lr = new double[200];
		for(int i = 0; i < 200; i++) {
			lr[i] = decay.decayLR(i);
		}
		System.out.println(Arrays.toString(lr));
	}
	*/
}
