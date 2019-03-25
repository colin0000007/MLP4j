package com.outsider.model;

import org.junit.Test;

/**
 * 学习率指数衰减
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
	 * 衰减周期:越大衰减越快
	 * 当staircase为true时，每一个衰减周期才衰减一次
	 */
	public int decaySteps;
	/**
	 * 初始学习率
	 */
	public double lr;
	/**
	 * 衰减率，注意0到1之间，越大衰减越慢
	 * 如果不想衰减，直接设置为1
	 */
	public double decayRate;
	/**
	 * 是否呈阶梯型衰减，如果为true，那么每隔一个decaySteps衰减一次
	 */
	public boolean staircase;
	
	public ExponentialLearningRateDecay(int decaySteps, double lr, double decayRate, boolean staircase) {
		this.decayRate = decayRate;
		this.decaySteps = decaySteps;
		this.lr = lr;
		this.staircase = staircase;
	}
	
	/**
	 * 计算某个globalStep下的lr
	 * @param globalStep 指的就是迭代索引。比如如果以epoch来衰减lr，epoch=2时，globalStep=2
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
		//globalStep/decaySteps这里只保留整数部分，周期性的lr衰减可以直接体现
		return lr * Math.pow(decayRate,(globalStep/decaySteps));
	}
	
	//测试
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
