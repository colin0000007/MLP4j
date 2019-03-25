package com.outsider.model.parallel;

import java.util.concurrent.CountDownLatch;

import org.jblas.DoubleMatrix;

import com.outsider.model.Gradient;
import com.outsider.model.Sequential;

public class MultiSampleGradientComputingThread implements Runnable{
	private Sequential sequential;
	private double[][] x;
	private double[][] y;
	public double[][][] totalWeightsGradient;
	public double[][] totalBiasesGradient;
	private int startIndex;//样本开始位置
	private int endIndex;//样本结束为止（注意不包含）
	private CountDownLatch countDownLatch;
	public MultiSampleGradientComputingThread(Sequential sequential, double[][] x, double[][] y,
			int startIndex, int endIndex, CountDownLatch countDownLatch) {
		this.sequential = sequential;
		this.x = x;
		this.y = y;
		this.startIndex = startIndex;
		this.endIndex = endIndex;
		this.countDownLatch = countDownLatch;
	}
	@Override
	public void run() {
		Gradient gradient = sequential.backPropagation(x[startIndex], y[startIndex]);
		totalWeightsGradient = gradient.weightsGradient;
		totalBiasesGradient = gradient.biasesGradient;
		for(int m = startIndex + 1; m < endIndex; m++) {
			Gradient gradient2 = sequential.backPropagation(x[m], y[m]);
			double[][][] weightsGradient = gradient2.weightsGradient;
			double[][] biasesGradient = gradient2.biasesGradient;
			for(int f = 0; f < weightsGradient.length; f++) {
				totalWeightsGradient[f] = new DoubleMatrix(totalWeightsGradient[f]).
						addi(new DoubleMatrix(weightsGradient[f])).toArray2();
				totalBiasesGradient[f] = new DoubleMatrix(totalBiasesGradient[f]).
						addi(new DoubleMatrix(biasesGradient[f])).toArray();
			}
		}
		countDownLatch.countDown();
	}
	
}
