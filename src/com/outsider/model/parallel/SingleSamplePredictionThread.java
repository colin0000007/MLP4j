package com.outsider.model.parallel;

import java.util.concurrent.CountDownLatch;

import com.outsider.model.Sequential;

public class SingleSamplePredictionThread implements Runnable{
	
	private Sequential sequential;
	private double[] x;
	//当前样本在多个样本的索引是多少，执行完后返回，方便顺序存储结果
	private int sampleIndex;
	//用于实现主线程等待子线程
	private CountDownLatch countDownLatch;
	//总的预测结果，这里引用为线程的公共资源
	private double[][] result;
	public SingleSamplePredictionThread(Sequential sequential, double[] x, int sampleIndex
			,CountDownLatch countDownLatch, double[][] result) {
		this.sequential = sequential;
		this.x = x;
		this.sampleIndex = sampleIndex;
		this.countDownLatch = countDownLatch;
		this.result = result;
	}
	
	@Override
	public void run() {
		result[sampleIndex] = sequential.predict(x);
		countDownLatch.countDown();
	}
	
}
