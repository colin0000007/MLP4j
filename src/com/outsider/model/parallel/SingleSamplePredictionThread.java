package com.outsider.model.parallel;

import java.util.concurrent.CountDownLatch;

import com.outsider.model.Sequential;

public class SingleSamplePredictionThread implements Runnable{
	
	private Sequential sequential;
	private double[] x;
	//��ǰ�����ڶ�������������Ƕ��٣�ִ����󷵻أ�����˳��洢���
	private int sampleIndex;
	//����ʵ�����̵߳ȴ����߳�
	private CountDownLatch countDownLatch;
	//�ܵ�Ԥ��������������Ϊ�̵߳Ĺ�����Դ
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
