package com.outsider.model.parallel;

import java.util.concurrent.CountDownLatch;

import com.outsider.model.Sequential;

public class MultiSamplePredictionThread implements Runnable{
	
	private Sequential sequential;
	private double[][] x;
	//��ǰ�����ڶ�������������Ƕ��٣�ִ����󷵻أ�����˳��洢���
	private int startIndex;
	private int endIndex;
	//����ʵ�����̵߳ȴ����߳�
	private CountDownLatch countDownLatch;
	//�ܵ�Ԥ��������������Ϊ�̵߳Ĺ�����Դ
	private double[][] result;
	public MultiSamplePredictionThread(Sequential sequential, double[][] x, int startIndex
			,int endIndex, CountDownLatch countDownLatch, double[][] result) {
		this.sequential = sequential;
		this.x = x;
		this.countDownLatch = countDownLatch;
		this.result = result;
		this.startIndex =startIndex;
		this.endIndex = endIndex;
	}
	
	@Override
	public void run() {
		for(int i = startIndex; i < endIndex; i++) {
			result[i] = sequential.predict(x[i]);
		}
		countDownLatch.countDown();
	}
	
}
