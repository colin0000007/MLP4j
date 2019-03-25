package com.outsider.model;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

import com.outsider.model.activationFunction.ActivationFunction;
import com.outsider.model.layer.Dense;
import com.outsider.model.layer.Input;
import com.outsider.model.parallel.MultiSampleGradientComputingThread;
import com.outsider.model.parallel.MultiSamplePredictionThread;
import com.outsider.model.parallel.SingleSamplePredictionThread;
/**
 * layer����
 * ����keras�ĸ���
 * @author outsider
 *
 */
public class Sequential {
	private List<Dense> layers = new ArrayList<>();
	private int lastUnits = 0;
	private int classNum;
	public void setInput(Input input) {
		this.lastUnits = input.getInputDim();
	}
	
	public Sequential addLayer(Dense dense) {
		if(lastUnits <= 0) {
			System.err.println("make sure units is greater than zero or the Input Layer is set!");
		}
		layers.add(dense);
		dense.init(lastUnits);
		lastUnits = dense.getUnits();
		return this;
	}
	
	/**
	 * 
	 * @param x ����������x
	 * @param y ����������y
	 * @return ÿһ����ݶȾ���ע����һ��3ά���飬ÿһ������ĳߴ粻һ��
	 */
	public Gradient backPropagation(double[] x, double[] y) {
		//(1) ǰ�򴫲�
		ArrayList<double[]>[] za = forwardPropagation(x);
		ArrayList<double[]> z = za[0];
		ArrayList<double[]> a = za[1];
		//(2) ���򴫲����� partial L / partial Z,��ֱ����ƫ�õ��ݶ�
		double[][] LZ = new double[layers.size()][];
		//�������һ��� partial L / partial z
		LZ[LZ.length - 1] = layers.get(layers.size() - 1).getActivationFunction()
				.partialLoss2PartialZ(null, a.get(a.size() - 1), y);
		//����������� partial L / partial z 
		for(int i = LZ.length -2; i >=0; i--) {
			Dense layer = layers.get(i);
			//1.�󼤻����һ�׵���
			double[] sigmaP = layer.getActivationFunction().firstDerivativeValue(z.get(i));
			//2.��һ���Ȩ�ؾ����ת�ó��� partial L / partial z ����
			DoubleMatrix m1 = new DoubleMatrix(layers.get(i + 1).getWeights()).
					transpose().mmul(new DoubleMatrix(LZ[i+1]));
			// 1��2�õ���������ӦԪ����˼�Ϊ���
			LZ[i] = new DoubleMatrix(sigmaP).muli(m1).toArray();
		}
		//(3).�����ݶ�
		//partial L / partial Z�� ������������һ�������a������
		double[] input = x;
		double[][][] gradients = new double[layers.size()][][];
		for(int i = 0; i < gradients.length; i++) {
			DoubleMatrix aVec = new DoubleMatrix(input);
			DoubleMatrix lzVec = new DoubleMatrix(LZ[i]);
			aVec = aVec.reshape(aVec.columns, aVec.rows);
			gradients[i] = lzVec.mmul(aVec).toArray2();
			input = a.get(i);
		}
		Gradient gradient = new Gradient(gradients, LZ);
		return gradient;
	}
	
	/**
	 * ǰ�򴫲�����a��z������list
	 * ÿ��list������ÿһ���a����z
	 * ֻ���㵥������
	 * ÿ��������forwardPropagation���Բ��У�����������forwardPropagation����������Բ���
	 * @return
	 */
	public ArrayList<double[]>[] forwardPropagation(double[] x) {
		ArrayList<double[]>[] za = new ArrayList[2];
		za[0] = new ArrayList<>(layers.size());
		za[1] = new ArrayList<>(layers.size());
		double[] input = x;
		for(int i = 0; i < layers.size(); i++) {
			double[][] weights = layers.get(i).getWeights();
			double[] biases = layers.get(i).getBiases();
			DoubleMatrix w = new DoubleMatrix(weights);
			DoubleMatrix a = new DoubleMatrix(input);
			DoubleMatrix b = new DoubleMatrix(biases);
			// z = w*input + b
			DoubleMatrix zMatrix = w.mmul(a).addi(b);
			double[] zs = zMatrix.toArray();
			//������һ������
			ActivationFunction sigma = layers.get(i).getActivationFunction();
			double[] as = sigma.functionValue(zs);
			za[0].add(zs);
			za[1].add(as);
			input = as;
		}
		return za;
	}
	/**
	 * ���м�����������Ԥ��
	 * ����������ж�ˣ�ǿ�Ҳ�����
	 * @param x
	 * @return
	 */
	public double[][] predict(double[][] x){
		double[][] res = new double[x.length][];
		for(int i = 0; i < x.length; i++) {
			res[i] = predict(x[i]);
		}
		return res;
	}
	/**
	 * ����Ԥ�⣬���ڵ�������һ���߳�
	 * @param x
	 * @return
	 */
	public double[][] predictParallel(double[][] x){
		double[][] res = new double[x.length][];
		//ʹ���̳߳ص��ȣ�����̸߳���Ϊ������+1(�����ܼ��ͽ����������)
		ExecutorService threadPool = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors()+1);
		//����Ԥ��������
		CountDownLatch countDownLatch = new CountDownLatch(x.length);
		for(int i = 0; i < x.length; i++) {
			SingleSamplePredictionThread thread = new SingleSamplePredictionThread(this, x[i], i, countDownLatch,res);
			threadPool.submit(thread);
		}
		try {
			countDownLatch.await();
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
		threadPool.shutdown();
		return res;
	}
	/**
	 * ����Ԥ�⣬���ڶ������һ���߳�
	 * @param x
	 * @return
	 */
	public double[][] predictParallel2(double[][] x){
		double[][] res = new double[x.length][];
		//ʹ���̳߳ص��ȣ�����̸߳���Ϊ������+1(�����ܼ��ͽ����������)
		//����Ԥ��������
		int pNum = Runtime.getRuntime().availableProcessors();
		CountDownLatch countDownLatch = new CountDownLatch(pNum);
		int sampleNum = x.length / pNum;
		int left = x.length % pNum;
		for(int i = 0; i < pNum; i++) {
			//(i/pNum)*left����˼ʱ����iʱpNumʱҲ�������һ��ʱ���������������ôӦ�ü�������
			MultiSamplePredictionThread runer = 
					new MultiSamplePredictionThread(this, x, i*sampleNum, (i+1)*sampleNum+((i+1)/pNum)*left, countDownLatch, res);
			new Thread(runer).start();
		}
		try {
			countDownLatch.await();
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
		return res;
	}
	
	public double[] predict(double[] x) {
		double[] input = x;
		for(int i = 0; i < layers.size(); i++) {
			double[][] weights = layers.get(i).getWeights();
			double[] biases = layers.get(i).getBiases();
			DoubleMatrix w = new DoubleMatrix(weights);
			DoubleMatrix a = new DoubleMatrix(input);
			DoubleMatrix b = new DoubleMatrix(biases);
			// z = w*input + b
			DoubleMatrix zMatrix = w.mmul(a).addi(b);
			double[] zs = zMatrix.toArray();
			//������һ������
			ActivationFunction sigma = layers.get(i).getActivationFunction();
			double[] as = sigma.functionValue(zs);
			input = as;
		}
		return input;
	}
	
	public int predictAndReturnClassIndex(double[] x) {
		double[] prba = predict(x);
		DoubleMatrix m = new DoubleMatrix(prba);
		return m.argmax();
	}
	/**
	 * ʹ�ò���Ԥ��
	 * @param x
	 * @return
	 */
	public int[] predictAndReturnClassIndex(double[][] x) {
		double[][] prba = predictParallel(x);
		DoubleMatrix m = new DoubleMatrix(prba);
		return m.rowArgmaxs();
	}
	
	
	
	/**
	 * ѵ��
	 * @param x 
	 * @param y ÿһ����ǩ����one-hot���룬�����Ƕ�ά����
	 * @param lr ѧϰ��
	 * @param batchSize ÿ���ݶ��½��ĸ��²����õ�����������
	 * @param epochs �������������������ٴε�ѵ��
	 * @param lrDecay ѧϰ��ָ��˥��
	 */
	public void train(double[][] x, double[][] y, 
			int batchSize, int epochs, ExponentialLearningRateDecay lrDecay) {
		//һ��epoch�еĸ���batch���Բ��У�һ��batch֮�����������֮����Բ���
		double lr = lrDecay.lr;
		DoubleMatrix yM = new DoubleMatrix(y);
		int[] yTrue = yM.rowArgmaxs();
		this.classNum = layers.get(layers.size() - 1).getUnits();
		//1.��������
		int sampleNum = x.length;
		//�ֶ�����
		int batchNum = sampleNum / batchSize;
		System.out.println("batchNum:"+batchNum);
		int left = sampleNum % batchSize;//��������
		//��������������λ��
		List<Integer> sampleIndices = new ArrayList<>(x.length);
		for(int i = 0; i < sampleNum; i++) {
			sampleIndices.add(i);
		}
		// epochѭ��
		for(int i = 0; i < epochs; i++) {
			//��ѧϰ��˥��
			lr = lrDecay.decayLR(i);
			System.out.println("epoch "+(i+1)+"/"+epochs+",lr="+lr);
			//batchѭ��
			for(int j = 0; j < batchNum; j++) {
				int c = left *((j+1)/batchNum);
				double[][] batchX = new double[batchSize + c][];
				double[][] bathcY = new double[batchSize + c][];
				System.out.println("epoch..."+(i+1)+"/"+epochs+",batch..."+(j+1)+"/"+batchNum+","+"batchSize="+batchX.length);
				int offset = j * batchSize;
				for(int k = j * batchSize; k < (j+1)*batchSize + c; k++) {
					batchX[k-offset] = x[sampleIndices.get(k)];
					bathcY[k-offset] = y[sampleIndices.get(k)];
				}
				//ѵ����һ��һ�����������ݶȣ�������̿���ͬʱ���С�
				//��ÿ���������ݶ��ۼ�
				Gradient gradient = backPropagation(batchX[0], bathcY[0]);
				double[][][] totalWeightsGradient = gradient.weightsGradient;
				double[][] totalBiasesGradient = gradient.biasesGradient;
				for(int m = 1; m < batchX.length; m++) {
					Gradient gradient2 = backPropagation(batchX[m], bathcY[m]);
					double[][][] weightsGradient = gradient2.weightsGradient;
					double[][] biasesGradient = gradient2.biasesGradient;
					for(int f = 0; f < weightsGradient.length; f++) {
						totalWeightsGradient[f] = new DoubleMatrix(totalWeightsGradient[f]).
								addi(new DoubleMatrix(weightsGradient[f])).toArray2();
						totalBiasesGradient[f] = new DoubleMatrix(totalBiasesGradient[f]).
								addi(new DoubleMatrix(biasesGradient[f])).toArray();
					}
				}
				//�ݶȳ���N
				for(int k = 0; k < layers.size(); k++) {
					totalBiasesGradient[k] = new DoubleMatrix(totalBiasesGradient[k])
							.divi(batchX.length).toArray();
					totalWeightsGradient[k] = new DoubleMatrix(totalWeightsGradient[k]).
							divi(batchX.length).toArray2();
				}
				//���²���
				for(int f = 0; f < layers.size(); f++) {
					Dense dense = layers.get(f);
					double[][] oldW = dense.getWeights();
					double[] oldBia = dense.getBiases();
					// w = w - lr * gradient
					double[][] newW = new DoubleMatrix(oldW).
							subi(new DoubleMatrix(totalWeightsGradient[f]).mul(lr)).toArray2();
					double[] newBia = new DoubleMatrix(oldBia).
							subi(new DoubleMatrix(totalBiasesGradient[f]).mul(lr)).toArray();
					dense.setWeights(newW);
					dense.setBiases(newBia);
				}
				//ÿ20���������һ��ѵ����Ϣ
				/*if((j+1) % 20 == 0) {
					int[] yPredicted = predictAndReturnClassIndex(x);
					//ѵ�����ϵ�׼ȷ��
					float acc = accuracy(yPredicted, yTrue);
					System.out.println("\n"+(j+1)*batchSize + "/" + x.length+"........acc:"+acc+"\n");
				}*/
			}
			double[][] yPre = predictParallel(x);
			int[] yPredicted = new DoubleMatrix(yPre).rowArgmaxs();
			double loss = loss(yPre, y);
			float acc = accuracy(yPredicted, yTrue);
			String print = "epoch " +(i+1)+ " done"+"........acc:"+acc+",total loss:"+loss;
			System.out.println(print);
			//��������λ��
			//acc����һЩ���ȶ�������Ҳ����������й�ϵ��
			//���Ժ��֣�shuffle�������ݶ��½�
			Collections.shuffle(sampleIndices);
		}
	}
	/**
	 * ����ѵ��
	 * �Ƽ�ʹ��
	 * @param x
	 * @param y
	 * @param batchSize
	 * @param epochs
	 * @param lrDecay
	 */
	public void trainParallel(double[][] x, double[][] y, 
			int batchSize, int epochs, ExponentialLearningRateDecay lrDecay) {
		//һ��epoch�еĸ���batch���Բ��У�һ��batch֮�����������֮����Բ���
		double lr = lrDecay.lr;
		DoubleMatrix yM = new DoubleMatrix(y);
		int[] yTrue = yM.rowArgmaxs();
		this.classNum = layers.get(layers.size() - 1).getUnits();
		//1.��������
		int sampleNum = x.length;
		//�ֶ�����
		int batchNum = sampleNum / batchSize;
		System.out.println("batchNum:"+batchNum);
		int left = sampleNum % batchSize;//��������
		//��������������λ��
		List<Integer> sampleIndices = new ArrayList<>(x.length);
		for(int i = 0; i < sampleNum; i++) {
			sampleIndices.add(i);
		}
		// epochѭ��
		for(int i = 0; i < epochs; i++) {
			//��ѧϰ��˥��
			lr = lrDecay.decayLR(i);
			System.out.println("epoch "+(i+1)+"/"+epochs+",lr="+lr);
			//batchѭ��
			for(int j = 0; j < batchNum; j++) {
				int c = left *((j+1)/batchNum);
				double[][] batchX = new double[batchSize + c][];
				double[][] bathcY = new double[batchSize + c][];
				System.out.println("epoch..."+(i+1)+"/"+epochs+",batch..."+(j+1)+"/"+batchNum+","+"batchSize="+batchX.length);
				int offset = j * batchSize;
				for(int k = j * batchSize; k < (j+1)*batchSize + c; k++) {
					batchX[k-offset] = x[sampleIndices.get(k)];
					bathcY[k-offset] = y[sampleIndices.get(k)];
				}
				//ѵ����һ��һ�����������ݶȣ�������̿���ͬʱ���С�
				//��ÿ���������ݶ��ۼ�
				//��1����Ҫͬʱ���ܵ��ݶ���������޸ģ������Ĵ��ڻ���
				//��2����Ȼ��ֻ�в��м��������ݶȶ�������
				//Ŀǰ������2�������ܣ�����������е��ݶ���Ҫһ���������ά���飬�ڴ��������
				//��ֻ�򵥵Ĳ�����double[][][][] max= new double[400][4][400][400]�ͱ��ڴ���
				//�������У�����֮������ݶȵĵ��ӣ������ʱ����
				double[][][] totalWeightsGradient = null;
				double[][] totalBiasesGradient = null;
				int pNum = Runtime.getRuntime().availableProcessors();
				int batchSizeOfThread = batchX.length / pNum;
				int leftSample = batchX.length % pNum;//���µ�����
				List<MultiSampleGradientComputingThread> runners = new ArrayList<>();
				CountDownLatch countDownLatch = new CountDownLatch(pNum);
				for(int k = 0; k < pNum; k++) {
					MultiSampleGradientComputingThread runner = 
							new MultiSampleGradientComputingThread(this, batchX, bathcY, k * batchSizeOfThread , (k+1)*batchSizeOfThread + ((k+1) / pNum)*leftSample, countDownLatch);
					runners.add(runner);
					new Thread(runner).start();
				}
				//�ȴ����߳��������
				try {
					countDownLatch.await();
				} catch (InterruptedException e) {
					e.printStackTrace();
				}
				//�ۼ����߳��е��ݶ�
				totalWeightsGradient = runners.get(0).totalWeightsGradient;
				totalBiasesGradient = runners.get(0).totalBiasesGradient;
				for(int k = 1; k < runners.size(); k++) {
					double[][][] weightsGradient = runners.get(k).totalWeightsGradient;
					double[][] biasesGradient = runners.get(k).totalBiasesGradient;
					for(int f = 0; f < weightsGradient.length; f++) {
						totalWeightsGradient[f] = new DoubleMatrix(totalWeightsGradient[f]).
								addi(new DoubleMatrix(weightsGradient[f])).toArray2();
						totalBiasesGradient[f] = new DoubleMatrix(totalBiasesGradient[f]).
								addi(new DoubleMatrix(biasesGradient[f])).toArray();
					}
				}
				//�ݶȳ���N
				for(int k = 0; k < layers.size(); k++) {
					totalBiasesGradient[k] = new DoubleMatrix(totalBiasesGradient[k])
							.divi(batchX.length).toArray();
					totalWeightsGradient[k] = new DoubleMatrix(totalWeightsGradient[k]).
							divi(batchX.length).toArray2();
				}
				//���²���
				for(int f = 0; f < layers.size(); f++) {
					Dense dense = layers.get(f);
					double[][] oldW = dense.getWeights();
					double[] oldBia = dense.getBiases();
					// w = w - lr * gradient
					double[][] newW = new DoubleMatrix(oldW).
							subi(new DoubleMatrix(totalWeightsGradient[f]).mul(lr)).toArray2();
					double[] newBia = new DoubleMatrix(oldBia).
							subi(new DoubleMatrix(totalBiasesGradient[f]).mul(lr)).toArray();
					dense.setWeights(newW);
					dense.setBiases(newBia);
				}
				//ÿ20���������һ��ѵ����Ϣ
				/*if((j+1) % 20 == 0) {
					int[] yPredicted = predictAndReturnClassIndex(x);
					//ѵ�����ϵ�׼ȷ��
					float acc = accuracy(yPredicted, yTrue);
					System.out.println("\n"+(j+1)*batchSize + "/" + x.length+"........acc:"+acc+"\n");
				}*/
			}
			double[][] yPre = predictParallel(x);
			int[] yPredicted = new DoubleMatrix(yPre).rowArgmaxs();
			double loss = loss(yPre, y);
			float acc = accuracy(yPredicted, yTrue);
			String print = "epoch " +(i+1)+ " done"+"........acc:"+acc+",total loss:"+loss;
			System.out.println(print);
			//��������λ��
			//acc����һЩ���ȶ�������Ҳ����������й�ϵ��
			//���Ժ��֣�shuffle�������ݶ��½�
			Collections.shuffle(sampleIndices);
		}
	}
	
	public float accuracy(int[] yPredicted, int[] yTrue) {
		int count = 0;
		for(int i = 0; i < yPredicted.length; i++) {
			if(yPredicted[i] == yTrue[i]) count++;
		}
		return (float) (count*1.0 / yTrue.length);
	}
	
	/**
	 * ����ȫ��loss���˴��ǽ�����loss
	 * @param x
	 * @param y
	 * @return
	 */
	public static double loss(double[][] yPredicted, double[][] yTrue) {
		int N = yTrue.length;
		double loss = 0;
		for(int i = 0; i < N; i++) {
			DoubleMatrix v1 = new DoubleMatrix(yTrue[i]);
			DoubleMatrix v2 = new DoubleMatrix(yPredicted[i]);
			v2 = MatrixFunctions.logi(v2);
			loss += (-v1.muli(v2).sum());
		}
		return loss / N;
	}
}
