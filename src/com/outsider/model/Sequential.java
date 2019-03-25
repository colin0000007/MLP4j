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
 * layer序列
 * 类似keras的概念
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
	 * @param x 单个样本的x
	 * @param y 单个样本的y
	 * @return 每一层的梯度矩阵，注意是一个3维数组，每一个矩阵的尺寸不一样
	 */
	public Gradient backPropagation(double[] x, double[] y) {
		//(1) 前向传播
		ArrayList<double[]>[] za = forwardPropagation(x);
		ArrayList<double[]> z = za[0];
		ArrayList<double[]> a = za[1];
		//(2) 反向传播计算 partial L / partial Z,这直接是偏置的梯度
		double[][] LZ = new double[layers.size()][];
		//计算最后一层的 partial L / partial z
		LZ[LZ.length - 1] = layers.get(layers.size() - 1).getActivationFunction()
				.partialLoss2PartialZ(null, a.get(a.size() - 1), y);
		//计算其他层的 partial L / partial z 
		for(int i = LZ.length -2; i >=0; i--) {
			Dense layer = layers.get(i);
			//1.求激活函数的一阶导数
			double[] sigmaP = layer.getActivationFunction().firstDerivativeValue(z.get(i));
			//2.后一层的权重矩阵的转置乘以 partial L / partial z 向量
			DoubleMatrix m1 = new DoubleMatrix(layers.get(i + 1).getWeights()).
					transpose().mmul(new DoubleMatrix(LZ[i+1]));
			// 1和2得到的向量对应元素相乘即为结果
			LZ[i] = new DoubleMatrix(sigmaP).muli(m1).toArray();
		}
		//(3).计算梯度
		//partial L / partial Z的 列向量乘以上一层的输入a行向量
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
	 * 前向传播计算a和z这两个list
	 * 每个list保存了每一层的a或者z
	 * 只计算单个样本
	 * 每个样本的forwardPropagation可以并行，单个样本的forwardPropagation矩阵运算可以并行
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
			//计算这一层的输出
			ActivationFunction sigma = layers.get(i).getActivationFunction();
			double[] as = sigma.functionValue(zs);
			za[0].add(zs);
			za[1].add(as);
			input = as;
		}
		return za;
	}
	/**
	 * 串行计算多个样本的预测
	 * 如果处理器有多核，强烈不建议
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
	 * 并行预测，基于单个样本一个线程
	 * @param x
	 * @return
	 */
	public double[][] predictParallel(double[][] x){
		double[][] res = new double[x.length][];
		//使用线程池调度，最大线程个数为处理器+1(计算密集型建议这样设计)
		ExecutorService threadPool = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors()+1);
		//并行预测多个样本
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
	 * 并行预测，基于多个数据一个线程
	 * @param x
	 * @return
	 */
	public double[][] predictParallel2(double[][] x){
		double[][] res = new double[x.length][];
		//使用线程池调度，最大线程个数为处理器+1(计算密集型建议这样设计)
		//并行预测多个样本
		int pNum = Runtime.getRuntime().availableProcessors();
		CountDownLatch countDownLatch = new CountDownLatch(pNum);
		int sampleNum = x.length / pNum;
		int left = x.length % pNum;
		for(int i = 0; i < pNum; i++) {
			//(i/pNum)*left的意思时，当i时pNum时也就是最后一批时，如果有余数，那么应该加上余数
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
			//计算这一层的输出
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
	 * 使用并行预测
	 * @param x
	 * @return
	 */
	public int[] predictAndReturnClassIndex(double[][] x) {
		double[][] prba = predictParallel(x);
		DoubleMatrix m = new DoubleMatrix(prba);
		return m.rowArgmaxs();
	}
	
	
	
	/**
	 * 训练
	 * @param x 
	 * @param y 每一个标签必须one-hot编码，所以是二维数组
	 * @param lr 学习率
	 * @param batchSize 每次梯度下降的更新参数用到的样本个数
	 * @param epochs 在整个样本集上做多少次的训练
	 * @param lrDecay 学习率指数衰减
	 */
	public void train(double[][] x, double[][] y, 
			int batchSize, int epochs, ExponentialLearningRateDecay lrDecay) {
		//一个epoch中的各个batch可以并行，一个batch之间各单个样本之间可以并行
		double lr = lrDecay.lr;
		DoubleMatrix yM = new DoubleMatrix(y);
		int[] yTrue = yM.rowArgmaxs();
		this.classNum = layers.get(layers.size() - 1).getUnits();
		//1.划分样本
		int sampleNum = x.length;
		//分多少批
		int batchNum = sampleNum / batchSize;
		System.out.println("batchNum:"+batchNum);
		int left = sampleNum % batchSize;//样本余数
		//保存样本的索引位置
		List<Integer> sampleIndices = new ArrayList<>(x.length);
		for(int i = 0; i < sampleNum; i++) {
			sampleIndices.add(i);
		}
		// epoch循环
		for(int i = 0; i < epochs; i++) {
			//对学习率衰减
			lr = lrDecay.decayLR(i);
			System.out.println("epoch "+(i+1)+"/"+epochs+",lr="+lr);
			//batch循环
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
				//训练，一个一个样本计算梯度，这个过程可以同时进行。
				//将每个样本的梯度累加
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
				//梯度除以N
				for(int k = 0; k < layers.size(); k++) {
					totalBiasesGradient[k] = new DoubleMatrix(totalBiasesGradient[k])
							.divi(batchX.length).toArray();
					totalWeightsGradient[k] = new DoubleMatrix(totalWeightsGradient[k]).
							divi(batchX.length).toArray2();
				}
				//更新参数
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
				//每20倍批次输出一次训练信息
				/*if((j+1) % 20 == 0) {
					int[] yPredicted = predictAndReturnClassIndex(x);
					//训练集上的准确率
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
			//打乱数据位置
			//acc会有一些不稳定，估计也和这个操作有关系。
			//测试后发现，shuffle有利用梯度下降
			Collections.shuffle(sampleIndices);
		}
	}
	/**
	 * 并行训练
	 * 推荐使用
	 * @param x
	 * @param y
	 * @param batchSize
	 * @param epochs
	 * @param lrDecay
	 */
	public void trainParallel(double[][] x, double[][] y, 
			int batchSize, int epochs, ExponentialLearningRateDecay lrDecay) {
		//一个epoch中的各个batch可以并行，一个batch之间各单个样本之间可以并行
		double lr = lrDecay.lr;
		DoubleMatrix yM = new DoubleMatrix(y);
		int[] yTrue = yM.rowArgmaxs();
		this.classNum = layers.get(layers.size() - 1).getUnits();
		//1.划分样本
		int sampleNum = x.length;
		//分多少批
		int batchNum = sampleNum / batchSize;
		System.out.println("batchNum:"+batchNum);
		int left = sampleNum % batchSize;//样本余数
		//保存样本的索引位置
		List<Integer> sampleIndices = new ArrayList<>(x.length);
		for(int i = 0; i < sampleNum; i++) {
			sampleIndices.add(i);
		}
		// epoch循环
		for(int i = 0; i < epochs; i++) {
			//对学习率衰减
			lr = lrDecay.decayLR(i);
			System.out.println("epoch "+(i+1)+"/"+epochs+",lr="+lr);
			//batch循环
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
				//训练，一个一个样本计算梯度，这个过程可以同时进行。
				//将每个样本的梯度累加
				//（1）需要同时对总的梯度这个变量修改，这样的存在互斥
				//（2）不然就只有并行计算所有梯度而再起来
				//目前来看（2）不可能，如果保存所有的梯度需要一个超大的四维数组，内存根本不够
				//我只简单的测试了double[][][][] max= new double[400][4][400][400]就爆内存了
				//分批并行：批内之间进行梯度的叠加，最后临时变量
				double[][][] totalWeightsGradient = null;
				double[][] totalBiasesGradient = null;
				int pNum = Runtime.getRuntime().availableProcessors();
				int batchSizeOfThread = batchX.length / pNum;
				int leftSample = batchX.length % pNum;//余下的样本
				List<MultiSampleGradientComputingThread> runners = new ArrayList<>();
				CountDownLatch countDownLatch = new CountDownLatch(pNum);
				for(int k = 0; k < pNum; k++) {
					MultiSampleGradientComputingThread runner = 
							new MultiSampleGradientComputingThread(this, batchX, bathcY, k * batchSizeOfThread , (k+1)*batchSizeOfThread + ((k+1) / pNum)*leftSample, countDownLatch);
					runners.add(runner);
					new Thread(runner).start();
				}
				//等待子线程运行完成
				try {
					countDownLatch.await();
				} catch (InterruptedException e) {
					e.printStackTrace();
				}
				//累加子线程中的梯度
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
				//梯度除以N
				for(int k = 0; k < layers.size(); k++) {
					totalBiasesGradient[k] = new DoubleMatrix(totalBiasesGradient[k])
							.divi(batchX.length).toArray();
					totalWeightsGradient[k] = new DoubleMatrix(totalWeightsGradient[k]).
							divi(batchX.length).toArray2();
				}
				//更新参数
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
				//每20倍批次输出一次训练信息
				/*if((j+1) % 20 == 0) {
					int[] yPredicted = predictAndReturnClassIndex(x);
					//训练集上的准确率
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
			//打乱数据位置
			//acc会有一些不稳定，估计也和这个操作有关系。
			//测试后发现，shuffle有利用梯度下降
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
	 * 计算全局loss，此处是交叉熵loss
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
