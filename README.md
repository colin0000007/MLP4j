# MLP4j
使用直接参照源码中的MinistExample：
```
/**
 * 手写识别体的例子测试
 * @author outsider
 *
 */
public class MinistExample {
	public static double[][] x;
	public static double[][] y;
	public static void main(String[] args) {
		loadData();
		Sequential model = new Sequential();
		//必须首先设置Input
		model.setInput(new Input(x[0].length));
		model.addLayer(new Dense(400,ActivationFunction.RELU));
		model.addLayer(new Dense(400,ActivationFunction.RELU));
		model.addLayer(new Dense(10,SoftMax.SOFTMAX));
		System.out.println("x_train:("+x.length+","+x[0].length+")");
		//衰减可以使用默认的ExponentialLearningRateDecay.defaultDecay，默认lr初始0.2可以修改
		//model.trainParallel(x, y, 400, 20, ExponentialLearningRateDecay.defaultDecay);
		//衰减使用ExponentialLearningRateDecay.noDecay可以不进行lr的衰减，lr默认0.2可以修改
		//model.trainParallel(x, y, 400, 20, ExponentialLearningRateDecay.noDecay);
		ExponentialLearningRateDecay decay = 
				new ExponentialLearningRateDecay(2, 0.2, 0.8, false);
		model.trainParallel(x, y, 400, 20, decay);
		//以上参数迭代20个epoch后训练集acc=0.9860282
	}
	public static void loadData() {
		BufferedReader reader = null;
		try {
			reader = new BufferedReader(new FileReader("./data/minst_train.csv"));
			Stream<String> strea = reader.lines();
			int len = 34999;
			x = new double[len][];
			y = new double[len][];
			Iterator<String> lines = strea.iterator();
			int count = 0;
			while(lines.hasNext()) {
				String[] strs = lines.next().split(",");
				int yIndex = Integer.valueOf(strs[0]);
				double[] oney = new double[10];
				oney[yIndex] = 1.0;
				double[] onex = new double[strs.length];
				for(int i = 1; i < strs.length;i++) {
					onex[i] = Double.valueOf(strs[i]) / 255.0;
				}
				x[count] = onex;
				y[count] = oney;
				count++;
			}
		} catch (Exception e) {
			e.printStackTrace();
		} finally {
			try {
				reader.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
	}
}
```
#其他
1.遇到的问题：nn参数初始化
如果参数初始太大，而且没有负数，会造成溢出的情况，  
因为softmax作为输出层，里面存在着指数，输入太大则  
会溢出。建议初始化较小的参数，并且必须有正负，比如  
-0.25到0.25之间，关于参数初始化也有很大的学问，好的  
参数初始化可以使收敛更快。  

2.遇到的问题
//之前不work主要有两个原因：  
//（1）梯度最后一步计算的矩阵化公式有错  
//（2）损失函数没有除以N（我不知道为什么这个影响这么大，这直接导致了效果完全不行）  
//目前存在的问题  
//（1）效率不够高  
//（2）收敛不够快，前面设置的参数情况下，epoch 2才达到0.82左右的acc  
//3.23更新：  
//使用了学习率指数衰减，效果好多了  
//参数建议，如果epochs比较小，那么lr也要小一点  
3.关于并行  
之前使用call(),或者线程的join()来实现主线程等待子线程完成后继续  
发现效果都不好，比串行都慢，但使用countDownLatch这个对象后发现  
竟然速度快了2倍左右  
3.线程级别的计算基于单个样本还是多个样本？  
在我自己的实现中，单个线程就是单个样本的计算，因为线程池最大也就处理器个数+1，  
所以如果直接将样本划分为处理器个数份，来并行，可能效果会好一些，不用涉及到大量  
线程的管理。   
实验了下：差不多能快个1000毫秒  
4.并行对比  
//train_parallel:4m22s one epoch  
//train:7m07s  
//train_parallel 比 train快1.63倍  
使用并行预测大约快出2倍  

5.关于并行训练的实现  
将每个batch拆分成处理器个数这么多份，然后并行，每个线程保存了该线程计算的所有样本的梯度和  
最后将这些线程中的梯度加起来就可以求得这个batch的梯度  
不能单个样本作为一个线程，这样保存梯度会占用很大的内存  
不能并行去修改总的梯度变量，这样互斥使得并行效果不那么有效了  

6.使用方法  
借鉴了keras的设计，  
model = new Sequential()  
model.add(layer)  
.....  
model.train()  
