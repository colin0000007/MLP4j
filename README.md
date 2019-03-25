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
