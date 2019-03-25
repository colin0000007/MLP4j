package com.outisder.jblasExamples;

import org.jblas.DoubleMatrix;
import org.jblas.FloatMatrix;
import org.junit.Test;
/**
 * 
 * @author outsider
 *
 */
public class Example {
	public static void main(String[] args)
	{
		
		double[][] data = new double[][]
	                   {{ 1,  2,  3,   4,   5},
	                    { 6,  7,  8,   9,  10},
	                    {11, 12, 13, 14, 15}};
		DoubleMatrix matrix = new DoubleMatrix(data);
		DoubleMatrix vector = new DoubleMatrix(new double[]{3, 3, 3, 3,3});
		//01矩阵相乘
		DoubleMatrix result = matrix.mmul(vector);
		System.out.println(result.rows+"x"+result.columns+": "+result);
		//02矩阵所有元素加一个常数
		matrix = matrix.add(0.5);
		System.out.println(matrix);
		FloatMatrix m1 = new FloatMatrix(new float[][] {{1,1},{1,1}});
		FloatMatrix m2 = new FloatMatrix(new float[][] {{2,2},{2,2}});
		//03 矩阵对应元素相乘 对应方法mul开头
		FloatMatrix m3 = m1.mul(m2);
		System.out.println(m3.rows+","+m3.columns);
		System.out.println(m3);
		//04 矩阵相乘 对应mmul方法开头
		FloatMatrix m4 = m1.mmul(m2);
		System.out.println(m4);
		//05矩阵转置
		FloatMatrix m5 = new FloatMatrix(new float[][] {{1,2},{3,4}});
		FloatMatrix m6 = m5.transpose();
		System.out.println(m6);
		System.out.println(m1.mmuli(m2));
	}
	
	@Test
	public void t1() {
		DoubleMatrix m1 = new DoubleMatrix(new double[] {1,1});
		DoubleMatrix m2 = new DoubleMatrix(new double[] {2,2});
		System.out.println(m1.add(m2));
	}
	
	
	//测试mmuli和mmul
	//发现有的地方不能用带i后缀的，如果矩阵被调整过尺寸或者被转置过
	//i代表什么意思？源码中应该是指in place，如果调用mmul还需要重新new对象。
	@Test
	public void t2() {
		int rows = 100;
		int columns = 100;
		DoubleMatrix zs = DoubleMatrix.zeros(rows, columns);
		DoubleMatrix os = DoubleMatrix.ones(rows, columns);
		long s1 = System.currentTimeMillis();
		DoubleMatrix r1 = zs.mmul(os);
		long e1 = System.currentTimeMillis();
		System.out.println("耗时:"+(e1-s1)+"毫秒!");
		//System.out.println(r1);
		//下面这种方法缺失要快些
		long s2 = System.currentTimeMillis();
		DoubleMatrix r2 = zs.mmuli(os);
		long e2 = System.currentTimeMillis();
		System.out.println("耗时:"+(e2-s2)+"毫秒!");
		//System.out.println(r2);
		zs.transpose().mmuli(os);
	}
	
}
