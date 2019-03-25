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
		//01�������
		DoubleMatrix result = matrix.mmul(vector);
		System.out.println(result.rows+"x"+result.columns+": "+result);
		//02��������Ԫ�ؼ�һ������
		matrix = matrix.add(0.5);
		System.out.println(matrix);
		FloatMatrix m1 = new FloatMatrix(new float[][] {{1,1},{1,1}});
		FloatMatrix m2 = new FloatMatrix(new float[][] {{2,2},{2,2}});
		//03 �����ӦԪ����� ��Ӧ����mul��ͷ
		FloatMatrix m3 = m1.mul(m2);
		System.out.println(m3.rows+","+m3.columns);
		System.out.println(m3);
		//04 ������� ��Ӧmmul������ͷ
		FloatMatrix m4 = m1.mmul(m2);
		System.out.println(m4);
		//05����ת��
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
	
	
	//����mmuli��mmul
	//�����еĵط������ô�i��׺�ģ�������󱻵������ߴ���߱�ת�ù�
	//i����ʲô��˼��Դ����Ӧ����ָin place���������mmul����Ҫ����new����
	@Test
	public void t2() {
		int rows = 100;
		int columns = 100;
		DoubleMatrix zs = DoubleMatrix.zeros(rows, columns);
		DoubleMatrix os = DoubleMatrix.ones(rows, columns);
		long s1 = System.currentTimeMillis();
		DoubleMatrix r1 = zs.mmul(os);
		long e1 = System.currentTimeMillis();
		System.out.println("��ʱ:"+(e1-s1)+"����!");
		//System.out.println(r1);
		//�������ַ���ȱʧҪ��Щ
		long s2 = System.currentTimeMillis();
		DoubleMatrix r2 = zs.mmuli(os);
		long e2 = System.currentTimeMillis();
		System.out.println("��ʱ:"+(e2-s2)+"����!");
		//System.out.println(r2);
		zs.transpose().mmuli(os);
	}
	
}
