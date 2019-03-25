package com.outsider.model;

import javax.swing.JFrame;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartFrame;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

/**
 * ��ѧϰ��˥������
 * ��Ҫ������jfree
 * �������Ҫֱ��ɾ���������
 * @author outsider
 *
 */
public class DrawLearningRateDecyCurve {
	/**
	 * 
	 * @param decay lr˥���������ö���
	 * @param iters ��������
	 */
	public static void draw(ExponentialLearningRateDecay decay, int iters) {
		XYSeries series = new XYSeries("xySeries");
		for(int i = 0; i < iters; i++) {
			series.add(i,decay.decayLR(i));
		}
		XYSeriesCollection dataset = new XYSeriesCollection();
		dataset.addSeries(series);
		JFreeChart chart = ChartFactory.createXYLineChart(
				"LR decay curve", // chart title
				"iters", // x axis label
				"learning rate", // y axis label
				dataset, // data
				PlotOrientation.VERTICAL,
				false, // include legend
				false, // tooltips
				false // urls
				);
 
		ChartFrame frame = new ChartFrame("my picture", chart);
		frame.pack();
		frame.setVisible(true);
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
	}
	
	//����
	public static void main(String[] args) {
		ExponentialLearningRateDecay decay = 
				new ExponentialLearningRateDecay(2, 0.2, 0.8, false);
		DrawLearningRateDecyCurve.draw(decay, 20);
	}
}
