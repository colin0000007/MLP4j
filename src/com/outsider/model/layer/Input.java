package com.outsider.model.layer;
/**
 * 输入层
 * @author outsider
 *
 */
public class Input extends Layer{
	//输入向量维度
	private int inputDim;
	public Input(int inputDim) {
		this.inputDim = inputDim;
	}
	public int getInputDim() {
		return inputDim;
	}
	public void setInputDim(int inputDim) {
		this.inputDim = inputDim;
	}
}
