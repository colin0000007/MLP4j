package com.outsider.model.layer;
/**
 * �����
 * @author outsider
 *
 */
public class Input extends Layer{
	//��������ά��
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
