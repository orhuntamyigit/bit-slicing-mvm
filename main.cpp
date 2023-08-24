#include <iostream>
#include <ctime>
#include <cstdlib>
#include "mvm.h"
#include "test.h"

int main(int argc, char** argv) {
		
	int8_t kernel_size = 3;  // 3 by 3 convolution kernel
	int8_t conv_channels = 4;  // 4 conv channels
	int8_t window_side_len = 7;  // 7 by 7 input window
	int8_t filter_shape = kernel_size*kernel_size;
	int8_t input_size = window_side_len*window_side_len;
	mvm_bitslicing::MVM mvm(filter_shape, conv_channels, input_size);
	
	int8_t* inputs = new int8_t[filter_shape*input_size];
	int range = 127 - (-128) + 1;

	for(int i=0; i<filter_shape*input_size; i++)
	{
		inputs[i] = get_inputs[i];
	}
	
	int8_t* weights = new int8_t[conv_channels*filter_shape];
	
	for(int i=0; i<conv_channels*filter_shape; i++)
	{
		weights[i] = get_weights[i];
	}
	
	mvm.set_inputs(inputs);
	mvm.set_weights(weights);
	mvm.multiply();
	int32_t** result = mvm.get_mvm_result();
	
	std::cout << "\nresult:\n";
	for (int ch=0; ch<conv_channels; ch++)
	{
		for (int i=0; i<input_size; i++)
			std::cout << (int) result[ch][i] << " ";
		std::cout << "\n";
	}
	
	return 0;
}
