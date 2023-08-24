/*
 implementation of MVM by using balanced bit-slicing scheme
 for 2D convolution operation.
 inputs: 1D unrolled input matrix of signed integer values. shape: (filter_shape*input_size, 1) i.e. (9*49, 1)
 weights: 1D filter matrix of signed integer values: (num_channels*filter_shape, 1) i.e. (4*9, 1)
 data quantization is 8-bits.
*/
#include <vector>
#include <stdint.h>

namespace mvm_bitslicing
{
	
const int8_t DATA_RES = 8;
const int32_t XBAR_COLS = 32;
const int32_t XBAR_ROWS = 512;
const uint8_t BIAS = 128;
const uint8_t INPUT_MASK = 1;

class MVM
{
	public:
		MVM(int8_t filter_shape, int8_t num_channels, int8_t input_size);
		~MVM();
		void set_inputs(int8_t* inputs);
		void set_weights(int8_t* weights);
		void multiply();
		int32_t** get_mvm_result();
	private:
		int8_t filter_shape_;
		int8_t num_channels_;
		int8_t input_size_;
		std::vector<int8_t> inputs_;
		std::vector<int8_t> input_buf_;
		int8_t num_set_bits_;
		std::vector<int8_t> output_reg_;  // size: num_channels_ * DATA_RES i.e. 32
		std::vector<uint8_t> weights_;
		void set_weights_on_xbar_();  // dummy function that imitates comm with the xbar to set the weights
		std::vector<std::vector<uint8_t> > weights_xbar_;  // represents the xbar. its shape:(filter_shape_,num_channels_*DATA_RES) i.e. (9,32)
		void mvm_on_xbar_();  // dummy function that computes and stores the results in output_reg_ (for single bitline of input vector)
		void fill_input_buffer_(uint8_t input_vector_idx, uint8_t cycle);
		void update_running_sums_(int32_t* running_sums, uint8_t cycle);
		int32_t* multiply_single_vector_(uint8_t input_row_idx);  // input_row_idx: 0-48
		int32_t** mvm_result_;  // main result to be returned
};


}

