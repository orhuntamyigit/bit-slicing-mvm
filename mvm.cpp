#include <iostream>
#include "mvm.h"
#include <cmath>

namespace mvm_bitslicing
{
	MVM::MVM(int8_t filter_shape, int8_t num_channels,
			int8_t input_size)
			:filter_shape_(filter_shape),
			num_channels_(num_channels),
			input_size_(input_size)
			{
				output_reg_.resize(DATA_RES*num_channels_);
				weights_xbar_.resize(filter_shape_, std::vector<uint8_t>(DATA_RES*num_channels_));
				
				mvm_result_ = new int32_t*[num_channels_];
				for (int i = 0; i<num_channels_; ++i)
					mvm_result_[i] = new int32_t[input_size_];
			}

	MVM::~MVM(){}

	void MVM::set_inputs(int8_t* inputs)
	{
		// transpose the unrolled input matrix and flatten. new shape: (49*9, 1)
		for(int i=0; i<input_size_; i++)
			for(int j=0; j<filter_shape_; j++)
				inputs_.push_back(inputs[j*input_size_ + i]);
	}

	void MVM::set_weights(int8_t* weights)
	{
		
		for(int i=0; i<num_channels_*filter_shape_; i++)
		{
			uint8_t weight_biased = weights[i] + BIAS;
			weights_.push_back(weight_biased);
		}
		set_weights_on_xbar_(); // dummy call to xbar to set the resistances that represent the weights
	}
	
	void MVM::set_weights_on_xbar_()
	{
		for(int ch=0; ch<num_channels_; ch++)
			for(int dig=0; dig<DATA_RES; dig++)
				for(int row_idx=0; row_idx<filter_shape_; row_idx++)
					weights_xbar_[row_idx][ch*DATA_RES + dig] = ((weights_[ch*filter_shape_ + row_idx] >> (DATA_RES - 1 - dig)) & INPUT_MASK);
		// ----------------------------------------------------
		// the variable 'weights_xbar_' must be sent to Core M4
		// ----------------------------------------------------
	}
	
	void MVM::fill_input_buffer_(uint8_t input_vector_idx, uint8_t cycle)
	{
		input_buf_.clear();
		for(int col_idx=0; col_idx<filter_shape_; col_idx++)
			input_buf_.push_back((inputs_.at(input_vector_idx*filter_shape_ + col_idx) >> cycle) & INPUT_MASK);
		// ----------------------------------------------------
		// the variable 'input_buf_' must be sent to Core M4
		// ----------------------------------------------------
		num_set_bits_ = 0;
		for(int i=0; i<filter_shape_; i++)
			num_set_bits_ += input_buf_[i];
	}
	
	void MVM::mvm_on_xbar_()
	{	
		// ----------------------------------------------------
		// this dummy function will be replaced later
		// ----------------------------------------------------
		for(int ch=0; ch<num_channels_; ch++)
			for(int dig=0; dig<DATA_RES; dig++)
				for(int row_idx=0; row_idx<filter_shape_; row_idx++)
					output_reg_[ch*DATA_RES + dig] += input_buf_[row_idx] * weights_xbar_[row_idx][ch*DATA_RES + dig];
	}
	
	void MVM::update_running_sums_(int32_t* running_sums, uint8_t cycle)
	{
		for(int ch=0; ch<num_channels_; ch++)
		{
			int32_t partial_sum = 0;
			for(int dig=DATA_RES; dig>0; dig--)
				partial_sum += (int32_t) (pow(2, DATA_RES-dig) * output_reg_[ch*DATA_RES + dig-1]);
			
			partial_sum -= (int32_t) (pow(2, DATA_RES-1) * num_set_bits_);
			
			if(cycle==DATA_RES - 1)
				running_sums[ch] -= (int32_t) partial_sum * pow(2, cycle);  // MSB is the sign bit
			else
				running_sums[ch] += (int32_t) partial_sum * pow(2, cycle);
		}
	}

	int32_t* MVM::multiply_single_vector_(uint8_t input_vector_idx)
	{
		int32_t* running_sums = new int32_t[num_channels_];
		for(int ch=0; ch<num_channels_; ch++)
			running_sums[ch] = 0;  // running sums per channel
		
		for(int cycle=0; cycle<DATA_RES; cycle++)
		{
			fill_input_buffer_(input_vector_idx, cycle);
			std::fill(output_reg_.begin(), output_reg_.end(), 0);
			mvm_on_xbar_();
			update_running_sums_(running_sums, cycle);
		}
		return running_sums;
	}

	void MVM::multiply()
	{
		for(int input_vector_idx=0; input_vector_idx<input_size_; input_vector_idx++)
		{
			int32_t* res = multiply_single_vector_(input_vector_idx);
			for(int ch=0; ch<num_channels_; ch++)
				mvm_result_[ch][input_vector_idx] = res[ch];
		}
	}
	
	int32_t** MVM::get_mvm_result()
	{
		return mvm_result_;
	}
}











