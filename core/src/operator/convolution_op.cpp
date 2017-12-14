////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#include "operator/convolution_op.h"
#include <sstream>

namespace dlex_cnn
{
	template <typename Dtype>
	ConvolutionOp<Dtype>::ConvolutionOp()
	{
		op_type_ = "Convolution";
	}
	template <typename Dtype>
	ConvolutionOp<Dtype>::ConvolutionOp(ConvolutionOpParam param)
	{
		op_type_ = "Convolution";
		param_ = param;
	}
	template <typename Dtype>
	ConvolutionOp<Dtype>::~ConvolutionOp()
	{

	}
	template <typename Dtype>
	int ConvolutionOp<Dtype>::setOpParam(const std::string &op_param_str)
	{
		std::string opt_str = op_param_str;

		param_.blas_enable = atoi(fetchSubStr(opt_str, "blas_enable:", ",").c_str());
		param_.kernel_num = atoi(fetchSubStr(opt_str, "kernel_num:", ",").c_str());
		param_.kernel_h = atoi(fetchSubStr(opt_str, "kernel_h:", ",").c_str());
		param_.kernel_w = atoi(fetchSubStr(opt_str, "kernel_w:", ",").c_str());
		param_.stride_h = atoi(fetchSubStr(opt_str, "stride_h:", ",").c_str());
		param_.stride_w = atoi(fetchSubStr(opt_str, "stride_w:", ",").c_str());
		param_.pad_w = atoi(fetchSubStr(opt_str, "pad_w:", ",").c_str());
		param_.dilation_h = atoi(fetchSubStr(opt_str, "dilation_h:", ",").c_str());
		param_.dilation_w = atoi(fetchSubStr(opt_str, "dilation_w:", ",").c_str());

		return 0;
	}
	template <typename Dtype>
	std::string ConvolutionOp<Dtype>::genOpParamStr() const
	{
		std::stringstream param_str;
		param_str << "blas_enable:" << param_.blas_enable << ",kernel_num:" << param_.kernel_num
			<< ",kernel_h:" << param_.kernel_h << ",kernel_w:" << param_.kernel_w
			<< ",stride_h:" << param_.stride_h << ",stride_w:" << param_.stride_w
			<< ",pad_h:" << param_.pad_h << ",pad_w:" << param_.pad_w
			<< ",dilation_h:" << param_.dilation_h << ",dilation_w:" << param_.dilation_w << ",";
		return param_str.str();
	}
	template <typename Dtype>
	int ConvolutionOp<Dtype>::inferOutShape(std::vector<int> &in_shape, std::vector<int> &out_shape)
	{
		out_shape.clear();

		out_shape.push_back(in_shape[tind::eNum]);
		out_shape.push_back(param_.kernel_num);

		out_shape.push_back((in_shape[tind::eHeight] + 2 * param_.pad_h - 
			(param_.dilation_h * (param_.kernel_h - 1) + 1)) / param_.stride_h + 1);
		out_shape.push_back((in_shape[tind::eWidth] + 2 * param_.pad_w - 
			(param_.dilation_w * (param_.kernel_w - 1) + 1)) / param_.stride_w + 1);

		return 0;
	}
	template <typename Dtype>
	int ConvolutionOp<Dtype>::allocBuf4Node(const std::vector<int> &in_shape,
		const std::vector<int> &out_shape,
		std::vector<std::shared_ptr<Tensor<Dtype>>> &data) const
	{
		data.clear();
		//printf("data and gradient: size() : %d, %d\n", data.size(), gradient_.size());
		data.push_back(std::make_shared<Tensor<Dtype>>(in_shape));

		//weight (kernels for convolution)
		data.push_back(std::make_shared<Tensor<Dtype>>(
			param_.kernel_num, 
			in_shape[tind::eChannels],
			param_.kernel_w, 
			param_.kernel_h));

		//blas
		if (param_.blas_enable)
			data.push_back(std::make_shared<Tensor<Dtype>>(out_shape[tind::eChannels], 1, 1, 1));

		return 0;
	}
	template <typename Dtype>
	int ConvolutionOp<Dtype>::allocOpBuf4Train(const std::vector<int> &in_shape, const std::vector<int> &out_shape)
	{
		if (in_shape[tind::eNum] <= 0 || in_shape[tind::eChannels] <= 0 ||
			in_shape[tind::eHeight] <= 0 || in_shape[tind::eWidth] <= 0 ||
			in_shape[tind::eNum] > 5000 || in_shape[tind::eChannels] > 5000 ||
			in_shape[tind::eHeight] > 5000 || in_shape[tind::eWidth] > 5000)
		{
			DLOG_ERR("[ InnerProductOp::allocOpBuf4Train ]: in_shape is invalid -> (%d, %d, %d, %d) \n",
				in_shape[tind::eNum], in_shape[tind::eChannels], in_shape[tind::eHeight], in_shape[tind::eWidth]);
			return -1;
		}

		diff_.clear();
		diff_.push_back(std::make_shared<Tensor<Dtype>>(in_shape));
	
		gradient_.clear();
		gradient_.push_back(std::make_shared<Tensor<Dtype>>(
			param_.kernel_num, 
			in_shape[tind::eChannels], 
			param_.kernel_w, 
			param_.kernel_h));

		if (param_.blas_enable)
			gradient_.push_back(std::make_shared<Tensor<Dtype>>(out_shape[tind::eChannels], 1, 1, 1));

		return 0;
	}

	template <typename Dtype>
	void ConvolutionOp<Dtype>::forward(
		const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev, 
		const std::vector<std::shared_ptr<Tensor<Dtype>>> &next)
	{
		const std::vector<int> prev_shape = prev[0]->getShape();
		const std::vector<int> next_shape = next[0]->getShape();

		const std::vector<int> prev_size = prev[0]->getSize();
		const std::vector<int> next_size = next[0]->getSize();

		const std::vector<int> kernel_shape = prev[1]->getShape();

		/////////////////////////////////////////////////////
		//auto worker = [&](const size_t start, const size_t stop){
		//	convolution2d(prev_data + start*prev_size[tind::e3D], kernel_data, bias_data, next_data + start*next_size[tind::e3D],
		//		stop - start, prev_shape[tind::eChannels], prev_shape[tind::eWidth], prev_shape[tind::eHeight],
		//		param_.kernel_num, param_.kernel_w, param_.kernel_h, param_.stride_w, param_.stride_h,
		//		next_shape[tind::eWidth], next_shape[tind::eHeight], (int)param_.pad_type);
		//};
		//worker(0, prev_shape[tind::eNum]);
		/////////////////////////////////////////////////////

		const Dtype* prev_data = (Dtype *)prev[0]->getPushCpuData();
		const Dtype* kernel_data = (Dtype *)prev[1]->getPushCpuData();
		const Dtype* bias_data = (Dtype *)prev[2]->getPushCpuData();
		Dtype* next_data = (Dtype *)next[0]->getPushCpuData();

		// (1, channels*kernel_h*kernel_w, output_h*output_w)
		const int output_h = (prev_shape[tind::eHeight] + 2 * param_.pad_h - 
			(param_.dilation_h * (param_.kernel_h - 1) + 1)) / param_.stride_h + 1;
		const int output_w = (prev_shape[tind::eWidth] + 2 * param_.pad_w - 
			(param_.dilation_w * (param_.kernel_w - 1) + 1)) / param_.stride_w + 1;

		// The dimension of col_buffer is relevent to "prev". -> From prev to col_buffer.
		// prev channel num is equal to kernel's channel num.
		int col_height = prev_shape[tind::eChannels] * param_.kernel_h * param_.kernel_w;
		int col_width = output_h * output_w;
		if (col_buffer_ == NULL)
			col_buffer_ = std::make_shared<Tensor<Dtype>>(1, 1, col_height, col_width);
		else if (col_buffer_->getSize()[tind::e4D] != 1 * 1 * col_height * col_width)
			col_buffer_.reset(new Tensor<Dtype>(1, 1, col_height, col_width));

		Dtype* col_data = (Dtype *)col_buffer_->getPushCpuData();

		next[0]->setCpuZero();
		for (int ni = 0; ni < prev_shape[tind::eNum]; ni++)
		{
			//printf("address: %d\n", col_data);
			im2col_cpu<Dtype>(prev_data + ni*prev_size[tind::e3D], prev_shape[tind::eChannels],
				prev_shape[tind::eHeight], prev_shape[tind::eWidth], 
				param_.kernel_h, param_.kernel_w,
				param_.pad_h, param_.pad_w,
				param_.stride_h, param_.stride_w,
				param_.dilation_h, param_.dilation_w,
				col_data);
			//printf("address: %d\n", col_data);

			//bool bTransA, bool bTransB, const int M, const int N, const int K, const float alpha, const Dtype* A, const Dtype* B, const float beta, Dtype* C
			gemm_cpu(false, false, param_.kernel_num, 
				col_width, col_height, 
				(Dtype)1, kernel_data, col_data, 
				(Dtype)0, next_data + ni * next_size[tind::e3D]);
		}

		// kernel_num is equal to output channel number, one kernel corresponds to one bias.
		// So take channels to be the index to select bias, and use the same bias in a channel.
		if (param_.blas_enable)
			add_bias(
			next_shape[tind::eNum], 
			next_shape[tind::eChannels], 
			next_size[tind::e2D], 
			bias_data, next_data);
	}

	template <typename Dtype>
	void ConvolutionOp<Dtype>::backward(
		const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev, 
		const std::vector<std::shared_ptr<Tensor<Dtype>>> &next,
		const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev_diff, 
		const std::vector<std::shared_ptr<Tensor<Dtype>>> &next_diff)
	{
		// data
		const std::vector<int> prev_shape = prev[0]->getShape();
		const std::vector<int> next_shape = next[0]->getShape();

		const std::vector<int> prev_size = prev[0]->getSize();
		const std::vector<int> next_size = next[0]->getSize();

		// diff
		const std::vector<int> prev_diff_shape = prev_diff[0]->getShape();
		const std::vector<int> next_diff_shape = next_diff[0]->getShape();

		const std::vector<int> prev_diff_size = prev_diff[0]->getSize();
		const std::vector<int> next_diff_size = next_diff[0]->getSize();

		// weight
		const std::vector<int> kernel_shape = prev[1]->getShape();
		const std::vector<int> kernel_size = prev[1]->getSize();

		// bias
		//const std::vector<int> biasShape = prev[2]->getShape();
		const std::vector<int> bias_size = prev[2]->getSize();

		const Dtype* prev_data = (Dtype*)prev[0]->getPushCpuData();
		const Dtype* next_data = (Dtype*)next[0]->getPushCpuData();
		Dtype* prev_diff_data = (Dtype*)prev_diff[0]->getPushCpuData();
		Dtype* next_diff_data = (Dtype*)next_diff[0]->getPushCpuData();
		Dtype *kernel_data = (Dtype*)prev[1]->getPushCpuData();
		//Dtype *bias_data = (Dtype*)prev[2]->getPushCpuData();

		Dtype* col_data = (Dtype *)col_buffer_->getPushCpuData();

		
		//update prev_diff
		prev_diff[0]->setCpuZero();
		for (int i = 0; i < prev_diff_shape[tind::eNum]; i++)
		{
			gemm_cpu(true, false, kernel_size[tind::e3D],
				next_diff_size[tind::e2D], kernel_shape[tind::eNum],
				(Dtype)1.0, kernel_data, next_diff_data + i * next_diff_size[tind::e3D],
				(Dtype)0.0, col_data);

			col2im_cpu(col_data, prev_diff_shape[tind::eChannels],
				prev_diff_shape[tind::eHeight], prev_diff_shape[tind::eWidth],
				param_.kernel_h, param_.kernel_w,
				param_.pad_h, param_.pad_w,
				param_.stride_h, param_.stride_w,
				param_.dilation_h, param_.dilation_w,
				prev_diff_data + i * prev_diff_size[tind::e3D]);
		}

		//update weight Diff
		gradient_[0]->setCpuZero();
		//const std::vector<int> kernelGradientSize = gradient_[0]->getSize();
		Dtype* kernel_gradient_data = (Dtype *)gradient_[0]->getPushCpuData();

		for (int ni = 0; ni < prev_diff_shape[tind::eNum]; ni++)
		{
			im2col_cpu<Dtype>(prev_data + ni*prev_size[tind::e3D], prev_shape[tind::eChannels],
				prev_shape[tind::eHeight], prev_shape[tind::eWidth],
				param_.kernel_h, param_.kernel_w,
				param_.pad_h, param_.pad_w,
				param_.stride_h, param_.stride_w,
				param_.dilation_h, param_.dilation_w,
				col_data);

			// kernel_shape[tind::eNum] == next_shape[tind::eChannels]
			gemm_cpu(false, true, kernel_shape[tind::eNum], 
				kernel_size[tind::e3D], next_size[tind::e2D],
				(Dtype)1.0, next_diff_data + ni * next_diff_size[tind::e3D], col_data,
				(Dtype)1.0, kernel_gradient_data);

		}
		div_inplace_cpu(kernel_size[tind::e4D], (Dtype)next_shape[tind::eNum], kernel_gradient_data);

		//update bias gradient
		gradient_[1]->setCpuZero();
		Dtype* bias_gradient_data = (Dtype *)gradient_[1]->getPushCpuData();

		backward_bias(
			next_diff_shape[tind::eNum], 
			next_diff_shape[tind::eChannels], 
			next_diff_size[tind::e2D], 
			next_diff_data, bias_gradient_data);
		div_inplace_cpu(bias_size[tind::e4D], (Dtype)next_shape[tind::eNum], bias_gradient_data);

	}

#ifdef USE_CUDA
	template <typename Dtype>
	void ConvolutionOp<Dtype>::forward_gpu(
		const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev, 
		const std::vector<std::shared_ptr<Tensor<Dtype>>> &next)
	{
		const std::vector<int> prev_shape = prev[0]->getShape();
		const std::vector<int> next_shape = next[0]->getShape();

		const std::vector<int> prev_size = prev[0]->getSize();
		const std::vector<int> next_size = next[0]->getSize();

		const std::vector<int> kernel_shape = prev[1]->getShape();

		const Dtype* prev_data = (Dtype *)prev[0]->getPushGpuData();
		const Dtype* kernel_data = (Dtype *)prev[1]->getPushGpuData();
		const Dtype* bias_data = (Dtype *)prev[2]->getPushGpuData();
		Dtype* next_data = (Dtype *)next[0]->getPushGpuData();

		// (1, channels*kernel_h*kernel_w, output_h*output_w)
		const int output_h = (prev_shape[tind::eHeight] + 2 * param_.pad_h - 
			(param_.dilation_h * (param_.kernel_h - 1) + 1)) / param_.stride_h + 1;
		const int output_w = (prev_shape[tind::eWidth] + 2 * param_.pad_w - 
			(param_.dilation_w * (param_.kernel_w - 1) + 1)) / param_.stride_w + 1;

		// The dimension of col_buffer is relevent to "prev". -> From prev to col_buffer.
		// prev channel num is equal to kernel's channel num.
		int col_height = prev_shape[tind::eChannels] * param_.kernel_h * param_.kernel_w;
		int col_width = output_h * output_w;
		if (col_buffer_ == NULL)
			col_buffer_ = std::make_shared<Tensor<Dtype>>(1, 1, col_height, col_width);
		else if (col_buffer_->getSize()[tind::e4D] != 1 * 1 * col_height * col_width)
			col_buffer_.reset(new Tensor<Dtype>(1, 1, col_height, col_width));

		Dtype* col_data = (Dtype *)col_buffer_->getPushGpuData();

		next[0]->setGpuZero();
		for (int ni = 0; ni < prev_shape[tind::eNum]; ni++)
		{
			//printf("address: %d\n", col_data);
			im2col_gpu<Dtype>(prev_data + ni*prev_size[tind::e3D], prev_shape[tind::eChannels],
				prev_shape[tind::eHeight], prev_shape[tind::eWidth],
				param_.kernel_h, param_.kernel_w,
				param_.pad_h, param_.pad_w,
				param_.stride_h, param_.stride_w,
				param_.dilation_h, param_.dilation_w,
				col_data);

			gemm_gpu(CuHandleManager::cublas_handle(), false, false, param_.kernel_num, 
				col_width, col_height,
				(Dtype)1, kernel_data, col_data,
				(Dtype)0, next_data + ni * next_size[tind::e3D]);
		}

		//if (param_.blas_enable)
		//	add_bias(next_shape[tind::eNum], next_shape[tind::eChannels], next_size[tind::e2D], bias_data, next_data);
	}

	template <typename Dtype>
	void ConvolutionOp<Dtype>::backward_gpu(
		const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev,
		const std::vector<std::shared_ptr<Tensor<Dtype>>> &next,
		const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev_diff,
		const std::vector<std::shared_ptr<Tensor<Dtype>>> &next_diff)
	{
		// data
		const std::vector<int> prev_shape = prev[0]->getShape();
		const std::vector<int> next_shape = next[0]->getShape();

		const std::vector<int> prev_size = prev[0]->getSize();
		const std::vector<int> next_size = next[0]->getSize();

		// diff
		const std::vector<int> prev_diff_shape = prev_diff[0]->getShape();
		const std::vector<int> next_diff_shape = next_diff[0]->getShape();

		const std::vector<int> prev_diff_size = prev_diff[0]->getSize();
		const std::vector<int> next_diff_size = next_diff[0]->getSize();

		// weight
		const std::vector<int> kernel_shape = prev[1]->getShape();
		const std::vector<int> kernel_size = prev[1]->getSize();

		// bias
		//const std::vector<int> biasShape = prev[2]->getShape();
		const std::vector<int> bias_size = prev[2]->getSize();

		const Dtype* prev_data = (Dtype*)prev[0]->getPushGpuData();
		const Dtype* next_data = (Dtype*)next[0]->getPushGpuData();
		Dtype* prev_diff_data = (Dtype*)prev_diff[0]->getPushGpuData();
		Dtype* next_diff_data = (Dtype*)next_diff[0]->getPushGpuData();
		Dtype *kernel_data = (Dtype*)prev[1]->getPushGpuData();
		//Dtype *bias_data = (Dtype*)prev[2]->getPushGpuData();

		Dtype* col_data = (Dtype *)col_buffer_->getPushGpuData();


		//update prev_diff
		prev_diff[0]->setGpuZero();
		for (int i = 0; i < prev_diff_shape[tind::eNum]; i++)
		{
			gemm_gpu(CuHandleManager::cublas_handle(), true, false, kernel_size[tind::e3D], 
				next_diff_size[tind::e2D], kernel_shape[tind::eNum],
				(Dtype)1.0, kernel_data, next_diff_data + i * next_diff_size[tind::e3D],
				(Dtype)0.0, col_data);

			col2im_gpu(col_data, prev_diff_shape[tind::eChannels],
				prev_diff_shape[tind::eHeight], prev_diff_shape[tind::eWidth],
				param_.kernel_h, param_.kernel_w,
				param_.pad_h, param_.pad_w,
				param_.stride_h, param_.stride_w,
				param_.dilation_h, param_.dilation_w,
				prev_diff_data + i * prev_diff_size[tind::e3D]);
		}

		//update weight Diff
		gradient_[0]->setGpuZero();
		//const std::vector<int> kernelGradientSize = gradient_[0]->getSize();
		Dtype* kernel_gradient_data = (Dtype *)gradient_[0]->getPushGpuData();

		for (int ni = 0; ni < prev_diff_shape[tind::eNum]; ni++)
		{
			im2col_gpu<Dtype>(prev_data + ni*prev_size[tind::e3D], prev_shape[tind::eChannels],
				prev_shape[tind::eHeight], prev_shape[tind::eWidth],
				param_.kernel_h, param_.kernel_w,
				param_.pad_h, param_.pad_w,
				param_.stride_h, param_.stride_w,
				param_.dilation_h, param_.dilation_w,
				col_data);

			// kernel_shape[tind::eNum] == next_shape[tind::eChannels]
			gemm_gpu(CuHandleManager::cublas_handle(), false, true, kernel_shape[tind::eNum],
				kernel_size[tind::e3D], next_size[tind::e2D],
				(Dtype)1.0, next_diff_data + ni * next_diff_size[tind::e3D], col_data,
				(Dtype)1.0, kernel_gradient_data);

		}
		div_inplace_gpu(kernel_size[tind::e4D], (Dtype)next_shape[tind::eNum], kernel_gradient_data);

		////update bias gradient
		//gradient_[1]->setGpuZero();
		//Dtype* bias_gradient_data = (Dtype *)gradient_[1]->getPushGpuData();

		//backward_bias(next_diff_shape[tind::eNum], next_diff_shape[tind::eChannels], next_diff_size[tind::e2D], next_diff_data, bias_gradient_data);
		//div_inplace_gpu(bias_size[tind::e4D], (Dtype)next_shape[tind::eNum], bias_gradient_data);

	}
#endif
	INSTANTIATE_CLASS(ConvolutionOp);

}//namespace