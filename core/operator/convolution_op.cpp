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
	int ConvolutionOp<Dtype>::setOpParam(const std::string &opParamStr)
	{
		std::string optStr = opParamStr;

		param_.blas_enable = atoi(fetchSubStr(optStr, "blas_enable:", ",").c_str());
		param_.kernel_num = atoi(fetchSubStr(optStr, "kernel_num:", ",").c_str());
		param_.kernel_h = atoi(fetchSubStr(optStr, "kernel_h:", ",").c_str());
		param_.kernel_w = atoi(fetchSubStr(optStr, "kernel_w:", ",").c_str());
		param_.stride_h = atoi(fetchSubStr(optStr, "stride_h:", ",").c_str());
		param_.stride_w = atoi(fetchSubStr(optStr, "stride_w:", ",").c_str());
		param_.pad_w = atoi(fetchSubStr(optStr, "pad_w:", ",").c_str());
		param_.dilation_h = atoi(fetchSubStr(optStr, "dilation_h:", ",").c_str());
		param_.dilation_w = atoi(fetchSubStr(optStr, "dilation_w:", ",").c_str());

		return 0;
	}
	template <typename Dtype>
	std::string ConvolutionOp<Dtype>::genOpParamStr() const
	{
		std::stringstream paramStr;
		paramStr << "blas_enable:" << param_.blas_enable << ",kernel_num:" << param_.kernel_num
			<< ",kernel_h:" << param_.kernel_h << ",kernel_w:" << param_.kernel_w
			<< ",stride_h:" << param_.stride_h << ",stride_w:" << param_.stride_w
			<< ",pad_h:" << param_.pad_h << ",pad_w:" << param_.pad_w
			<< ",dilation_h:" << param_.dilation_h << ",dilation_w:" << param_.dilation_w << ",";
		return paramStr.str();
	}
	template <typename Dtype>
	int ConvolutionOp<Dtype>::inferOutShape(std::vector<int> &inShape, std::vector<int> &outShape)
	{
		outShape.clear();

		outShape.push_back(inShape[tind::eNum]);
		outShape.push_back(param_.kernel_num);

		outShape.push_back((inShape[tind::eHeight] + 2 * param_.pad_h - (param_.dilation_h * (param_.kernel_h - 1) + 1)) / param_.stride_h + 1);
		outShape.push_back((inShape[tind::eWidth] + 2 * param_.pad_w - (param_.dilation_w * (param_.kernel_w - 1) + 1)) / param_.stride_w + 1);

		return 0;
	}
	template <typename Dtype>
	int ConvolutionOp<Dtype>::allocBuf4Node(const std::vector<int> &inShape,
		const std::vector<int> &outShape,
		std::vector<std::shared_ptr<Tensor<Dtype>>> &data) const
	{
		data.clear();
		//printf("data and gradient: size() : %d, %d\n", data.size(), gradient_.size());
		data.push_back(std::make_shared<Tensor<Dtype>>(inShape));

		//weight (kernels for convolution)
		data.push_back(std::make_shared<Tensor<Dtype>>(param_.kernel_num, inShape[tind::eChannels], param_.kernel_w, param_.kernel_h));
		normal_distribution_init<Dtype>((Dtype *)data[1]->getData(), data[1]->getSize()[tind::e4D], 0.0f, 0.1f);

		//blas
		if (param_.blas_enable)
		{
			data.push_back(std::make_shared<Tensor<Dtype>>(outShape[tind::eChannels], 1, 1, 1));
			dlex_set<Dtype>((Dtype *)data[2]->getData(), data[2]->getSize()[tind::e4D], 0.0f);
		}
		return 0;
	}
	template <typename Dtype>
	int ConvolutionOp<Dtype>::allocOpBuf4Train(const std::vector<int> &inShape, const std::vector<int> &outShape)
	{
		if (inShape[tind::eNum] <= 0 || inShape[tind::eChannels] <= 0 ||
			inShape[tind::eHeight] <= 0 || inShape[tind::eWidth] <= 0 ||
			inShape[tind::eNum] > 5000 || inShape[tind::eChannels] > 5000 ||
			inShape[tind::eHeight] > 5000 || inShape[tind::eWidth] > 5000)
		{
			DLOG_ERR("[ InnerProductOp::allocOpBuf4Train ]: inShape is invalid -> (%d, %d, %d, %d) \n",
				inShape[tind::eNum], inShape[tind::eChannels], inShape[tind::eHeight], inShape[tind::eWidth]);
			return -1;
		}

		diff_.clear();
		diff_.push_back(std::make_shared<Tensor<Dtype>>(inShape));
	
		gradient_.clear();
		gradient_.push_back(std::make_shared<Tensor<Dtype>>(param_.kernel_num, inShape[tind::eChannels], param_.kernel_w, param_.kernel_h));
		dlex_set<Dtype>((Dtype *)gradient_[0]->getData(), gradient_[0]->getSize()[tind::e4D], 0.0f);

		if (param_.blas_enable)
		{
			gradient_.push_back(std::make_shared<Tensor<Dtype>>(outShape[tind::eChannels], 1, 1, 1));
			dlex_set<Dtype>((Dtype *)gradient_[1]->getData(), gradient_[1]->getSize()[tind::e4D], 0.0f);
		}
		return 0;
	}

	template <typename Dtype>
	void ConvolutionOp<Dtype>::forward(const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev, const std::vector<std::shared_ptr<Tensor<Dtype>>> &next)
	{
		const std::vector<int> prevShape = prev[0]->getShape();
		const std::vector<int> nextShape = next[0]->getShape();

		const std::vector<int> prevSize = prev[0]->getSize();
		const std::vector<int> nextSize = next[0]->getSize();

		const std::vector<int> kernelShape = prev[1]->getShape();

		/////////////////////////////////////////////////////
		//auto worker = [&](const size_t start, const size_t stop){
		//	convolution2d(prevData + start*prevSize[tind::e3D], kernelData, biasData, nextData + start*nextSize[tind::e3D],
		//		stop - start, prevShape[tind::eChannels], prevShape[tind::eWidth], prevShape[tind::eHeight],
		//		param_.kernel_num, param_.kernel_w, param_.kernel_h, param_.stride_w, param_.stride_h,
		//		nextShape[tind::eWidth], nextShape[tind::eHeight], (int)param_.pad_type);
		//};
		//worker(0, prevShape[tind::eNum]);
		/////////////////////////////////////////////////////

		const Dtype* prevData = (Dtype *)prev[0]->getData();
		const Dtype* kernelData = (Dtype *)prev[1]->getData();
		const Dtype* biasData = (Dtype *)prev[2]->getData();
		Dtype* nextData = (Dtype *)next[0]->getData();

		// (1, channels*kernel_h*kernel_w, output_h*output_w)
		const int output_h = (prevShape[tind::eHeight] + 2 * param_.pad_h - (param_.dilation_h * (param_.kernel_h - 1) + 1)) / param_.stride_h + 1;
		const int output_w = (prevShape[tind::eWidth] + 2 * param_.pad_w - (param_.dilation_w * (param_.kernel_w - 1) + 1)) / param_.stride_w + 1;

		// The dimension of col_buffer is relevent to "prev". -> From prev to col_buffer.
		// prev channel num is equal to kernel's channel num.
		int colHeight = prevShape[tind::eChannels] * param_.kernel_h * param_.kernel_w;
		int colWidth = output_h * output_w;
		if (col_buffer_ == NULL)
			col_buffer_ = std::make_shared<Tensor<Dtype>>(1, 1, colHeight, colWidth);
		else if (col_buffer_->getSize()[tind::e4D] != 1 * 1 * colHeight * colWidth)
			col_buffer_.reset(new Tensor<Dtype>(1, 1, colHeight, colWidth));

		Dtype* colData = (Dtype *)col_buffer_->getData();

		for (int ni = 0; ni < prevShape[tind::eNum]; ni++)
		{
			//printf("address: %d\n", colData);
			im2col_cpu<Dtype>(prevData + ni*prevSize[tind::e3D], prevShape[tind::eChannels],
				prevShape[tind::eHeight], prevShape[tind::eWidth], 
				param_.kernel_h, param_.kernel_w,
				param_.pad_h, param_.pad_w,
				param_.stride_h, param_.stride_w,
				param_.dilation_h, param_.dilation_w,
				colData);
			//printf("address: %d\n", colData);

			//bool bTransA, bool bTransB, const int M, const int N, const int K, const float alpha, const Dtype* A, const Dtype* B, const float beta, Dtype* C
			gemm(false, false, param_.kernel_num, colWidth, colHeight, 1, kernelData, colData, 0, nextData + ni * nextSize[tind::e3D]);
		}

		// kernel_num与输出channels一致，一个kernel对应一个bias，则以channels为下标，channel内使用同一个bias // chinese
		if (param_.blas_enable)
			add_bias(nextShape[tind::eNum], nextShape[tind::eChannels], nextSize[tind::e2D], biasData, nextData);
	}

	template <typename Dtype>
	void ConvolutionOp<Dtype>::backward(const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev, const std::vector<std::shared_ptr<Tensor<Dtype>>> &next,
		const std::vector<std::shared_ptr<Tensor<Dtype>>> &prevDiff, const std::vector<std::shared_ptr<Tensor<Dtype>>> &nextDiff)
	{
		// data
		const std::vector<int> prevShape = prev[0]->getShape();
		const std::vector<int> nextShape = next[0]->getShape();

		const std::vector<int> prevSize = prev[0]->getSize();
		const std::vector<int> nextSize = next[0]->getSize();

		// diff
		const std::vector<int> prevDiffShape = prevDiff[0]->getShape();
		const std::vector<int> nextDiffShape = nextDiff[0]->getShape();

		const std::vector<int> prevDiffSize = prevDiff[0]->getSize();
		const std::vector<int> nextDiffSize = nextDiff[0]->getSize();

		// weight
		const std::vector<int> kernelShape = prev[1]->getShape();
		const std::vector<int> kernelSize = prev[1]->getSize();

		// bias
		//const std::vector<int> biasShape = prev[2]->getShape();
		const std::vector<int> biasSize = prev[2]->getSize();

		const Dtype* prevData = (Dtype*)prev[0]->getData();
		const Dtype* nextData = (Dtype*)next[0]->getData();
		Dtype* prevDiffData = (Dtype*)prevDiff[0]->getData();
		Dtype* nextDiffData = (Dtype*)nextDiff[0]->getData();
		Dtype *kernelData = (Dtype*)prev[1]->getData();
		//Dtype *biasData = (Dtype*)prev[2]->getData();

		Dtype* colData = (Dtype *)col_buffer_->getData();

		
		//update prevDiff
		prevDiff[0]->setZero();
		for (int i = 0; i < prevDiffShape[tind::eNum]; i++)
		{
			gemm(true, false, kernelSize[tind::e3D], nextDiffSize[tind::e2D], kernelShape[tind::eNum],
				1.0, kernelData, nextDiffData + i * nextDiffSize[tind::e3D],
				0.0, colData);

			col2im_cpu(colData, prevDiffShape[tind::eChannels],
				prevDiffShape[tind::eHeight], prevDiffShape[tind::eWidth],
				param_.kernel_h, param_.kernel_w,
				param_.pad_h, param_.pad_w,
				param_.stride_h, param_.stride_w,
				param_.dilation_h, param_.dilation_w,
				prevDiffData + i * prevDiffSize[tind::e3D]);
		}

		//update weight Diff
		gradient_[0]->setZero();
		//const std::vector<int> kernelGradientSize = gradient_[0]->getSize();
		Dtype* kernelGradientData = (Dtype *)gradient_[0]->getData();

		for (int ni = 0; ni < prevDiffShape[tind::eNum]; ni++)
		{
			im2col_cpu<Dtype>(prevData + ni*prevSize[tind::e3D], prevShape[tind::eChannels],
				prevShape[tind::eHeight], prevShape[tind::eWidth],
				param_.kernel_h, param_.kernel_w,
				param_.pad_h, param_.pad_w,
				param_.stride_h, param_.stride_w,
				param_.dilation_h, param_.dilation_w,
				colData);

			// kernelShape[tind::eNum] == nextShape[tind::eChannels]
			gemm(false, true, kernelShape[tind::eNum], kernelSize[tind::e3D], nextSize[tind::e2D],
				1.0, nextDiffData + ni * nextDiffSize[tind::e3D], colData,
				1.0, kernelGradientData);

		}
		div_inplace(kernelGradientData, (Dtype)nextShape[tind::eNum], kernelSize[tind::e4D]);

		//update bias gradient
		gradient_[1]->setZero();
		Dtype* biasGradientData = (Dtype *)gradient_[1]->getData();

		backward_bias(nextDiffShape[tind::eNum], nextDiffShape[tind::eChannels], nextDiffSize[tind::e2D], nextDiffData, biasGradientData);
		div_inplace(biasGradientData, (Dtype)nextShape[tind::eNum], biasSize[tind::e4D]);

	}

	INSTANTIATE_CLASS(ConvolutionOp);

}//namespace