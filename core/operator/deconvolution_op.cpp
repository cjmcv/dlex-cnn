////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#include "operator/deconvolution_op.h"
#include <sstream>

namespace dlex_cnn
{
	template <typename Dtype>
	DeconvolutionOp<Dtype>::DeconvolutionOp()
	{
		op_type_ = "Deconvolution";
	}
	template <typename Dtype>
	DeconvolutionOp<Dtype>::DeconvolutionOp(DeconvolutionOpParam param)
	{
		op_type_ = "Deconvolution";
		param_ = param;
	}
	template <typename Dtype>
	DeconvolutionOp<Dtype>::~DeconvolutionOp()
	{

	}
	template <typename Dtype>
	int DeconvolutionOp<Dtype>::setOpParam(const std::string &opParamStr)
	{
		std::string optStr = opParamStr;
		param_.blas_enable = atoi(fetchSubStr(optStr, "blas_enable:", ",").c_str());
		param_.kernel_channels = atoi(fetchSubStr(optStr, "kernel_channels:", ",").c_str());
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
	std::string DeconvolutionOp<Dtype>::genOpParamStr() const
	{
		std::stringstream paramStr;
		paramStr << "blas_enable:" << param_.blas_enable << ",kernel_channels:" << param_.kernel_channels
			<< ",kernel_h:" << param_.kernel_h << ",kernel_w:" << param_.kernel_w
			<< ",stride_h:" << param_.stride_h << ",stride_w:" << param_.stride_w
			<< ",pad_h:" << param_.pad_h << ",pad_w:" << param_.pad_w
			<< ",dilation_h:" << param_.dilation_h << ",dilation_w:" << param_.dilation_w << ",";
		return paramStr.str();
	}
	template <typename Dtype>
	int DeconvolutionOp<Dtype>::inferOutShape(std::vector<int> &inShape, std::vector<int> &outShape)
	{
		outShape.clear();

		outShape.push_back(inShape[tind::eNum]);
		outShape.push_back(param_.kernel_channels); // different from conv

		// just reverse the input and output dimension of conv.
		outShape.push_back((inShape[tind::eHeight] - 1) * param_.stride_h + (param_.dilation_h * (param_.kernel_h - 1) + 1) - 2 * param_.pad_h);
		outShape.push_back((inShape[tind::eWidth] - 1) * param_.stride_w + (param_.dilation_w * (param_.kernel_w - 1) + 1) - 2 * param_.pad_w);

		return 0;
	}
	template <typename Dtype>
	int DeconvolutionOp<Dtype>::allocBuf4Node(const std::vector<int> &inShape,
		const std::vector<int> &outShape,
		std::vector<std::shared_ptr<Tensor<Dtype>>> &data) const
	{
		data.clear();
		//printf("data and gradient: size() : %d, %d\n", data.size(), gradient_.size());
		data.push_back(std::make_shared<Tensor<Dtype>>(inShape));

		//weight (kernels for convolution)
		data.push_back(std::make_shared<Tensor<Dtype>>(inShape[tind::eChannels], param_.kernel_channels, param_.kernel_w, param_.kernel_h)); // Different from conv, switch the input channels and output channels.
		normal_distribution_init<Dtype>((Dtype *)data[1]->getData(), data[1]->getSize()[tind::e4D], 0.0f, 0.1f);

		//blas
		if (param_.blas_enable)
		{
			data.push_back(std::make_shared<Tensor<Dtype>>(outShape[tind::eChannels], 1, 1, 1));	// The same with conv, bias size is equal to output channel num
			dlex_set<Dtype>((Dtype *)data[2]->getData(), data[2]->getSize()[tind::e4D], 0.0f);
		}
		return 0;
	}
	template <typename Dtype>
	int DeconvolutionOp<Dtype>::allocOpBuf4Train(const std::vector<int> &inShape, const std::vector<int> &outShape)
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
		gradient_.push_back(std::make_shared<Tensor<Dtype>>(inShape[tind::eChannels], param_.kernel_channels, param_.kernel_w, param_.kernel_h)); // Different from conv
		dlex_set<Dtype>((Dtype *)gradient_[0]->getData(), gradient_[0]->getSize()[tind::e4D], 0.0f);

		if (param_.blas_enable)
		{
			gradient_.push_back(std::make_shared<Tensor<Dtype>>(outShape[tind::eChannels], 1, 1, 1));
			dlex_set<Dtype>((Dtype *)gradient_[1]->getData(), gradient_[1]->getSize()[tind::e4D], 0.0f);
		}
		return 0;
	}

	////
	void matrixShow_float(std::string name, float *data, int num, int channel, int height, int width)
	{
		printf("Matrix :%s\n", name.c_str());
		printf("(%d, %d, %d, %d \n", num, channel, height, width);
		int c_size = height * width;
		int n_size = channel * c_size;
		for (int n = 0; n < num; n++)
		{
			for (int c = 0; c < channel; c++)
			{
				printf(" n - ch : %d (%d)\n", n, c);
				for (int i = 0; i < height; i++)
				{
					for (int j = 0; j < width; j++)
					{
						printf("%f, ", *(data + n*n_size + c*c_size + i*width + j));
					}
					printf("\n");
				}
			}
		}
		printf(")\n");
	}
	////
	template <typename Dtype>
	void DeconvolutionOp<Dtype>::forward(const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev, const std::vector<std::shared_ptr<Tensor<Dtype>>> &next)
	{
		const std::vector<int> prevShape = prev[0]->getShape();
		const std::vector<int> nextShape = next[0]->getShape();

		const std::vector<int> prevSize = prev[0]->getSize();
		const std::vector<int> nextSize = next[0]->getSize();

		const std::vector<int> kernelShape = prev[1]->getShape();
		const std::vector<int> kernelSize = prev[1]->getSize();

		const Dtype* prevData = (Dtype *)prev[0]->getData();
		const Dtype* kernelData = (Dtype *)prev[1]->getData();
		const Dtype* biasData = (Dtype *)prev[2]->getData();
		Dtype* nextData = (Dtype *)next[0]->getData();

		// (1, channels*kernel_h*kernel_w, output_h*output_w)
		const int output_h = (prevShape[tind::eHeight] - 1) * param_.stride_h + (param_.dilation_h * (param_.kernel_h - 1) + 1) - 2 * param_.pad_h;
		const int output_w = (prevShape[tind::eWidth] - 1) * param_.stride_w + (param_.dilation_w * (param_.kernel_w - 1) + 1) - 2 * param_.pad_w;

		// The dimension of col_buffer is relevent to "kernel * prev". -> From the output of "kernel * prev" to col_buffer.
		int colHeight = kernelSize[tind::e3D];// prevShape[tind::eChannels] * param_.kernel_h * param_.kernel_w;
		int colWidth = prevSize[tind::e2D];
		if (col_buffer_ == NULL)
			col_buffer_ = std::make_shared<Tensor<Dtype>>(1, 1, colHeight, colWidth);
		else if (col_buffer_->getSize()[tind::e4D] != 1 * 1 * colHeight * colWidth)
			col_buffer_.reset(new Tensor<Dtype>(1, 1, colHeight, colWidth));

		Dtype* colData = (Dtype *)col_buffer_->getData();

		for (int ni = 0; ni < prevShape[tind::eNum]; ni++)
		{
			gemm(true, false, kernelSize[tind::e3D], prevSize[tind::e2D], prevShape[tind::eChannels],
				1.0, kernelData, prevData + ni * prevSize[tind::e3D],
				0.0, colData);

			//matrixShow_float("kernel", (float *)kernelData, 1, 1, kernelShape[tind::eNum], kernelSize[tind::e3D]);
			//matrixShow_float("prev", (float *)prevData, 1, 1, prevShape[tind::eChannels], prevSize[tind::e2D]);
			//matrixShow_float("col", (float *)colData, 1, 1, colHeight, colWidth);

			col2im_cpu(colData, nextShape[tind::eChannels],
				nextShape[tind::eHeight], nextShape[tind::eWidth],
				param_.kernel_h, param_.kernel_w,
				param_.pad_h, param_.pad_w,
				param_.stride_h, param_.stride_w,
				param_.dilation_h, param_.dilation_w,
				nextData + ni * nextSize[tind::e3D]);

			//matrixShow_float("res", (float *)nextData, 1, nextShape[tind::eChannels], nextShape[tind::eHeight], nextShape[tind::eWidth]);
		}
		if (param_.blas_enable)
			add_bias(nextShape[tind::eNum], nextShape[tind::eChannels], nextSize[tind::e2D], biasData, nextData);
	}

	template <typename Dtype>
	void DeconvolutionOp<Dtype>::backward(const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev, const std::vector<std::shared_ptr<Tensor<Dtype>>> &next,
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

		const std::vector<int> colShape = col_buffer_->getShape();
		Dtype* colData = (Dtype *)col_buffer_->getData();

		// update prevDiff
		prevDiff[0]->setZero();
		for (int ni = 0; ni < prevShape[tind::eNum]; ni++)
		{
			//printf("address: %d\n", colData);
			im2col_cpu<Dtype>(nextDiffData + ni*nextDiffSize[tind::e3D], nextDiffShape[tind::eChannels],
				nextDiffShape[tind::eHeight], nextDiffShape[tind::eWidth],
				param_.kernel_h, param_.kernel_w,
				param_.pad_h, param_.pad_w,
				param_.stride_h, param_.stride_w,
				param_.dilation_h, param_.dilation_w,
				colData);

			// colShape[tind::eHeight] == kernelSize[tind::e3D]
			// kernel_num == prev channels
			gemm(false, false, prevShape[tind::eChannels], colShape[tind::eWidth], colShape[tind::eHeight], 1, kernelData, colData, 0, prevDiffData + ni * prevDiffSize[tind::e3D]);
		}

		// update weight Diff
		gradient_[0]->setZero();
		Dtype* kernelGradientData = (Dtype *)gradient_[0]->getData();
		for (int ni = 0; ni < prevDiffShape[tind::eNum]; ni++)
		{
			im2col_cpu<Dtype>(nextDiffData + ni*nextDiffSize[tind::e3D], nextDiffShape[tind::eChannels],
				nextDiffShape[tind::eHeight], nextDiffShape[tind::eWidth],
				param_.kernel_h, param_.kernel_w,
				param_.pad_h, param_.pad_w,
				param_.stride_h, param_.stride_w,
				param_.dilation_h, param_.dilation_w,
				colData);

			gemm(false, true, prevShape[tind::eChannels], kernelSize[tind::e3D], prevSize[tind::e2D],
				1.0, prevData + ni * prevSize[tind::e3D], colData,
				1.0, kernelGradientData);
		}
		div_inplace(kernelGradientData, (Dtype)nextShape[tind::eNum], kernelSize[tind::e4D]);

		//update bias gradient
		gradient_[1]->setZero();
		Dtype* biasGradientData = (Dtype *)gradient_[1]->getData();

		backward_bias(nextDiffShape[tind::eNum], nextDiffShape[tind::eChannels], nextDiffSize[tind::e2D], nextDiffData, biasGradientData);
		div_inplace(biasGradientData, (Dtype)nextShape[tind::eNum], biasSize[tind::e4D]);
	}

	INSTANTIATE_CLASS(DeconvolutionOp);

}//namespace