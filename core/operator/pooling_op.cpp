////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#include "operator/pooling_op.h"
#include <sstream>

namespace dlex_cnn
{
	template <typename Dtype>
	PoolingOp<Dtype>::PoolingOp()
	{
		op_type_ = "Pooling";
	}
	template <typename Dtype>
	PoolingOp<Dtype>::PoolingOp(PoolingOpParam param)
	{
		op_type_ = "Pooling";
		param_ = param;
	}
	template <typename Dtype>
	PoolingOp<Dtype>::~PoolingOp()
	{

	}
	template <typename Dtype>
	int PoolingOp<Dtype>::setOpParam(const std::string &opParamStr)
	{
		std::string optStr = opParamStr;
		param_.poolingType = (tind::PoolingType)atoi(fetchSubStr(optStr, "poolingType:", ",").c_str());
		param_.kernel_h = atoi(fetchSubStr(optStr, "kernel_h:", ",").c_str());
		param_.kernel_w = atoi(fetchSubStr(optStr, "kernel_w:", ",").c_str());
		param_.stride_h = atoi(fetchSubStr(optStr, "stride_h:", ",").c_str());
		param_.stride_w = atoi(fetchSubStr(optStr, "stride_w:", ",").c_str());
		param_.global_pooling = atoi(fetchSubStr(optStr, "global_pooling:", ",").c_str());

		return 0;
	}
	template <typename Dtype>
	std::string PoolingOp<Dtype>::genOpParamStr() const
	{
		//tind::PoolingType type = tind::eMAX;
		//int kernel_h = 3, kernel_w = 3;
		//int stride_h = 1, stride_w = 1;
		//int pad_h = 0, pad_w = 0;
		////int channels;
		////int height_, width_;	//输入图像数据大小
		////int pooled_height_, pooled_width_;	//实际池化输出大小，static_cast<int>(ceil(static_cast<float>(height_ + 2 * pad_h_ - kernel_h_) / stride_h_)) + 1;
		//bool global_pooling = false;	//为true时，kernel大小为输入大小

		std::stringstream paramStr;
		paramStr << "poolingType:" << param_.poolingType 
			<< ",kernel_h:" << param_.kernel_h << ",kernel_w:" << param_.kernel_w 
			<< ",stride_h:" << param_.stride_h << ",stride_w:" << param_.stride_w
			<< ",pad_h:" << param_.pad_h << ",pad_w:" << param_.pad_w
			<< ",global_pooling:" << param_.global_pooling << ",";
		return paramStr.str();
	}
	template <typename Dtype>
	int PoolingOp<Dtype>::inferOutShape(std::vector<int> &inShape, std::vector<int> &outShape)
	{
		if (param_.global_pooling)
		{
			param_.kernel_h = inShape[tind::eHeight];
			param_.kernel_w = inShape[tind::eWidth];
		}

		outShape.clear();

		outShape.push_back(inShape[tind::eNum]);
		outShape.push_back(inShape[tind::eChannels]);

		outShape.push_back(static_cast<int>(ceil(static_cast<float>(inShape[tind::eHeight] + 2 * param_.pad_h - param_.kernel_h) / param_.stride_h)) + 1);
		outShape.push_back(static_cast<int>(ceil(static_cast<float>(inShape[tind::eWidth] + 2 * param_.pad_w - param_.kernel_w) / param_.stride_w)) + 1);

		return 0;
	}
	template <typename Dtype>
	int PoolingOp<Dtype>::allocBuf4Node(const std::vector<int> &inShape,
		const std::vector<int> &outShape,
		std::vector<std::shared_ptr<Tensor<Dtype>>> &data) const
	{
		data.clear();
		data.push_back(std::make_shared<Tensor<Dtype>>(inShape));
		return 0;
	}
	template <typename Dtype>
	int PoolingOp<Dtype>::allocOpBuf4Train(const std::vector<int> &inShape, const std::vector<int> &outShape)
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
		if (outShape[tind::eNum] <= 0 || outShape[tind::eChannels] <= 0 ||
			outShape[tind::eHeight] <= 0 || outShape[tind::eWidth] <= 1 ||
			outShape[tind::eNum] > 50000 || outShape[tind::eChannels] > 50000)
		{
			DLOG_ERR("[ InnerProductOp::allocOpBuf4Train ]: outShape is invalid -> (%d, %d, %d, %d) \n",
				outShape[tind::eNum], outShape[tind::eChannels], outShape[tind::eHeight], outShape[tind::eWidth]);
			return -1;
		}

		diff_.clear();
		diff_.push_back(std::make_shared<Tensor<Dtype>>(inShape));

		if (param_.poolingType == tind::eMAX)
			max_idx_map_.reset(new Tensor<int>(outShape));

		return 0;
	}

	template <typename Dtype>
	void PoolingOp<Dtype>::forward(const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev, const std::vector<std::shared_ptr<Tensor<Dtype>>> &next)
	{
		const std::vector<int> prevSize = prev[0]->getSize();
		const std::vector<int> prevShape = prev[0]->getShape();
		const std::vector<int> nextShape = next[0]->getShape();

		Dtype* prevData = (Dtype *)prev[0]->getData();	//bottom_data
		Dtype* nextData = (Dtype *)next[0]->getData();	//top_data

		const std::vector<int> nextSize = next[0]->getSize(); // 4d = top_count
		const bool use_top_mask = next.size() > 1;

		int* mask = NULL;
		switch (this->param_.poolingType) 
		{
		case tind::eMAX:
			// Initialize
			if (max_idx_map_->getSize()[tind::e4D] != nextSize[tind::e4D])
				max_idx_map_.reset(new Tensor<int>(nextShape));

			mask = (int *)max_idx_map_->getData();
			max_idx_map_->setValue(Dtype(-1));

			next[0]->setZero();
			// The main loop
			for (int n = 0; n < prevShape[tind::eNum]; ++n) {
				for (int c = 0; c < prevShape[tind::eChannels]; ++c) {
					for (int ph = 0; ph < nextShape[tind::eHeight]; ++ph) {
						for (int pw = 0; pw < nextShape[tind::eWidth]; ++pw) {
							int hstart = ph * param_.stride_h - param_.pad_h;
							int wstart = pw * param_.stride_w - param_.pad_w;
							int hend = std::min(hstart + param_.kernel_h, prevShape[tind::eHeight]);
							int wend = std::min(wstart + param_.kernel_w, prevShape[tind::eWidth]);
							hstart = std::max(hstart, 0);
							wstart = std::max(wstart, 0);
							const int pool_index = ph * nextShape[tind::eWidth] + pw;
							for (int h = hstart; h < hend; ++h) {
								for (int w = wstart; w < wend; ++w) {
									const int index = h * prevShape[tind::eWidth] + w;
									if (prevData[index] > nextData[pool_index]) {
										nextData[pool_index] = prevData[index];
										mask[pool_index] = index;
									}
								}
							}
						}
					}
					// compute offset
					prevData += prevSize[tind::e2D];
					nextData += nextSize[tind::e2D];
					mask += nextSize[tind::e2D];
				}
			}
			break;
		case tind::eAVE:
			next[0]->setZero();
			// The main loop
			for (int n = 0; n < prevShape[tind::eNum]; ++n) {
				for (int c = 0; c < prevShape[tind::eChannels]; ++c) {
					for (int ph = 0; ph < nextShape[tind::eHeight]; ++ph) {
						for (int pw = 0; pw < nextShape[tind::eWidth]; ++pw) {
							int hstart = ph * param_.stride_h - param_.pad_h;
							int wstart = pw * param_.stride_w - param_.pad_w;
							int hend = std::min(hstart + param_.kernel_h, prevShape[tind::eHeight] + param_.pad_h);
							int wend = std::min(wstart + param_.kernel_w, prevShape[tind::eWidth] + param_.pad_w);
							int pool_size = (hend - hstart) * (wend - wstart);
							hstart = std::max(hstart, 0);
							wstart = std::max(wstart, 0);
							hend = std::min(hend, prevShape[tind::eHeight]);
							wend = std::min(wend, prevShape[tind::eWidth]);
							for (int h = hstart; h < hend; ++h) {
								for (int w = wstart; w < wend; ++w) {
									nextData[ph * nextShape[tind::eWidth] + pw] +=
										prevData[h * prevShape[tind::eWidth] + w];
								}
							}
							nextData[ph * nextShape[tind::eWidth] + pw] /= pool_size;
						}
					}
					// compute offset
					prevData += prevSize[tind::e2D];
					nextData += nextSize[tind::e2D];
				}
			}
			break;
		case tind::eSTOCHASTIC:
			DLOG_ERR("[ PoolingOp::forward ]: pooling method <STOCHASTIC>, not implemented.\n");
			break;
		default:
			DLOG_ERR("[ PoolingOp::forward ]: Unknown pooling method.\n");
		}
	}

	template <typename Dtype>
	void PoolingOp<Dtype>::backward(const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev, const std::vector<std::shared_ptr<Tensor<Dtype>>> &next,
		const std::vector<std::shared_ptr<Tensor<Dtype>>> &prevDiff, const std::vector<std::shared_ptr<Tensor<Dtype>>> &nextDiff)
	{

		const std::vector<int> prevDiffSize = prevDiff[0]->getSize();
		const std::vector<int> nextDiffSize = nextDiff[0]->getSize();
		const std::vector<int> prevDiffShape = prevDiff[0]->getShape();
		const std::vector<int> nextDiffShape = nextDiff[0]->getShape();

		const std::vector<int> prevSize = prev[0]->getSize();
		const std::vector<int> nextSize = next[0]->getSize();
		const std::vector<int> prevShape = prev[0]->getShape();
		const std::vector<int> nextShape = next[0]->getShape();

		Dtype* prevDiffData = (Dtype *)prevDiff[0]->getData();	//bottom_data
		Dtype* nextDiffData = (Dtype *)nextDiff[0]->getData();	//top_data

		prevDiff[0]->setZero();

		const int* mask = NULL;  // suppress warnings about uninitialized variables
		switch (this->param_.poolingType) {
		case tind::eMAX:
			// The main loop
			mask = (int *)max_idx_map_->getData();
			
			for (int n = 0; n < nextShape[tind::eNum]; ++n) {
				for (int c = 0; c < nextShape[tind::eChannels]; ++c) {
					for (int ph = 0; ph < nextShape[tind::eHeight]; ++ph) {
						for (int pw = 0; pw < nextShape[tind::eWidth]; ++pw) {
							const int index = ph * nextShape[tind::eWidth] + pw;
							const int bottom_index = mask[index];
							prevDiffData[bottom_index] += nextDiffData[index];
						}
					}
					prevDiffData += prevDiffSize[tind::e2D];
					nextDiffData += nextDiffSize[tind::e2D];

					mask += nextSize[tind::e2D];
				}
			}
			break;
		case tind::eAVE:
			// The main loop
			for (int n = 0; n < nextShape[tind::eNum]; ++n) {
				for (int c = 0; c < nextShape[tind::eChannels]; ++c) {
					for (int ph = 0; ph < nextShape[tind::eHeight]; ++ph) {
						for (int pw = 0; pw < nextShape[tind::eWidth]; ++pw) {
							int hstart = ph * param_.stride_h - param_.pad_h;
							int wstart = pw * param_.stride_w - param_.pad_w;
							int hend = std::min(hstart + param_.kernel_h, prevShape[tind::eHeight] + param_.pad_h);
							int wend = std::min(wstart + param_.kernel_w, prevShape[tind::eWidth] + param_.pad_w);
							int pool_size = (hend - hstart) * (wend - wstart);
							hstart = std::max(hstart, 0);
							wstart = std::max(wstart, 0);
							hend = std::min(hend, prevShape[tind::eHeight]);
							wend = std::min(wend, prevShape[tind::eWidth]);
							for (int h = hstart; h < hend; ++h) {
								for (int w = wstart; w < wend; ++w) {
									prevDiffData[h * prevShape[tind::eWidth] + w] +=
										nextDiffData[ph * nextShape[tind::eWidth] + pw] / pool_size;
								}
							}
						}
					}
					// offset
					prevDiffData += prevDiffSize[tind::e2D];
					nextDiffData += nextDiffSize[tind::e2D];
				}
			}
			break;
		case tind::eSTOCHASTIC:
			DLOG_ERR("[ PoolingOp::backward ]: pooling method <STOCHASTIC>, not implemented.\n");
			break;
		default:
			DLOG_ERR("[ PoolingOp::backward ]: Unknown pooling method.\n");
		}
	}

	INSTANTIATE_CLASS(PoolingOp);

}//namespace