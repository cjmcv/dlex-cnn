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
		max_idx_map_ = NULL;
	}
	template <typename Dtype>
	PoolingOp<Dtype>::PoolingOp(PoolingOpParam param)
	{
		op_type_ = "Pooling";
		max_idx_map_ = NULL;
		param_ = param;
	}
	template <typename Dtype>
	PoolingOp<Dtype>::~PoolingOp()
	{

	}
	template <typename Dtype>
	int PoolingOp<Dtype>::setOpParam(const std::string &op_param_str)
	{
		std::string opt_str = op_param_str;
		param_.pooling_type = (tind::PoolingType)atoi(fetchSubStr(opt_str, "pooling_type:", ",").c_str());
		param_.kernel_h = atoi(fetchSubStr(opt_str, "kernel_h:", ",").c_str());
		param_.kernel_w = atoi(fetchSubStr(opt_str, "kernel_w:", ",").c_str());
		param_.stride_h = atoi(fetchSubStr(opt_str, "stride_h:", ",").c_str());
		param_.stride_w = atoi(fetchSubStr(opt_str, "stride_w:", ",").c_str());
		param_.pad_h = atoi(fetchSubStr(opt_str, "pad_h:", ",").c_str());
		param_.pad_w = atoi(fetchSubStr(opt_str, "pad_w:", ",").c_str());
		param_.global_pooling = atoi(fetchSubStr(opt_str, "global_pooling:", ",").c_str());

		return 0;
	}
	template <typename Dtype>
	std::string PoolingOp<Dtype>::genOpParamStr() const
	{
		std::stringstream param_str;
		param_str << "pooling_type:" << param_.pooling_type 
			<< ",kernel_h:" << param_.kernel_h << ",kernel_w:" << param_.kernel_w 
			<< ",stride_h:" << param_.stride_h << ",stride_w:" << param_.stride_w
			<< ",pad_h:" << param_.pad_h << ",pad_w:" << param_.pad_w
			<< ",global_pooling:" << param_.global_pooling << ",";
		return param_str.str();
	}
	template <typename Dtype>
	int PoolingOp<Dtype>::inferOutShape(std::vector<int> &in_shape, std::vector<int> &out_shape)
	{
		if (param_.global_pooling)
		{
			param_.kernel_h = in_shape[tind::eHeight];
			param_.kernel_w = in_shape[tind::eWidth];
		}

		out_shape.clear();

		out_shape.push_back(in_shape[tind::eNum]);
		out_shape.push_back(in_shape[tind::eChannels]);

		out_shape.push_back(static_cast<int>(ceil(static_cast<float>(in_shape[tind::eHeight] + 2 * param_.pad_h - param_.kernel_h) / param_.stride_h)) + 1);
		out_shape.push_back(static_cast<int>(ceil(static_cast<float>(in_shape[tind::eWidth] + 2 * param_.pad_w - param_.kernel_w) / param_.stride_w)) + 1);

		return 0;
	}
	template <typename Dtype>
	int PoolingOp<Dtype>::allocBuf4Node(const std::vector<int> &in_shape,
		const std::vector<int> &out_shape,
		std::vector<std::shared_ptr<Tensor<Dtype>>> &data) const
	{
		data.clear();
		data.push_back(std::make_shared<Tensor<Dtype>>(in_shape));
		return 0;
	}
	template <typename Dtype>
	int PoolingOp<Dtype>::allocOpBuf4Train(const std::vector<int> &in_shape, const std::vector<int> &out_shape)
	{
		if (in_shape[tind::eNum] <= 0 || in_shape[tind::eChannels] <= 0 ||
			in_shape[tind::eHeight] <= 0 || in_shape[tind::eWidth] <= 0 ||
			in_shape[tind::eNum] > 5000 || in_shape[tind::eChannels] > 5000 ||
			in_shape[tind::eHeight] > 5000 || in_shape[tind::eWidth] > 5000)
		{
			DLOG_ERR("[ PoolingOp::allocOpBuf4Train ]: in_shape is invalid -> (%d, %d, %d, %d) \n",
				in_shape[tind::eNum], in_shape[tind::eChannels], in_shape[tind::eHeight], in_shape[tind::eWidth]);
			return -1;
		}
		if (out_shape[tind::eNum] <= 0 || out_shape[tind::eChannels] <= 0 ||
			out_shape[tind::eHeight] <= 0 || out_shape[tind::eWidth] <= 0 ||
			out_shape[tind::eNum] > 50000 || out_shape[tind::eChannels] > 50000)
		{
			DLOG_ERR("[ PoolingOp::allocOpBuf4Train ]: out_shape is invalid -> (%d, %d, %d, %d) \n",
				out_shape[tind::eNum], out_shape[tind::eChannels], out_shape[tind::eHeight], out_shape[tind::eWidth]);
			return -1;
		}

		diff_.clear();
		diff_.push_back(std::make_shared<Tensor<Dtype>>(in_shape));

		if (param_.pooling_type == tind::eMAX)
			max_idx_map_.reset(new Tensor<int>(out_shape));

		return 0;
	}

	template <typename Dtype>
	void PoolingOp<Dtype>::forward(
		const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev, 
		const std::vector<std::shared_ptr<Tensor<Dtype>>> &next)
	{
		const std::vector<int> prev_size = prev[0]->getSize();
		const std::vector<int> prev_shape = prev[0]->getShape();
		const std::vector<int> next_shape = next[0]->getShape();

		Dtype* prev_data = (Dtype *)prev[0]->getPushCpuData();	//bottom_data
		Dtype* next_data = (Dtype *)next[0]->getPushCpuData();	//top_data

		const std::vector<int> next_size = next[0]->getSize(); // 4d = top_count

		int* mask = NULL;
		bool mflag = true;
		switch (this->param_.pooling_type) 
		{
		case tind::eMAX:
			// Initialize
			if (max_idx_map_ == NULL)	// max_idx_map_ only works in training phase for pooling's backward.
				mflag = false;
			else if (max_idx_map_->getSize()[tind::e4D] != next_size[tind::e4D])
				max_idx_map_.reset(new Tensor<int>(next_shape));
				
			mask = (int *)max_idx_map_->getPushCpuData();
			max_idx_map_->setCpuValue(Dtype(-1));

			next[0]->setCpuZero();
			// The main loop
			for (int n = 0; n < prev_shape[tind::eNum]; ++n) {
				for (int c = 0; c < prev_shape[tind::eChannels]; ++c) {
					for (int ph = 0; ph < next_shape[tind::eHeight]; ++ph) {
						for (int pw = 0; pw < next_shape[tind::eWidth]; ++pw) {
							int hstart = ph * param_.stride_h - param_.pad_h;
							int wstart = pw * param_.stride_w - param_.pad_w;
							int hend = std::min(hstart + param_.kernel_h, prev_shape[tind::eHeight]);
							int wend = std::min(wstart + param_.kernel_w, prev_shape[tind::eWidth]);
							hstart = std::max(hstart, 0);
							wstart = std::max(wstart, 0);
							const int pool_index = ph * next_shape[tind::eWidth] + pw;
							for (int h = hstart; h < hend; ++h) {
								for (int w = wstart; w < wend; ++w) {
									const int index = h * prev_shape[tind::eWidth] + w;
									if (prev_data[index] > next_data[pool_index]) {
										next_data[pool_index] = prev_data[index];
										if (mflag)
											mask[pool_index] = index;
									}
								}
							}
						}
					}
					// compute offset
					prev_data += prev_size[tind::e2D];
					next_data += next_size[tind::e2D];
					if (mflag)
						mask += next_size[tind::e2D];
				}
			}
			break;
		case tind::eAVE:
			next[0]->setCpuZero();
			// The main loop
			for (int n = 0; n < prev_shape[tind::eNum]; ++n) {
				for (int c = 0; c < prev_shape[tind::eChannels]; ++c) {
					for (int ph = 0; ph < next_shape[tind::eHeight]; ++ph) {
						for (int pw = 0; pw < next_shape[tind::eWidth]; ++pw) {
							int hstart = ph * param_.stride_h - param_.pad_h;
							int wstart = pw * param_.stride_w - param_.pad_w;
							int hend = std::min(hstart + param_.kernel_h, prev_shape[tind::eHeight] + param_.pad_h);
							int wend = std::min(wstart + param_.kernel_w, prev_shape[tind::eWidth] + param_.pad_w);
							int pool_size = (hend - hstart) * (wend - wstart);
							hstart = std::max(hstart, 0);
							wstart = std::max(wstart, 0);
							hend = std::min(hend, prev_shape[tind::eHeight]);
							wend = std::min(wend, prev_shape[tind::eWidth]);
							for (int h = hstart; h < hend; ++h) {
								for (int w = wstart; w < wend; ++w) {
									next_data[ph * next_shape[tind::eWidth] + pw] +=
										prev_data[h * prev_shape[tind::eWidth] + w];
								}
							}
							next_data[ph * next_shape[tind::eWidth] + pw] /= pool_size;
						}
					}
					// compute offset
					prev_data += prev_size[tind::e2D];
					next_data += next_size[tind::e2D];
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
	void PoolingOp<Dtype>::backward(
		const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev, 
		const std::vector<std::shared_ptr<Tensor<Dtype>>> &next,
		const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev_diff, 
		const std::vector<std::shared_ptr<Tensor<Dtype>>> &next_diff)
	{
		const std::vector<int> prev_diff_size = prev_diff[0]->getSize();
		const std::vector<int> next_diff_size = next_diff[0]->getSize();
		const std::vector<int> prev_diff_shape = prev_diff[0]->getShape();
		const std::vector<int> next_diff_shape = next_diff[0]->getShape();

		const std::vector<int> prev_size = prev[0]->getSize();
		const std::vector<int> next_size = next[0]->getSize();
		const std::vector<int> prev_shape = prev[0]->getShape();
		const std::vector<int> next_shape = next[0]->getShape();

		Dtype* prev_diff_data = (Dtype *)prev_diff[0]->getPushCpuData();	//bottom_data
		Dtype* next_diff_data = (Dtype *)next_diff[0]->getPushCpuData();	//top_data

		prev_diff[0]->setCpuZero();

		const int* mask = NULL;  // suppress warnings about uninitialized variables
		switch (this->param_.pooling_type) {
		case tind::eMAX:
			// The main loop
			mask = (int *)max_idx_map_->getPushCpuData();
			
			for (int n = 0; n < next_shape[tind::eNum]; ++n) {
				for (int c = 0; c < next_shape[tind::eChannels]; ++c) {
					for (int ph = 0; ph < next_shape[tind::eHeight]; ++ph) {
						for (int pw = 0; pw < next_shape[tind::eWidth]; ++pw) {
							const int index = ph * next_shape[tind::eWidth] + pw;
							const int bottom_index = mask[index];
							if (bottom_index >= 0)
								prev_diff_data[bottom_index] += next_diff_data[index];
						}
					}
					prev_diff_data += prev_diff_size[tind::e2D];
					next_diff_data += next_diff_size[tind::e2D];

					mask += next_size[tind::e2D];
				}
			}
			break;
		case tind::eAVE:
			// The main loop
			for (int n = 0; n < next_shape[tind::eNum]; ++n) {
				for (int c = 0; c < next_shape[tind::eChannels]; ++c) {
					for (int ph = 0; ph < next_shape[tind::eHeight]; ++ph) {
						for (int pw = 0; pw < next_shape[tind::eWidth]; ++pw) {
							int hstart = ph * param_.stride_h - param_.pad_h;
							int wstart = pw * param_.stride_w - param_.pad_w;
							int hend = std::min(hstart + param_.kernel_h, prev_shape[tind::eHeight] + param_.pad_h);
							int wend = std::min(wstart + param_.kernel_w, prev_shape[tind::eWidth] + param_.pad_w);
							int pool_size = (hend - hstart) * (wend - wstart);
							hstart = std::max(hstart, 0);
							wstart = std::max(wstart, 0);
							hend = std::min(hend, prev_shape[tind::eHeight]);
							wend = std::min(wend, prev_shape[tind::eWidth]);
							for (int h = hstart; h < hend; ++h) {
								for (int w = wstart; w < wend; ++w) {
									prev_diff_data[h * prev_shape[tind::eWidth] + w] +=
										next_diff_data[ph * next_shape[tind::eWidth] + pw] / pool_size;
								}
							}
						}
					}
					// offset
					prev_diff_data += prev_diff_size[tind::e2D];
					next_diff_data += next_diff_size[tind::e2D];
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

#ifdef USE_CUDA
	//template <typename Dtype>
	//void PoolingOp<Dtype>::forward_gpu(
	//	const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev,
	//	const std::vector<std::shared_ptr<Tensor<Dtype>>> &next)
	//{
	//	forward_gpu(prev, next);
	//}
	//template <typename Dtype>
	//void PoolingOp<Dtype>::backward_gpu(
	//	const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev,
	//	const std::vector<std::shared_ptr<Tensor<Dtype>>> &next,
	//	const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev_diff,
	//	const std::vector<std::shared_ptr<Tensor<Dtype>>> &next_diff)
	//{
	//	backward(prev, next, prev_diff, next_diff);
	//}
#endif

	INSTANTIATE_CLASS(PoolingOp);

}//namespace