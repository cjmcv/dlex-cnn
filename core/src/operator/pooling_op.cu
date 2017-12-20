////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  Pooling.
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#ifdef USE_CUDA
#include "operator/pooling_op.h"
#include <sstream>

namespace dlex_cnn
{
	template <typename Dtype>
	__global__ void MaxPoolForward(const int nthreads,
		const Dtype* const bottom_data, const int num, const int channels,
		const int height, const int width, const int pooled_height,
		const int pooled_width, const int kernel_h, const int kernel_w,
		const int stride_h, const int stride_w, const int pad_h, const int pad_w,
		Dtype* const top_data, int* mask) {	//, Dtype* top_mask
		CUDA_KERNEL_LOOP(index, nthreads) {
			const int pw = index % pooled_width;
			const int ph = (index / pooled_width) % pooled_height;
			const int c = (index / pooled_width / pooled_height) % channels;
			const int n = index / pooled_width / pooled_height / channels;
			int hstart = ph * stride_h - pad_h;
			int wstart = pw * stride_w - pad_w;
			const int hend = min(hstart + kernel_h, height);
			const int wend = min(wstart + kernel_w, width);
			hstart = max(hstart, 0);
			wstart = max(wstart, 0);
			Dtype maxval = -FLT_MAX;
			int maxidx = -1;
			const Dtype* const bottom_slice =
				bottom_data + (n * channels + c) * height * width;
			for (int h = hstart; h < hend; ++h) {
				for (int w = wstart; w < wend; ++w) {
					if (bottom_slice[h * width + w] > maxval) {
						maxidx = h * width + w;
						maxval = bottom_slice[maxidx];
					}
				}
			}
			top_data[index] = maxval;
			if (mask) {
				mask[index] = maxidx;
			}
		}
	}

	template <typename Dtype>
	__global__ void AvePoolForward(const int nthreads,
		const Dtype* const bottom_data, const int num, const int channels,
		const int height, const int width, const int pooled_height,
		const int pooled_width, const int kernel_h, const int kernel_w,
		const int stride_h, const int stride_w, const int pad_h, const int pad_w,
		Dtype* const top_data) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			const int pw = index % pooled_width;
			const int ph = (index / pooled_width) % pooled_height;
			const int c = (index / pooled_width / pooled_height) % channels;
			const int n = index / pooled_width / pooled_height / channels;
			int hstart = ph * stride_h - pad_h;
			int wstart = pw * stride_w - pad_w;
			int hend = min(hstart + kernel_h, height + pad_h);
			int wend = min(wstart + kernel_w, width + pad_w);
			const int pool_size = (hend - hstart) * (wend - wstart);
			hstart = max(hstart, 0);
			wstart = max(wstart, 0);
			hend = min(hend, height);
			wend = min(wend, width);
			Dtype aveval = 0;
			const Dtype* const bottom_slice =
				bottom_data + (n * channels + c) * height * width;
			for (int h = hstart; h < hend; ++h) {
				for (int w = wstart; w < wend; ++w) {
					aveval += bottom_slice[h * width + w];
				}
			}
			top_data[index] = aveval / pool_size;
		}
	}

	template <typename Dtype>
	void PoolingOp<Dtype>::forward_gpu(
		const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev,
		const std::vector<std::shared_ptr<Tensor<Dtype>>> &next)
	{
		const std::vector<int> prev_size = prev[0]->getSize();
		const std::vector<int> prev_shape = prev[0]->getShape();
		const std::vector<int> next_shape = next[0]->getShape();

		Dtype* prev_data = (Dtype *)prev[0]->getPushGpuData();
		Dtype* next_data = (Dtype *)next[0]->getPushGpuData();
		const std::vector<int> next_size = next[0]->getSize();

		int* mask = NULL;
		bool mflag = true;
		switch (this->param_.pooling_type) {
		case tind::eMAX:
			if (max_idx_map_ == NULL)	// max_idx_map_ only works in training phase for pooling's backward.
				mflag = false;
			else if (max_idx_map_->getSize()[tind::e4D] != next_size[tind::e4D])
				max_idx_map_.reset(new Tensor<int>(next_shape));

			mask = (int *)max_idx_map_->getPushGpuData();
			max_idx_map_->setGpuValue(Dtype(-1));

			// NOLINT_NEXT_LINE(whitespace/operators)
			MaxPoolForward<Dtype> << <DLEX_GET_BLOCKS(next_size[tind::e4D]), DLEX_CUDA_NUM_THREADS >> >(
				next_size[tind::e4D], prev_data, prev_shape[tind::eNum], prev_shape[tind::eChannels],
				prev_shape[tind::eHeight], prev_shape[tind::eWidth], next_shape[tind::eHeight], next_shape[tind::eWidth],
				param_.kernel_h, param_.kernel_w, param_.stride_h, param_.stride_w, param_.pad_h, param_.pad_w, next_data,
				mask);
			break;
		case tind::eAVE:
			// NOLINT_NEXT_LINE(whitespace/operators)
			AvePoolForward<Dtype> << <DLEX_GET_BLOCKS(next_size[tind::e4D]), DLEX_CUDA_NUM_THREADS >> >(
				next_size[tind::e4D], prev_data, prev_shape[tind::eNum], prev_shape[tind::eChannels],
				prev_shape[tind::eHeight], prev_shape[tind::eWidth], next_shape[tind::eHeight], next_shape[tind::eWidth], param_.kernel_h,
				param_.kernel_w, param_.stride_h, param_.stride_w, param_.pad_h, param_.pad_w, next_data);
			break;
		default:
			DLOG_ERR("Unknown pooling method.");
		}
		CUDA_POST_KERNEL_CHECK;
	}


	template <typename Dtype>
	__global__ void MaxPoolBackward(const int nthreads, const Dtype* const top_diff,
		const int* const mask, const int num,
		const int channels, const int height, const int width,
		const int pooled_height, const int pooled_width, const int kernel_h,
		const int kernel_w, const int stride_h, const int stride_w, const int pad_h,
		const int pad_w, Dtype* const bottom_diff) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			// find out the local index
			// find out the local offset
			const int w = index % width;
			const int h = (index / width) % height;
			const int c = (index / width / height) % channels;
			const int n = index / width / height / channels;
			const int phstart =
				(h + pad_h < kernel_h) ? 0 : (h + pad_h - kernel_h) / stride_h + 1;
			const int phend = min((h + pad_h) / stride_h + 1, pooled_height);
			const int pwstart =
				(w + pad_w < kernel_w) ? 0 : (w + pad_w - kernel_w) / stride_w + 1;
			const int pwend = min((w + pad_w) / stride_w + 1, pooled_width);
			Dtype gradient = 0;
			const int offset = (n * channels + c) * pooled_height * pooled_width;
			const Dtype* const top_diff_slice = top_diff + offset;
			if (mask) {
				const int* const mask_slice = mask + offset;
				for (int ph = phstart; ph < phend; ++ph) {
					for (int pw = pwstart; pw < pwend; ++pw) {
						if (mask_slice[ph * pooled_width + pw] == h * width + w) {
							gradient += top_diff_slice[ph * pooled_width + pw];
						}
					}
				}
			}
			bottom_diff[index] = gradient;
		}
	}

	template <typename Dtype>
	__global__ void AvePoolBackward(const int nthreads, const Dtype* const top_diff,
		const int num, const int channels, const int height,
		const int width, const int pooled_height, const int pooled_width,
		const int kernel_h, const int kernel_w, const int stride_h,
		const int stride_w, const int pad_h, const int pad_w,
		Dtype* const bottom_diff) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			// find out the local index
			// find out the local offset
			const int w = index % width + pad_w;
			const int h = (index / width) % height + pad_h;
			const int c = (index / width / height) % channels;
			const int n = index / width / height / channels;
			const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
			const int phend = min(h / stride_h + 1, pooled_height);
			const int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
			const int pwend = min(w / stride_w + 1, pooled_width);
			Dtype gradient = 0;
			const Dtype* const top_diff_slice =
				top_diff + (n * channels + c) * pooled_height * pooled_width;
			for (int ph = phstart; ph < phend; ++ph) {
				for (int pw = pwstart; pw < pwend; ++pw) {
					// figure out the pooling size
					int hstart = ph * stride_h - pad_h;
					int wstart = pw * stride_w - pad_w;
					int hend = min(hstart + kernel_h, height + pad_h);
					int wend = min(wstart + kernel_w, width + pad_w);
					int pool_size = (hend - hstart) * (wend - wstart);
					gradient += top_diff_slice[ph * pooled_width + pw] / pool_size;
				}
			}
			bottom_diff[index] = gradient;
		}
	}

	template <typename Dtype>
	void PoolingOp<Dtype>::backward_gpu(
		const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev,
		const std::vector<std::shared_ptr<Tensor<Dtype>>> &next,
		const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev_diff,
		const std::vector<std::shared_ptr<Tensor<Dtype>>> &next_diff
		) 
	{
		const std::vector<int> prev_diff_size = prev_diff[0]->getSize();
		const std::vector<int> next_diff_size = next_diff[0]->getSize();
		const std::vector<int> prev_diff_shape = prev_diff[0]->getShape();
		const std::vector<int> next_diff_shape = next_diff[0]->getShape();

		const std::vector<int> prev_size = prev[0]->getSize();
		const std::vector<int> next_size = next[0]->getSize();
		const std::vector<int> prev_shape = prev[0]->getShape();
		const std::vector<int> next_shape = next[0]->getShape();

		Dtype* prev_diff_data = (Dtype *)prev_diff[0]->getPushGpuData();	//bottom_data
		Dtype* next_diff_data = (Dtype *)next_diff[0]->getPushGpuData();	//top_data

		prev_diff[0]->setGpuZero();

		const int* mask = NULL;  // suppress warnings about uninitialized variables

		switch (this->param_.pooling_type) {
		case tind::eMAX:
			mask = (int *)max_idx_map_->getPushGpuData();
			MaxPoolBackward<Dtype> << <DLEX_GET_BLOCKS(prev_diff_size[tind::e4D]), DLEX_CUDA_NUM_THREADS >> >(
				prev_diff_size[tind::e4D], next_diff_data, mask, prev_shape[tind::eNum], prev_shape[tind::eChannels],
				prev_shape[tind::eHeight], prev_shape[tind::eWidth], next_shape[tind::eHeight], next_shape[tind::eWidth],
				param_.kernel_h, param_.kernel_w, param_.stride_h, param_.stride_w, param_.pad_h, param_.pad_w,
				prev_diff_data);
			break;
		case tind::eAVE:
			AvePoolBackward<Dtype> << <DLEX_GET_BLOCKS(prev_diff_size[tind::e4D]), DLEX_CUDA_NUM_THREADS >> >(
				prev_diff_size[tind::e4D], next_diff_data, prev_shape[tind::eNum], prev_shape[tind::eChannels],
				prev_shape[tind::eHeight], prev_shape[tind::eWidth], next_shape[tind::eHeight], next_shape[tind::eWidth],
				param_.kernel_h, param_.kernel_w, param_.stride_h, param_.stride_w, param_.pad_h, param_.pad_w, prev_diff_data);
			break;
		default:
			DLOG_ERR("Unknown pooling method.");
		}
		CUDA_POST_KERNEL_CHECK;
	}
	template void PoolingOp<float>::forward_gpu(
		const std::vector<std::shared_ptr<Tensor<float>>> &prev,
		const std::vector<std::shared_ptr<Tensor<float>>> &next);
	template void PoolingOp<double>::forward_gpu(
		const std::vector<std::shared_ptr<Tensor<double>>> &prev,
		const std::vector<std::shared_ptr<Tensor<double>>> &next);
	template void PoolingOp<float>::backward_gpu(
		const std::vector<std::shared_ptr<Tensor<float>>> &prev,
		const std::vector<std::shared_ptr<Tensor<float>>> &next,
		const std::vector<std::shared_ptr<Tensor<float>>> &prev_diff,
		const std::vector<std::shared_ptr<Tensor<float>>> &next_diff);
	template void PoolingOp<double>::backward_gpu(
		const std::vector<std::shared_ptr<Tensor<double>>> &prev,
		const std::vector<std::shared_ptr<Tensor<double>>> &next,
		const std::vector<std::shared_ptr<Tensor<double>>> &prev_diff,
		const std::vector<std::shared_ptr<Tensor<double>>> &next_diff);
}//namespace
#endif