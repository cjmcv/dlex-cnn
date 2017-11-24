////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#include "util/device.h"

#include "operator/inner_product_op.h"
#include "util/math_functions.h"
#include <sstream>

namespace dlex_cnn
{
	template <typename Dtype>
	InnerProductOp<Dtype>::InnerProductOp()
	{
		op_type_ = "InnerProduct";
	}
	template <typename Dtype>
	InnerProductOp<Dtype>::InnerProductOp(InnerProductOpParam param)
	{
		op_type_ = "InnerProduct";
		param_ = param;
	}
	template <typename Dtype>
	InnerProductOp<Dtype>::~InnerProductOp()
	{

	}
	template <typename Dtype>
	int InnerProductOp<Dtype>::setOpParam(const std::string &op_param_str)
	{
		std::string opt_str = op_param_str;
		param_.blas_enable = atoi(fetchSubStr(opt_str, "blas_enable:", ",").c_str());
		param_.num_hidden = atoi(fetchSubStr(opt_str, "num_hidden:", ",").c_str());

		return 0;
	}
	template <typename Dtype>
	std::string InnerProductOp<Dtype>::genOpParamStr() const
	{
		std::stringstream param_str;
		param_str << "blas_enable:" << param_.blas_enable << ",num_hidden:" << param_.num_hidden << ",";
		return param_str.str();
	}
	template <typename Dtype>
	int InnerProductOp<Dtype>::inferOutShape(std::vector<int> &in_shape, std::vector<int> &out_shape)
	{
		out_shape.clear();
		out_shape.push_back(in_shape[tind::eNum]);
		out_shape.push_back(param_.num_hidden);
		out_shape.push_back(1);
		out_shape.push_back(1);
		return 0;
	}
	template <typename Dtype>
	int InnerProductOp<Dtype>::allocBuf4Node(const std::vector<int> &in_shape, 
		const std::vector<int> &out_shape,
		std::vector<std::shared_ptr<Tensor<Dtype>>> &data) const
	{
		data.clear();
		printf("data and gradient: size() : %d, %d\n", data.size(), gradient_.size());

		int inShape3DSize = in_shape[1] * in_shape[2] * in_shape[3];
		int outShape3DSize = out_shape[1] * out_shape[2] * out_shape[3];

		data.push_back(std::make_shared<Tensor<Dtype>>(in_shape));

		//weight
		data.push_back(std::make_shared<Tensor<Dtype>>(1, inShape3DSize * outShape3DSize, 1, 1));
		normal_distribution_init<Dtype>((Dtype *)data[1]->getCpuData(), data[1]->getSize()[tind::e4D], 0.0f, 0.1f);

		//blas
		if (param_.blas_enable)
		{
			data.push_back(std::make_shared<Tensor<Dtype>>(1, out_shape[1], 1, 1));
			dlex_set<Dtype>((Dtype *)data[2]->getCpuData(), data[2]->getSize()[tind::e4D], 0.0f);
		}
		return 0;
	}
	template <typename Dtype>
	int InnerProductOp<Dtype>::allocOpBuf4Train(const std::vector<int> &in_shape, const std::vector<int> &out_shape)
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
		if (out_shape[tind::eNum] <= 0 || out_shape[tind::eChannels] <= 0 || 
			out_shape[tind::eHeight] != 1 || out_shape[tind::eWidth] != 1 ||
			out_shape[tind::eNum] > 50000 || out_shape[tind::eChannels] > 50000)
		{
			DLOG_ERR("[ InnerProductOp::allocOpBuf4Train ]: out_shape is invalid -> (%d, %d, %d, %d) \n",
				out_shape[tind::eNum], out_shape[tind::eChannels], out_shape[tind::eHeight], out_shape[tind::eWidth]);
			return -1;
		}
		
		//data.clear();
		gradient_.clear();
		diff_.clear();
		
		//printf("data and gradient: size() : %d, %d\n", data.size(), gradient_.size());

		int inShape3DSize = in_shape[1] * in_shape[2] * in_shape[3];
		int outShape3DSize = out_shape[1] * out_shape[2] * out_shape[3];
		
		diff_.push_back(std::make_shared<Tensor<Dtype>>(in_shape));

		//data.push_back(std::make_shared<Tensor<Dtype>>(in_shape));
		//data.push_back(std::make_shared<Tensor<Dtype>>(1, inShape3DSize * outShape3DSize, 1, 1));
		//normal_distribution_init<Dtype>((Dtype *)data[1]->getCpuData(), data[1]->getSize()[tind::e4D], 0.0f, 0.1f);

		gradient_.push_back(std::make_shared<Tensor<Dtype>>(1, inShape3DSize * outShape3DSize, 1, 1));
		dlex_set<Dtype>((Dtype *)gradient_[0]->getCpuData(), gradient_[0]->getSize()[tind::e4D], 0.0f);

		if (param_.blas_enable)
		{
			//data.push_back(std::make_shared<Tensor<Dtype>>(1, out_shape[1], 1, 1));
			//dlex_set<Dtype>((Dtype *)data[2]->getCpuData(), data[2]->getSize()[tind::e4D], 0.0f);

			gradient_.push_back(std::make_shared<Tensor<Dtype>>(1, out_shape[1], 1, 1));
			dlex_set<Dtype>((Dtype *)gradient_[1]->getCpuData(), gradient_[1]->getSize()[tind::e4D], 0.0f);
		}
		return 0;
	}

	template <typename Dtype>
	void InnerProductOp<Dtype>::forward(const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev, const std::vector<std::shared_ptr<Tensor<Dtype>>> &next)
	{
		//printf("InnerProductOp: prev = %d, next = %d\n", prev.size(), next.size());
		const std::vector<int> prev_data_size = prev[0]->getSize();
		const std::vector<int> next_data_size = next[0]->getSize();

		//printf("into innerProduct forward:(%d, %d), (%d, %d)\n", prev3DSize, next3DSize, prev[1]->get4DSize(), prev[2]->get4DSize());

		const Dtype* prev_data = (Dtype *)prev[0]->getCpuData();
		Dtype* next_data = (Dtype *)next[0]->getCpuData();
		const Dtype* weight_data = (Dtype *)prev[1]->getCpuData();
		const Dtype* bias_data = param_.blas_enable ? (Dtype *)prev[2]->getCpuData() : nullptr;

		//for (int i = 0; i < prev[2]->get4DSize(); i++)
		//{
		//	printf("%f, ", bias_data[i]);
		//}

		auto worker = [&](const int start, const int stop){
			gemm(false, true, stop - start, next_data_size[tind::e3D], prev_data_size[tind::e3D], 1.0, prev_data + start * prev_data_size[tind::e3D], weight_data, 0.0, next_data + start * next_data_size[tind::e3D]);
		};
		//dispatch_worker(worker,prev_size.number);
		worker(0, prev[0]->getShape()[tind::eNum]);

		if (param_.blas_enable)
			add_bias(prev[0]->getShape()[tind::eNum], next_data_size[tind::e3D], bias_data, next_data);

		//// The method for adding bias to data in caffe
		//Dtype *bias_multiplier = (Dtype *)malloc(sizeof(Dtype) * prev[0]->getShape()[tind::eNum]);
		//for (int i = 0; i < prev[0]->getShape()[tind::eNum]; i++)
		//	bias_multiplier[i] = 1;
		//auto worker2 = [&](const int start, const int stop){
		//	gemm(false, false, stop - start, next_data_size[tind::e3D], 1, 1.0, bias_multiplier, bias_data, 1.0, next_data + start * next_data_size[tind::e3D]);
		//};
		//worker2(0, prev[0]->getShape()[0]);

	}

	template <typename Dtype>
	void InnerProductOp<Dtype>::backward(const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev, const std::vector<std::shared_ptr<Tensor<Dtype>>> &next,
		const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev_diff, const std::vector<std::shared_ptr<Tensor<Dtype>>> &next_diff)
	{
		const Dtype* prev_data = (Dtype*)prev[0]->getCpuData();
		const Dtype* next_data = (Dtype*)next[0]->getCpuData();
		Dtype* prev_diff_data = (Dtype*)prev_diff[0]->getCpuData();
		const Dtype* next_diff_data = (Dtype*)next_diff[0]->getCpuData();
		const Dtype* weight_data = (Dtype*)prev[1]->getCpuData();
		//const Dtype* bias_data = param_.blas_enable ? (Dtype*)prev[2]->getCpuData() : nullptr;

		const std::vector<int> prev_diff_size = prev_diff[0]->getSize();
		const std::vector<int> prev_data_size = prev[0]->getSize();
		const std::vector<int> next_diff_size = next_diff[0]->getSize();
		const std::vector<int> next_data_size = next[0]->getSize();

		const std::vector<int> weight_size = prev[1]->getSize();

		const std::vector<int> prev_data_shape = prev[0]->getShape();

		const std::vector<int> next_data_shape = next[0]->getShape();
		//const std::vector<int> next_diff_shape = next_diff[0]->getShape();

		if (next_data_shape[tind::eHeight] != 1 || next_data_shape[tind::eWidth] != 1)
		{
			DLOG_ERR("[ InnerProductOp::backward ]: using channels as label only, height and width shoule be 1 \n");
			return;
		}
		if (weight_size[tind::e4D] != prev_data_size[tind::e3D] * next_data_size[2])
		{
			DLOG_ERR("[ InnerProductOp::backward ]: weight_size is invalidate!\n");
			return;
		}
		if (param_.blas_enable)
		{
			if (prev[2]->getSize()[tind::e4D] != next_data_size[tind::e3D])
			{
				DLOG_ERR("[ InnerProductOp::backward ]: bias size is invalidate!\n");
				return;
			}
		}
		if (prev_diff_size[tind::e4D] != prev_data_size[tind::e4D])
		{
			DLOG_ERR("[ InnerProductOp::backward ]: the size of prev_diff and prev must be equal\n");
			return;
		}

		////////////////////////////////////////////////////////////////////////////////////////
		//update prev_diff
		// prev_diff(num, in3DSize) = next_diff(num, hidden_num) * weight(hidden_num, in3DSize)
		// -> prev_diff(num, prev_diff_size[tind::e3D]) = next_diff(num, next_diff_size[tind::e3D]) * weight(next_diff_size[tind::e3D], in3DSize)
		auto worker = [&](const int start, const int stop){
			gemm(false, false, stop - start, prev_diff_size[tind::e3D], next_diff_size[tind::e3D], 
				1.0, next_diff_data + start * next_diff_size[tind::e3D], weight_data, 
				0.0, prev_diff_data + start * prev_diff_size[tind::e3D]);
		};
		//dispatch_worker(worker, prev_size.number);
		worker(0, prev_data_shape[tind::eNum]);

		
		////////////////////////////////////////////////////////////////////////////
		//update this layer's param
		//get weight gradient
		Dtype* weight_gradient_data = (Dtype *)gradient_[0]->getCpuData();

		// next_diff(num, hidden_num) -> next_diff'(hidden_num, num)
		// O(M,N) = weightGradient(hidden_num, in3DSize) = next_diff'(hidden_num, num) * prev_data(num, in3DSize)
		// -> M=hidden_num, N=in3DSize, K=num
		auto worker2 = [&](const int start, const int stop){
			gemm(true, false, next_diff_size[tind::e3D], prev_data_size[tind::e3D], prev_data_shape[tind::eNum],
				1.0, next_diff_data, prev_data,
				1.0, weight_gradient_data);	//1.0
		};
		//dispatch_worker(worker, prev_size.number);
		worker2(0, prev_data_shape[tind::eNum]);

		//div by batch size
		div_inplace(weight_gradient_data, (Dtype)next_data_shape[tind::eNum], weight_size[tind::e4D]);

		////////////////////////////////////////////////////////////////////////
		//update bias
		if (param_.blas_enable)
		{
			//get bias diff	
			Dtype* bias_gradient_data = (Dtype *)gradient_[1]->getCpuData();
			const std::vector<int> biasGradSize = gradient_[1]->getSize();

			gradient_[1]->setZero();
			backward_bias(next_data_shape[tind::eNum], biasGradSize[tind::e3D], next_diff_data, bias_gradient_data);

			//div by batch size
			div_inplace(bias_gradient_data, (Dtype)next_data_shape[tind::eNum], biasGradSize[tind::e4D]);
		}

	}

	INSTANTIATE_CLASS(InnerProductOp);

}//namespace