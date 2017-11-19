////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief 
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#ifndef DLEX_NODE_HPP_
#define DLEX_NODE_HPP_

//#include <iostream>
#include <vector>
#include <memory>
#include <stdlib.h>

//#include "dlex_datatype.h"
#include "tensor.h"
#include "operator/operator_base.h"
#include "operator/operator_hybrid.h"
#include "util/op_factory.h"

namespace dlex_cnn
{
	namespace tind
	{
		enum Phase { Train, Test };
	}

	template <typename Dtype>
	class Node
	{
	public:
		explicit Node();
		virtual ~Node();

		// get members
		inline int getPhase() { return phase_; }
		inline int getIndex() { return index_; }
		inline const std::string &getName() { return name_; }		
		inline const std::vector<int> &getInputShape() { return input_shape_; }
		inline const std::vector<int> &getOutputShape() { return output_shape_; }
		inline const std::vector<int> &getInputIdx() { return inputs_index_; }
		inline const std::vector<std::string> &getInputName() { return inputs_name_; }
		inline const std::vector<int> &getOutputIdx() { return outputs_index_; }
		inline const std::vector<std::string> &getOutputName() { return outputs_name_; }
		inline const std::shared_ptr<Op<Dtype>> getInteOp() { return inte_ops_; }
		inline const std::vector<std::shared_ptr<Tensor<Dtype>>> &getDataVec() { return cpu_data_; }
		inline const std::string getOpParamBufStr() { 
			op_param_str_ = inte_ops_->genOpParamStr();
			return op_param_str_;
		}

		// set members
		inline void setPhase(int phase) { phase_ = phase; }
		inline void setIndex(int index) { index_ = index; }
		inline void setName(std::string name) { name_ = name; }
		inline void setInputShape(std::vector<int> input_shape) { input_shape_ = input_shape; }
		inline void addInputIdx(int idx) { inputs_index_.push_back(idx); }
		inline void addInputName(std::string name) { inputs_name_.push_back(name); }
		inline void addOutputIdx(int idx) { outputs_index_.push_back(idx); }
		inline void addOutputName(std::string name) { outputs_name_.push_back(name); }
		inline void addSubOps(std::shared_ptr<Op<Dtype>> sub_op) { sub_ops_.push_back(sub_op); }
		inline void setOpParamBufStr(std::string str) { op_param_str_ = str; };
		
		// Allocate memory buffer for op, mainly includes diff_ and gradient_
		inline int initOp() {
			int ret = inte_ops_->allocOpBuf4Train(input_shape_, output_shape_);
			return ret;
		}

		// Allocate memory buffer for node according to inte_ops_
		inline int initNode() {
			int ret = inte_ops_->allocBuf4Node(input_shape_, output_shape_, cpu_data_);
			return ret;
		}

		inline int inferOutShape() {
			int ret = inte_ops_->inferOutShape(input_shape_, output_shape_);
			return ret;
		}

		// Includes input_shape_, output_shape_ and the size of cpu_data_
		int resetDataSize(int index, const std::vector<int> &shape);

		// get the mapping relationship between a hybrid operation and serval operations
		int hybridOpMap(std::string &inteOpType);

		// Infer and generate inte_ops_ on the basis of sub_ops_ and phase_
		int inferInteOp();

		// paramaters reader and loader	
		int writeNode2Text(FILE *fp);
		int writeNode2Bin(FILE *fp);
		int writeWB2Bin(FILE *fp);

		int writeBin2Node(FILE *fp);
		int readBin2WB(FILE *fp);
		void serializeFromString(const std::string content);

		// 网络结构: 可见，输入输出都行，也可在内部用字符串写网络结构
		// name, index, in_idx_count, in_idx, （out_idx_count, out_idx,全部加载完毕后，再数，建立） in_shape（根据前一个node决定）, out_shape（推断）
		// opParam

		// blob
		// cpu_data_.size(), length, cpu_data_[0], length, [1], (length, [2])

	private:
		int phase_;

		// node index, for searching node in graph 
		int index_;
		// node name, as a unique symbol in the whole network
		std::string name_;
		
		std::vector<int> input_shape_;
		// output shape is equal to the size of this node data[0]
		std::vector<int> output_shape_;
		
		// sub-operators
		std::vector<std::shared_ptr<Op<Dtype>>> sub_ops_;

		// final operator for this node in this phase.
		// it will be one of the sub-operators or be assemed by some of those sub-operators
		std::shared_ptr<Op<Dtype>> inte_ops_;
		std::string op_param_str_ = "";

		std::vector<int> inputs_index_;
		std::vector<std::string> inputs_name_;
		std::vector<int> outputs_index_;
		std::vector<std::string> outputs_name_;

		// include in_data/weight/blas
		std::vector<std::shared_ptr<Tensor<Dtype>>> cpu_data_;	
		//std::vector<std::shared_ptr<Tensor<float>>> gradients_;	//include weight_gra/blas_gra

	};
}
#endif //DLEX_NODE_HPP_