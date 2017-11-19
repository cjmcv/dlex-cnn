////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#ifndef DLEX_OP_OPTIMIZER_HPP_
#define DLEX_OP_OPTIMIZER_HPP_

#include <vector>
#include "Configure.h"
#include "node.h"

namespace dlex_cnn
{
	template <typename Dtype>
	class Optimizer
	{
	public:
		Optimizer() {}
		Optimizer(const float lr) :lr_(lr){}
		void setLearningRate(const float lr) { lr_ = lr; };
		inline float getLearningRate() { return lr_; };
		virtual void update(std::shared_ptr<Node<Dtype>> node) = 0;
		virtual inline const std::string &getOptName() { return ""; };
		static int getOptimizerByStr(std::string &optName, std::shared_ptr<Optimizer<Dtype>> &opt);
	protected:
		float lr_ = 0.1f;
	};

	template <typename Dtype>
	class SGD : public Optimizer<Dtype>
	{
	public:
		SGD() {}
		SGD(const float lr) :Optimizer(lr){}		
		virtual inline const std::string &getOptName() override { return opt_name_; };
		virtual void update(std::shared_ptr<Node<Dtype>> node) override;
	private:
		std::string opt_name_ = "SGD";
	};
}
#endif