////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#ifndef DLEX_CLASSFACTORY_HPP_
#define DLEX_CLASSFACTORY_HPP_

#include <iostream>
#include <string>
#include <map>
#include <memory> 

#include "operator/operator_base.h"

namespace dlex_cnn
{ 
	
#define GET_OPTIMIZER(type, Dtype, lr)                                             \
	std::shared_ptr< dlex_cnn::Optimizer<Dtype> >(new dlex_cnn::##type##<Dtype>(lr));

#define GET_OP_CLASS(type, Dtype)                                             \
    std::shared_ptr< dlex_cnn::Op<Dtype> >(new dlex_cnn::##type##Op<Dtype>());

#define GET_OP_PARAM_CLASS(type, param_str)                                    \
    std::shared_ptr< dlex_cnn::Op<Dtype> >(new dlex_cnn::##type##OpParam(param_str));

template <typename Dtype>
class OpFactory
{
	typedef std::shared_ptr<dlex_cnn::Op<Dtype>> (*opCreator)();
public:
	~OpFactory() {};

	// Create operator instance according to the name that has been registered.
	std::shared_ptr< dlex_cnn::Op<Dtype> > createOpByType(std::string type)
	{
		std::map<std::string, opCreator>::iterator it = op_creator_map_.find(type);
		if (it == op_creator_map_.end())
			return NULL;

		opCreator getOpFunc = it->second;
		if (!getOpFunc)
			return NULL;

		return getOpFunc();
	}

	std::shared_ptr< dlex_cnn::Op<Dtype> > createOpByType(std::string type, std::string param_str)
	{
		std::map<std::string, opCreator>::iterator it = op_creator_map_.find(type);
		if (it == op_creator_map_.end())
			return NULL;

		opCreator getOpFunc = it->second;
		if (!getOpFunc)
			return NULL;

		return getOpFunc(param_str);
	}

	// Registerer, set the mapping relation between operator class name and it's specific pointer function.
	int registerOpClass(std::string type, opCreator getOpFunc)
	{
		if (op_creator_map_.count(type) != 0)
		{
			std::cout << "Op type :" << type << " already registered.";
			return -1;
		}
		op_creator_map_[type] = getOpFunc;
		return 0;
	}

	// singleton mode
	static OpFactory& getInstance()
	{
		static OpFactory factory;
		return factory;
	}

private:
	OpFactory() {};
	std::map<std::string, opCreator> op_creator_map_;
};

}	//namespace dlex_cnn
#endif