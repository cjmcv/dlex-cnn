////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  A factory for operators.
//          allows users to create an operator only by string.
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#ifndef DLEX_CLASSFACTORY_HPP_
#define DLEX_CLASSFACTORY_HPP_

#include <iostream>
#include <string>
#include <map>
#include <memory> 

#include "operator/operator_base.h"

namespace dlex_cnn { 
	
#define GET_OPTIMIZER(type, Dtype, lr)                                             \
	std::shared_ptr< dlex_cnn::Optimizer<Dtype> >(new dlex_cnn::##type##<Dtype>(lr));

#define GET_OP_CLASS(type, Dtype)                                             \
    std::shared_ptr< dlex_cnn::Op<Dtype> >(new dlex_cnn::##type##Op<Dtype>());

#define GET_OP_PARAM_CLASS(type, param_str)                                    \
    std::shared_ptr< dlex_cnn::Op<Dtype> >(new dlex_cnn::##type##OpParam(param_str));

template <typename Dtype>
class OpFactory {
  typedef std::shared_ptr<dlex_cnn::Op<Dtype>>(*opCreator)();
public:
  ~OpFactory() {};

  // Create operator instance according to the name that has been registered.
  std::shared_ptr< dlex_cnn::Op<Dtype> > CreateOpByType(std::string type) {
    typename std::map<std::string, opCreator>::iterator it = op_creator_map_.find(type);
    if (it == op_creator_map_.end())
      return NULL;

    opCreator getOpFunc = it->second;
    if (!getOpFunc)
      return NULL;

    return getOpFunc();
  }

  std::shared_ptr< dlex_cnn::Op<Dtype> > CreateOpByType(std::string type, std::string param_str) {
    typename std::map<std::string, opCreator>::iterator it = op_creator_map_.find(type);
    if (it == op_creator_map_.end())
      return NULL;

    opCreator getOpFunc = it->second;
    if (!getOpFunc)
      return NULL;

    return getOpFunc(param_str);
  }

  // Registerer, set the mapping relation between operator's class name and it's specific pointer function.
  int RegisterOpClass(std::string type, opCreator getOpFunc) {
    if (op_creator_map_.count(type) != 0){
      std::cout << "Op type :" << type << " already registered.";
      return -1;
    }
    op_creator_map_[type] = getOpFunc;
    return 0;
  }

  // Singleton mode. Only one OpFactory exist.
  static OpFactory& GetInstance(){
    static OpFactory factory;
    return factory;
  }

private:
  OpFactory() {};
  typename std::map<std::string, opCreator> op_creator_map_;
};

}	//namespace dlex_cnn
#endif
