////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#include "dlex_task.h"

namespace dlex_cnn
{
	static Task *gTask = NULL;

	Task::Task()
	{
		
	}

	Task::~Task()
	{

	}

	//Task& Task::Get()
	//{
	//	if (gTask == NULL)
	//		gTask = new Task();

	//	return *gTask;
	//}
}