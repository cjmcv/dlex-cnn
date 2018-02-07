////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  Mainly for providing the device mode.
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#include "task.h"

namespace dlex_cnn {
static Task *gTask = NULL;

Task::Task() {
  device_mode_ = tind::CPU;
}

Task::~Task() {}

Task& Task::Get() {
  if (gTask == NULL)
    gTask = new Task();

  return *gTask;
}
}