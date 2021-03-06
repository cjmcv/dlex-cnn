////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  Train with mnist data.
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#ifndef DLEX_TEST_MNIST_TRAIN_HPP_
#define DLEX_TEST_MNIST_TRAIN_HPP_
//
#include <iostream>
#include <cassert>
#include "dlex_cnn.h"
#include "mnist_loader.h"
#include "trainer/network_creator.h"

namespace dlex_cnn {

class MnistTrainTest {
public:
  MnistTrainTest();
  virtual ~MnistTrainTest() {};
	
public:		
  void Train();
  void TrainWithPrefetcher();
  void Test(const std::string& model_file_path, const int iter);

  void StartPrefetchData(NetWork<float> &network);
  static bool LoadBatch(void *instant_ptr, std::pair < std::shared_ptr<Tensor<float>>, std::shared_ptr<Tensor<float>> > *tensor_pair);

private:
  bool LoadMnistData(tind::Phase phase);
  bool ReleaseMnistData();
  bool FetchBatchData(const std::vector<std::pair<dlex_cnn::IMAGE_DATUM, char>> &train_data,
    std::shared_ptr<dlex_cnn::Tensor<float>> input_data_tensor,
    std::shared_ptr<dlex_cnn::Tensor<float>> label_data_tensor,
    const int offset, const int length);

  std::shared_ptr<dlex_cnn::Tensor<float>> ConvertLabelToTensor(const std::vector< std::pair<dlex_cnn::IMAGE_DATUM, char> > &test_data, const int start, const int len);
  std::shared_ptr<dlex_cnn::Tensor<float>> ConvertVectorToTensor(const std::vector< std::pair<dlex_cnn::IMAGE_DATUM, char> > &test_data, const int start, const int len);
  uint8_t getMaxIdxInArray(const float* start, const float* stop);
  std::pair<float, float> testInTrain(dlex_cnn::NetWork<float>& network, const int batch, const std::vector< std::pair<dlex_cnn::IMAGE_DATUM, char> > &test_data);


private:
  // Super parameters.
  float learning_rate_;
  float decay_rate_;
  float min_learning_rate_;
  int test_after_batches_;
  int max_batches_;
  int batch_size_;
  int lr_setp_;
  int save_iter_;

  // Data parameters.
  int class_num_;
  int channels_;
  int width_;
  int height_;
  int data_size_4D_;

  // Data file path.
  std::string train_images_file_;
  std::string train_labels_file_;

  std::string test_images_file_;
  std::string test_labels_file_;

  // Others.
  int prefetch_batch_idx_;
  std::string model_saved_path_;

  std::vector<std::pair<dlex_cnn::IMAGE_DATUM, char>> train_data_;
  std::vector<std::pair<dlex_cnn::IMAGE_DATUM, char>> validate_data_;
};
}

void MnistTrain();
void MnistTest();

#endif
