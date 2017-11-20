////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#include "mnist_train.h"

namespace dlex_cnn
{
	MnistTrainTest::MnistTrainTest()
	{
		class_num_ = 10;
		train_images_file_ = "../../res/mnist_data/train-images.idx3-ubyte";
		train_labels_file_ = "../../res/mnist_data/train-labels.idx1-ubyte";

		test_images_file_ = "../../res/mnist_data/t10k-images.idx3-ubyte";
		test_labels_file_ = "../../res/mnist_data/t10k-labels.idx1-ubyte";

		printf("MnistTrainTest constructed.\n");
	}

	bool MnistTrainTest::loadMnistData(tind::Phase phase)
	{
		// Set super parameter = tind::Train;
		MnistLoader loader;

		float crossVal;
		std::string images_file, labels_file;
		if (phase == tind::Train)
		{
			images_file = train_images_file_;
			labels_file = train_labels_file_;

			crossVal = 0.9;
		}
		else
		{
			images_file = test_images_file_;
			labels_file = test_labels_file_;

			crossVal = 0.0;
		}
		
		bool success = false;
		//load train images
		printf("loading training data...\n");
		std::vector<dlex_cnn::IMAGE_DATUM> images;
		success = loader.load_mnist_images(images_file, images);
		assert(success && images.size() > 0);
		//load train labels
		std::vector<char> labels;
		success = loader.load_mnist_labels(labels_file, labels);
		assert(success && labels.size() > 0);
		assert(images.size() == labels.size());

		// train_data_
		for (int i = 0; i < images.size()*crossVal; i++)
			train_data_.push_back(std::make_pair(images[i], labels[i]));
		std::random_shuffle(train_data_.begin(), train_data_.end());

		// validate_data_
		for (int i = images.size()*crossVal; i < images.size(); i++)
			validate_data_.push_back(std::make_pair(images[i], labels[i]));
		std::random_shuffle(validate_data_.begin(), validate_data_.end());

		return true;
	}

	bool MnistTrainTest::releaseMnistData()
	{
		for (int i = 0; i < train_data_.size(); i++)
			delete train_data_[i].first.pdata;

		for (int i = 0; i < validate_data_.size(); i++)
			delete validate_data_[i].first.pdata;

		return true;
	}

	bool MnistTrainTest::fetchBatchData(const std::vector<std::pair<dlex_cnn::IMAGE_DATUM, char>> &train_data_,
		std::shared_ptr<dlex_cnn::Tensor<float>> inputDataTensor,
		std::shared_ptr<dlex_cnn::Tensor<float>> labelDataTensor,
		const int offset, const int length)
	{
		assert(inputDataTensor->getShape()[0] == labelDataTensor->getShape()[0]);
		if (offset >= train_data_.size())
		{
			return false;
		}
		int actualEndPos = offset + length;
		if (actualEndPos > train_data_.size())
		{
			//image data
			//auto inputDataSize = inputDataTensor->getSize();
			inputDataTensor->getShape()[0] = train_data_.size() - offset;
			actualEndPos = offset + inputDataTensor->getShape()[0];
			inputDataTensor.reset(new dlex_cnn::Tensor<float>(inputDataTensor->getShape()));
			//label data
			//auto labelDataSize = labelDataTensor->getSize();
			labelDataTensor->getShape()[0] = inputDataTensor->getShape()[0];
			labelDataTensor.reset(new dlex_cnn::Tensor<float>(labelDataTensor->getShape()));
		}
		//printf("ready to copy\n");
		//copy
		const int sizePerImage = inputDataTensor->getSize()[dlex_cnn::tind::e3D];
		const int sizePerLabel = labelDataTensor->getSize()[dlex_cnn::tind::e3D];
		assert(sizePerImage == train_data_[0].first.channels*train_data_[0].first.width*train_data_[0].first.height);
		//scale to 0.0f~1.0f
		const float scaleRate = 1.0f / 255.0f;
		for (int i = offset; i < actualEndPos; i++)
		{
			//image data
			float* inputData = (float *)inputDataTensor->getData() + (i - offset)*sizePerImage;//*sizeof(float)
			const uint8_t* imageData = train_data_[i].first.pdata;
			for (int j = 0; j < sizePerImage; j++)
			{
				inputData[j] = (float)imageData[j] * scaleRate;
			}
			//label data
			float* labelData = (float *)labelDataTensor->getData() + (i - offset)*sizePerLabel;//*sizeof(float)
			const uint8_t label = train_data_[i].second;
			for (int j = 0; j < sizePerLabel; j++)
			{
				labelData[j] = label;
			}
		}
		//printf("finish fetch\n");
		return true;
	}

	std::shared_ptr<dlex_cnn::Tensor<float>> MnistTrainTest::convertLabelToTensor(const std::vector< std::pair<dlex_cnn::IMAGE_DATUM, char> > &test_data, const int start, const int len)
	{
		assert(test_data.size() > 0);
		const int number = len;
		const int sizePerLabel = 1;

		std::shared_ptr<dlex_cnn::Tensor<float>> result(new dlex_cnn::Tensor<float>(number, sizePerLabel, 1, 1));
		for (int i = start; i < start + len; i++)
		{
			float* labelData = (float *)result->getData() + (i - start)*sizePerLabel;//*sizeof(float)
			const uint8_t label = test_data[i].second;
			for (int j = 0; j < sizePerLabel; j++)
			{
				labelData[j] = label;
			}
		}
		return result;
	}

	std::shared_ptr<dlex_cnn::Tensor<float>> MnistTrainTest::convertVectorToTensor(const std::vector< std::pair<dlex_cnn::IMAGE_DATUM, char> > &test_data, const int start, const int len)
	{
		assert(test_data.size() > 0);
		const int number = len;
		const int channel = test_data[0].first.channels;
		const int width = test_data[0].first.width;
		const int height = test_data[0].first.height;
		const int sizePerImage = channel*width*height;
		const float scaleRate = 1.0f / 255.0f;
		//std::shared_ptr<dlex_cnn::DataTensor> result(new dlex_cnn::DataTensor(dlex_cnn::DataSize(number, channel, width, height)));
		std::shared_ptr<dlex_cnn::Tensor<float>> result(new dlex_cnn::Tensor<float>(number, channel, height, width));
		for (int i = start; i < start + len; i++)
		{
			//image data
			float* inputData = (float *)result->getData() + (i - start)*sizePerImage;
			const uint8_t* imageData = test_data[i].first.pdata;
			for (int j = 0; j < sizePerImage; j++)
			{
				inputData[j] = (float)imageData[j] * scaleRate;
			}
		}
		return result;
	}

	uint8_t MnistTrainTest::getMaxIdxInArray(const float* start, const float* stop)
	{
		assert(start && stop && stop >= start);
		int result = 0;
		int len = stop - start;
		for (int i = 0; i < len; i++)
			if (start[i] >= start[result])
				result = i;

		return (uint8_t)result;
	}

	std::pair<float, float> MnistTrainTest::test_in_train(dlex_cnn::NetWork<float>& network, const int batch, const std::vector< std::pair<dlex_cnn::IMAGE_DATUM, char> > &test_data)
	{
		//printf("into test: batch = %d, %d, %d\n", batch, test_labels.size(), test_images.size());
		assert(test_data.size()>0);
		int correctCount = 0;
		float loss = 0.0f;
		int batchs = 0;
		for (int i = 0; i < test_data.size(); i += batch, batchs++)//test_data.size()
		{
			const int start = i;
			int tlabelSize = test_data.size();
			const int len = std::min(tlabelSize - start, batch);
			//printf("len = %d\n", len);
			const std::shared_ptr<dlex_cnn::Tensor<float>> inputDataTensor = convertVectorToTensor(test_data, start, len);	//data -> test_data.first
			const std::shared_ptr<dlex_cnn::Tensor<float>> labelDataTensor = convertLabelToTensor(test_data, start, len);	//label -> test_data.second

			network.testBatch(inputDataTensor, labelDataTensor);//
			std::shared_ptr<dlex_cnn::Tensor<float>> probDataTensor;
			network.getNodeData("output", probDataTensor);

			//float *pdata = (float *)probDataTensor->getData();
			//for (int i = 0; i < probDataTensor->getSize()[dlex_cnn::tind::e4D]; i++)
			//{
			//	printf("%f, ", pdata[i]);
			//}
			//network.graph_->getNodeData("output", std::vector<std::shared_ptr<Tensor<Dtype>>> &cpuData);

			//printf("ready to get loss\n");
			////get loss
			//const float batch_loss = 0; // network.getLoss(labelDataTensor, probDataTensor);
			//loss = dlex_cnn::moving_average(loss, batchs + 1, batch_loss);
			//printf("loss = %f\n", loss);
			////printf("loss: %f, %f\n", batch_loss, loss);

			//printf("ready to get acc\n");
			const int labelSize = probDataTensor->getSize()[dlex_cnn::tind::e3D];
			const float* probData = (float *)probDataTensor->getData();
			for (int j = 0; j < len; j++)
			{
				const uint8_t stdProb = test_data[i + j].second;
				const uint8_t testProb = getMaxIdxInArray(probData + j*labelSize, probData + (j + 1) * labelSize);
				if (stdProb == testProb)
				{
					correctCount++;
				}
			}
			//printf("finish\n");
		}

		const float accuracy = (float)correctCount / (float)test_data.size();
		return std::pair<float, float>(accuracy, 0);	//loss
	}

	dlex_cnn::NetWork<float> MnistTrainTest::buildConvNet(const int batch, const int channels, const int height, const int width)
	{
		printf("start building net\n");
		//registerOpClass();

		dlex_cnn::NetWork<float> network;
		network.netWorkInit("netA");
		//network.setInputSize(batch, channels, height, width);

		//input data layer
		//add_input_layer2(network);

		//////std::string nodeName = "input";
		//////dlex_cnn::InputParam inputParam;
		//////std::shared_ptr<dlex_cnn::Op<float>> input = std::make_shared<dlex_cnn::InputOp<float>>(*(dlex_cnn::CreateOp<float>(inputParam)));
		//////network.addNode(nodeName, input);
	
		std::shared_ptr<dlex_cnn::Op<float>> input_s = dlex_cnn::OpFactory<float>::getInstance().createOpByType("Input");
		assert(input_s != NULL);
		dlex_cnn::InputOpParam *inputParam = new dlex_cnn::InputOpParam(batch, channels, height, width);
		dynamic_cast<dlex_cnn::InputOp<float> *>(input_s.get())->setOpParam(*inputParam);
		std::vector < std::shared_ptr<dlex_cnn::Op<float>> > input;

		input.push_back(input_s);
		std::string nodeName_input = "input";
		network.addNode(nodeName_input, input);

		///////////////////////////////////////////////////
		// conv1
		std::shared_ptr<dlex_cnn::Op<float>> conv1_s = dlex_cnn::OpFactory<float>::getInstance().createOpByType("Convolution");
		dlex_cnn::ConvolutionOpParam ConvolutionParam;
		ConvolutionParam.blas_enable = true;
		ConvolutionParam.kernel_num = 6;
		ConvolutionParam.kernel_h = 3;
		ConvolutionParam.kernel_w = 3;
		ConvolutionParam.pad_h = 1;
		ConvolutionParam.pad_w = 1;
		ConvolutionParam.stride_h = 1;
		ConvolutionParam.stride_w = 1;
		ConvolutionParam.dilation_h = 1;
		ConvolutionParam.dilation_w = 1;

		dynamic_cast<dlex_cnn::ConvolutionOp<float> *>(conv1_s.get())->setOpParam(ConvolutionParam);
		std::string nodeName_conv1 = "conv1";
		std::vector<std::shared_ptr<dlex_cnn::Op<float>>> conv1;
		conv1.push_back(conv1_s);
		std::vector<std::string> inNodeName_conv1;
		inNodeName_conv1.push_back(nodeName_input);
		network.addNode(nodeName_conv1, conv1, inNodeName_conv1);

		///////////////////////////////////////////////////
		// deconv1
		std::shared_ptr<dlex_cnn::Op<float>> deconv1_s = dlex_cnn::OpFactory<float>::getInstance().createOpByType("Deconvolution");
		dlex_cnn::DeconvolutionOpParam DeconvolutionParam;
		DeconvolutionParam.blas_enable = true;
		DeconvolutionParam.kernel_channels = 6;
		DeconvolutionParam.kernel_h = 3;
		DeconvolutionParam.kernel_w = 3;
		DeconvolutionParam.pad_h = 1;
		DeconvolutionParam.pad_w = 1;
		DeconvolutionParam.stride_h = 1;
		DeconvolutionParam.stride_w = 1;
		DeconvolutionParam.dilation_h = 1;
		DeconvolutionParam.dilation_w = 1;

		dynamic_cast<dlex_cnn::DeconvolutionOp<float> *>(deconv1_s.get())->setOpParam(DeconvolutionParam);
		std::string nodeName_deconv1 = "deconv1";
		std::vector<std::shared_ptr<dlex_cnn::Op<float>>> deconv1;
		deconv1.push_back(deconv1_s);
		std::vector<std::string> inNodeName_deconv1;
		inNodeName_deconv1.push_back(nodeName_conv1);
		network.addNode(nodeName_deconv1, deconv1, inNodeName_deconv1);

		////////////////////////////////////////////
		// activation 1
		std::shared_ptr<dlex_cnn::Op<float>> relu1_s = dlex_cnn::OpFactory<float>::getInstance().createOpByType("Activation");
		dlex_cnn::ActivationOpParam ReLUParam1;
		ReLUParam1.activationType = dlex_cnn::tind::Activation::eReLU;

		dynamic_cast<dlex_cnn::ActivationOp<float> *>(relu1_s.get())->setOpParam(ReLUParam1);
		std::string nodeName_relu1 = "relu1";
		std::vector<std::shared_ptr<dlex_cnn::Op<float>>> relu1;
		relu1.push_back(relu1_s);
		std::vector<std::string> inNodeName_relu1;
		inNodeName_relu1.push_back(nodeName_deconv1);
		network.addNode(nodeName_relu1, relu1, inNodeName_relu1);


		////////////////////////////////////////////
		// pool 1
		std::shared_ptr<dlex_cnn::Op<float>> pool_s = dlex_cnn::OpFactory<float>::getInstance().createOpByType("Pooling");
		dlex_cnn::PoolingOpParam PoolingParam;
		PoolingParam.global_pooling = false;
		PoolingParam.kernel_h = 3;
		PoolingParam.kernel_w = 3;
		PoolingParam.pad_h = 1;
		PoolingParam.pad_w = 1;
		PoolingParam.stride_h = 1;
		PoolingParam.stride_w = 1;
		PoolingParam.poolingType = dlex_cnn::tind::eMAX;
		dynamic_cast<dlex_cnn::PoolingOp<float> *>(pool_s.get())->setOpParam(PoolingParam);

		std::string nodeName_pool = "pool1";
		std::vector<std::shared_ptr<dlex_cnn::Op<float>>> pool;
		pool.push_back(pool_s);
		std::vector<std::string> inNodeName_pool;
		inNodeName_pool.push_back(nodeName_relu1);
		network.addNode(nodeName_pool, pool, inNodeName_pool);
		/////////////////////////////////////////////////////
		//// conv2
		//std::shared_ptr<dlex_cnn::Op<float>> conv2_s = dlex_cnn::OpFactory<float>::getInstance().createOpByType("Convolution");
		//dlex_cnn::ConvolutionOpParam ConvolutionParam2;
		//ConvolutionParam2.blas_enable = true;
		//ConvolutionParam2.kernel_num = 6;
		//ConvolutionParam2.kernel_h = 3;
		//ConvolutionParam2.kernel_w = 3;
		//ConvolutionParam2.pad_h = 1;
		//ConvolutionParam2.pad_w = 1;
		//ConvolutionParam2.stride_h = 1;
		//ConvolutionParam2.stride_w = 1;
		//ConvolutionParam2.dilation_h = 1;
		//ConvolutionParam2.dilation_w = 1;

		//dynamic_cast<dlex_cnn::ConvolutionOp<float> *>(conv2_s.get())->setOpParam(ConvolutionParam2);
		//std::string nodeName_conv2 = "conv2";
		//std::vector<std::shared_ptr<dlex_cnn::Op<float>>> conv2;
		//conv2.push_back(conv2_s);
		//std::vector<std::string> inNodeName_conv2;
		//inNodeName_conv2.push_back(nodeName_relu1);
		//network.addNode(nodeName_conv2, conv2, inNodeName_conv2);

		//////////////////////////////////////////////
		//// activation2
		//std::shared_ptr<dlex_cnn::Op<float>> relu2_s = dlex_cnn::OpFactory<float>::getInstance().createOpByType("Activation");
		//dlex_cnn::ActivationOpParam ReLUParam2;
		//ReLUParam2.activationType = dlex_cnn::tind::Activation::eReLU;

		//dynamic_cast<dlex_cnn::ActivationOp<float> *>(relu2_s.get())->setOpParam(ReLUParam2);
		//std::string nodeName_relu2 = "relu2";
		//std::vector<std::shared_ptr<dlex_cnn::Op<float>>> relu2;
		//relu2.push_back(relu2_s);
		//std::vector<std::string> inNodeName_relu2;
		//inNodeName_relu2.push_back(nodeName_conv2);
		//network.addNode(nodeName_relu2, relu2, inNodeName_relu2);

		//////convolution layer
		////add_conv_layer(network, 6 ,1);
		////add_active_layer(network);
		//////pooling layer
		////add_pool_layer(network, 6);

		//////convolution layer
		////add_conv_layer(network, 12, 6);
		////add_active_layer(network);
		//////pooling layer
		////add_pool_layer(network, 12);

		//add_fc_layer(network, 256);
	
		//// fc1 - 256
		//std::shared_ptr<dlex_cnn::Op<float>> fc1_s = dlex_cnn::OpFactory<float>::getInstance().createOpByType("InnerProduct");
		//dlex_cnn::InnerProductOpParam innerProductParam;
		//innerProductParam.blas_enable = true;
		//innerProductParam.num_hidden = 256;
		//dynamic_cast<dlex_cnn::InnerProductOp<float> *>(fc1_s.get())->setOpParam(innerProductParam);
		//std::string nodeName_fc1 = "fc1";
		//std::vector<std::shared_ptr<dlex_cnn::Op<float>>> fc1;
		//fc1.push_back(fc1_s);
		//std::vector<std::string> inNodeName_fc1;
		//inNodeName_fc1.push_back(nodeName_relu1);
		//network.addNode(nodeName_fc1, fc1, inNodeName_fc1);

		// fc2 - 512
		std::shared_ptr<dlex_cnn::Op<float>> fc2_s = dlex_cnn::OpFactory<float>::getInstance().createOpByType("InnerProduct");
		dlex_cnn::InnerProductOpParam innerProductParam2;
		innerProductParam2.blas_enable = true;
		innerProductParam2.num_hidden = class_num_;
		dynamic_cast<dlex_cnn::InnerProductOp<float> *>(fc2_s.get())->setOpParam(innerProductParam2);
		std::string nodeName_fc2 = "fc2";
		std::vector<std::shared_ptr<dlex_cnn::Op<float>>> fc2;
		fc2.push_back(fc2_s);
		std::vector<std::string> inNodeName_fc2;
		inNodeName_fc2.push_back(nodeName_pool);
		network.addNode(nodeName_fc2, fc2, inNodeName_fc2);

		// softmax1
		//std::shared_ptr<dlex_cnn::Op<float>> sm_s = dlex_cnn::OpFactory<float>::getInstance().createOpByType("Softmax");
		//dlex_cnn::SoftmaxOpParam softmaxParam;
		//dynamic_cast<dlex_cnn::SoftmaxOp<float> *>(sm_s.get())->setOpParam(softmaxParam);

		//std::shared_ptr<dlex_cnn::Op<float>> cel_s = dlex_cnn::OpFactory<float>::getInstance().createOpByType("CrossEntropyLoss");
		//dlex_cnn::CrossEntropyLossOpParam CELParam;
		//dynamic_cast<dlex_cnn::CrossEntropyLossOp<float> *>(cel_s.get())->setOpParam(CELParam);
		//
		//std::string nodeName_softmax = "softmaxz";
		//std::vector<std::shared_ptr<dlex_cnn::Op<float>>> sm;
		//sm.push_back(sm_s);
		//sm.push_back(cel_s);
		//std::vector<std::string> inNodeName_softmax;
		//inNodeName_softmax.push_back(nodeName_fc2);
		//network.addNode(nodeName_softmax, sm, inNodeName_softmax);

		// softmax2
		std::shared_ptr<dlex_cnn::Op<float>> sm_s = dlex_cnn::OpFactory<float>::getInstance().createOpByType("SoftmaxCrossEntropyLossH");
		dlex_cnn::SoftmaxCrossEntropyLossHOpParam softmaxParam;
		dynamic_cast<dlex_cnn::SoftmaxCrossEntropyLossHOp<float> *>(sm_s.get())->setOpParam(softmaxParam);

		std::string nodeName_softmax = "softmaxz";
		std::vector<std::shared_ptr<dlex_cnn::Op<float>>> sm;
		sm.push_back(sm_s);
		std::vector<std::string> inNodeName_softmax;
		inNodeName_softmax.push_back(nodeName_fc2);
		network.addNode(nodeName_softmax, sm, inNodeName_softmax);

		// output node
		std::shared_ptr<dlex_cnn::Op<float>> output_s = dlex_cnn::OpFactory<float>::getInstance().createOpByType("Output");
		dlex_cnn::OutputOpParam outNodeParam;
		outNodeParam.label_dim = 1;
		dynamic_cast<dlex_cnn::OutputOp<float> *>(output_s.get())->setOpParam(outNodeParam);
		std::vector < std::shared_ptr<dlex_cnn::Op<float>> > output;
		output.push_back(output_s);
		std::string nodeName_output = "output";
		std::vector<std::string> inNodeName_output;
		inNodeName_output.push_back(nodeName_softmax);
		network.addNode(nodeName_output, output, inNodeName_output);

		printf("finish building net\n");

		return network;
	}

	void MnistTrainTest::train()
	{
		loadMnistData(tind::Train);

		float learningRate = 0.1f;
		const float decayRate = 0.8f;
		const float minLearningRate = 0.001f;
		const int testAfterBatches = 10;
		const int maxBatches = 10000;
		int batchSize = 128;
		const int channels = train_data_[0].first.channels;
		const int width = train_data_[0].first.width;
		const int height = train_data_[0].first.height;
		printf("testAfterBatches:%d\n", testAfterBatches);
		printf("learningRate:%f ,decayRate:%f , minLearningRate:%f\n", learningRate, decayRate, minLearningRate);
		printf("channels:%d , width:%d , height:%d\n", channels, width, height);

		printf("construct network begin...\n");
		registerOpClass();

		dlex_cnn::NetWork<float> network(buildConvNet(batchSize, channels, height, width));
		std::string optName = "SGD";
		std::shared_ptr<dlex_cnn::Optimizer<float>> optimizer;
		dlex_cnn::Optimizer<float>::getOptimizerByStr(optName, optimizer);
		network.setOptimizer(optimizer);
		network.setLearningRate(learningRate);

		//dlex_cnn::NetWork<float> network;
		//network.netWorkInit("netA");
		//network.loadStageModel("./", 1);

		printf("construct network done.\n");

		float val_accuracy = 0.0f;
		float train_total_loss = 0.0f;
		int train_batches = 0;
		float val_loss = 0.0f;

		//train
		printf("begin training...\n");
		std::shared_ptr<dlex_cnn::Tensor<float>> inputDataTensor = std::make_shared<dlex_cnn::Tensor<float>>(batchSize, channels, height, width);	//shoule be moved
		std::shared_ptr<dlex_cnn::Tensor<float>> labelDataTensor = std::make_shared<dlex_cnn::Tensor<float>>(batchSize, 1, 1, 1);//class_num_
	
		int savedIter = 59;
		std::string modelSavedPath = "./";

		int lrSetp = 20;
		int epochIdx = 0;

		//before epoch start, shuffle all train data first
		batchSize = 128;	//shoule be changed
		int fetchOffSet = 0;
		for (int batchIdx = 0; batchIdx < maxBatches; batchIdx++)
		{
			//batchSize++;
			printf("batch size = %d\n", batchSize);
			if (inputDataTensor->getShape()[tind::eNum] != batchSize)
			{
				inputDataTensor.reset(new dlex_cnn::Tensor<float>(batchSize, channels, height, width));
				labelDataTensor.reset(new dlex_cnn::Tensor<float>(batchSize, 1, 1, 1));
			}
			printf("start train batch : %d\n", batchIdx);
			if (!fetchBatchData(train_data_, inputDataTensor, labelDataTensor, fetchOffSet*batchSize, batchSize))
			{
				fetchOffSet = 0;
				std::random_shuffle(train_data_.begin(), train_data_.end());
				epochIdx++;
				continue;
			}
			else
				fetchOffSet++;

			printf("finish fetchBatchData\n");

			//network.netWorkShow();


			const float batch_loss = network.trainBatch(inputDataTensor, labelDataTensor);

			train_batches++;
			train_total_loss += batch_loss;
			printf("batch[%d]->train_batch_loss: %f\n", batchIdx, batch_loss);
			if (batchIdx && batchIdx % testAfterBatches == 0)
			{
				network.switchPhase(dlex_cnn::tind::Phase::Test);
				std::tie(val_accuracy, val_loss) = test_in_train(network, 128, validate_data_);	// 要注意最后一个batch为112，会修改掉所有节点的num维度，在下一轮训练过程中修改回来。
				printf("sample : %d/%d , learningRate : %f , train_avg_loss : %f , val_loss : %f , val_accuracy : %.4f%%\n",
					batchIdx*batchSize, train_data_.size(), learningRate, train_total_loss / train_batches, val_loss, val_accuracy*100.0f);

				train_total_loss = 0.0f;
				train_batches = 0;
				network.switchPhase(dlex_cnn::tind::Phase::Train);
			}
			if (batchIdx && batchIdx % savedIter == 0)
				network.saveStageModel(modelSavedPath, 1);

			if (batchIdx && batchIdx % lrSetp == 0)
			{
				//network.switchPhase(dlex_cnn::tind::Phase::Test);
				//std::tie(val_accuracy, val_loss) = test_in_train(network, 128, validate_data_);
				//network.switchPhase(dlex_cnn::tind::Phase::Train);

				//update learning rate
				learningRate = std::max(learningRate*decayRate, minLearningRate);
				network.setLearningRate(learningRate);
				//printf("epoch[%d] val_loss : %f , val_accuracy : %.4f%%\n", epochIdx++, val_loss, val_accuracy*100.0f);
			}
		}

		std::tie(val_accuracy, val_loss) = test_in_train(network, 128, validate_data_);
		printf("final val_loss : %f , final val_accuracy : %.4f%%\n", val_loss, val_accuracy*100.0f);
		//success = network.saveModel(modelFilePath);
		//assert(success);
		printf("finished training.\n");
		releaseMnistData();
	}

	void MnistTrainTest::test(const std::string& modelFilePath, const int iter)
	{
		loadMnistData(tind::Test);

		const int batch = 64;
		const int channels = validate_data_[0].first.channels;
		const int width = validate_data_[0].first.width;
		const int height = validate_data_[0].first.height;
		printf("channels:%d , width:%d , height:%d\n", channels, width, height);

		registerOpClass();

		printf("construct network begin...\n");
		dlex_cnn::NetWork<float> network;
		network.netWorkInit("netA");
		network.loadStageModel(modelFilePath, iter);
		printf("construct network done.\n");

		//test
		printf("begin test...\n");
		float accuracy = 0.0f, loss = std::numeric_limits<float>::max();
		std::tie(accuracy, loss) = test_in_train(network, batch, validate_data_);
		printf("accuracy : %.4f%%\n", accuracy*100.0f);
		printf("finished test.\n");

		//// free
		releaseMnistData();
	}
}

void mnistTrain()
{
	dlex_cnn::MnistTrainTest mnist;
	mnist.train();
	system("pause");

	return ;
}	
void mnistTest()
{
	dlex_cnn::MnistTrainTest mnist;
	const std::string model_file = "./";
	mnist.test(model_file, 1);
	system("pause");

	return ;
}