#include <iostream>
#include "util/math_functions.h"
#include "train_test/mnist_train.h"
#include "operator_test/convolution_op_test.h"
#include "operator_test/deconvolution_op_test.h"
#include "operator_test/pooling_op_test.h"

int main(int argc, char* argv[])
{
	//float A[] = { 1, 2, 3, 4, 
	//	1, 2, 3, 4,
	//	1, 2, 3, 4 };
	//float B[] = { 1, 2, 3, 4, 5, 
	//	1, 2, 3, 4, 5,
	//	1, 2, 3, 4, 5,
	//	1, 2, 3, 4, 5 };

	//float C[100] = {0};
	//float blas[100] = {0};
	//int M = 3;
	//int N = 5;
	//int K = 4;
	//dlex_cnn::gemm(M, N, K, 1, A, B, 1, C, blas);

	//for (int i = 0; i < 15; i++)
	//	printf("%f, ", C[i]);
	//system("pause");
	//return test_threadPool(argc, argv);

	//testPool();
	//testConv();
	//testDeconv();

	//bool ret = DCHECK_GE(15, 15);
	//printf("ret = %d\n", ret);
	//
	//std::cout << "12";

	//int i = 99;
	//DLOG_ERR("123%d, %d", i, i+1);
	//DLOG_ERR("123%d", i);
	//DLOG_ERR("smal");
	//system("pause");


	mnistTrain();
	//mnistTest();
	return 0;
}