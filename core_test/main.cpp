#include <iostream>
#include "util/math_functions.h"
#include "train_test/mnist_train.h"
#include "operator_test/convolution_op_test.h"
#include "operator_test/deconvolution_op_test.h"
#include "operator_test/pooling_op_test.h"
#include "util_test/util_test.h"

class A
{
public:
	A() {};
	virtual ~A() {};

public:
	void(*batch_loader)(int) = NULL;
	void load_batch(int a) 
	{ 
		if (batch_loader != NULL)
			batch_loader(a);
		else
			printf("didn't set\n");
	}
};

class B
{
public:
	B() {};
	virtual ~B() {};

	static void blTest(int kk)
	{
		printf("kk = %d\n", kk);
	}
};

#define DLOG_TEST(format, ...) fprintf(stderr,"ERROR: "#format"\n", ##__VA_ARGS__);

template<typename T>
void Print(T value)
{
	std::cout << value << std::endl;
}

//template<typename T>
//void Print(T head, Rail... rail)
//{
//	std::cout << head << ", ";
//	Print(rail...);
//}

#define DLOG_TEST2(format, ...) fprintf(stderr,"ERROR: "#format"\n", ##__VA_ARGS__);

int main(int argc, char* argv[])
{
	//A ac;
	//ac.batch_loader = B::blTest;
	//ac.load_batch(550);

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
	
	//TestThreadPool();

	//TestPool();
	//TestConv();
	//TestDeconv();

	//bool ret = DCHECK_GE(15, 15);
	//printf("ret = %d\n", ret);
	//
	//std::cout << "12";

	//int i = 99;
	//DLOG_ERR("123%d, %d", i, i+1);
	//DLOG_ERR("123%d", i);
	//DLOG_ERR("smal");
	//int a = 99;
	//int b = 66;
	////printf("abc:%d, %d", a, b);
	////std::cout << "abc: %d, %d" << (a, b) << "\n";, a, b
	//Print("abc:%d, %d");
	MnistTrain();
	//MnistTest();	

	system("pause");
	return 0;
}