////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  Test for thread pool.
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#include "util_test.h"

void CompTest(int *a, int *b, int len) {
  printf("a[0]: %d\n", a[0]);
  for (int i = 0; i < len; i++) {
    b[i] = a[i] + b[i];
    //printf("g(%d), ", b[i]);
  }
}

int TestThreadPool() {
  int len = 10;// 1000000;
  int num = 50;
  dlex_cnn::Timer timer;
  int **a = (int **)malloc(sizeof(int *) * num);
  int **b = (int **)malloc(sizeof(int *) * num);
  for (int i = 0; i < num; i++) {
    a[i] = (int *)malloc(sizeof(int) * len);
    b[i] = (int *)malloc(sizeof(int) * len);
  }

  for (int i = 0; i < num; i++) {
    for (int j = 0; j < len; j++) {
      a[i][j] = b[i][j] = i;// i*len + j;
    }
  }
  printf("sizeof(int *) = %zd, sizeof(int) = %zd\n", sizeof(int *), sizeof(int));

  int thread_num = 13;
  dlex_cnn::ThreadPool pool;
  pool.CreateThreads(thread_num);
  auto func = [&](const int start, const int end)	{
    printf("s(%d),e(%d)\n", start, end);
    for (int idx = start; idx < end; idx++)
      CompTest(*(a + idx), *(b + idx), len);
  };

  timer.Start();
  pool.Exec(func, num);

  printf("\ntime: %f, %d\n", timer.MicroSeconds(), b[num - 1][len - 1]);
  for (int i = 0; i < num; i++) {
    for (int j = 0; j < len; j++) {
      printf("%d, ", b[i][j]);
    }
  }

  system("pause");
  return 0;
}
