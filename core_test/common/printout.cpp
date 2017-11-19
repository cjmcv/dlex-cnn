#include "common/tools.h"

void matrixShow_float(std::string name, float *data, int num, int channel, int height, int width)
{
	printf("Matrix :%s\n", name.c_str());
	printf("(%d, %d, %d, %d \n", num, channel, height, width);
	int c_size = height * width;
	int n_size = channel * c_size;
	for (int n = 0; n < num; n++)
	{
		for (int c = 0; c < channel; c++)
		{
			printf(" n - ch : %d (%d)\n", n, c);
			for (int i = 0; i < height; i++)
			{
				for (int j = 0; j < width; j++)
				{
					printf("%f, ", *(data + n*n_size + c*c_size + i*width + j));
				}
				printf("\n");
			}
		}
	}
	printf(")\n");
}

void matrixShow_int(std::string name, int *data, int num, int channel, int height, int width)
{
	printf("Matrix :%s\n", name.c_str());
	printf("(%d, %d, %d, %d \n", num, channel, height, width);
	int c_size = height * width;
	int n_size = channel * c_size;
	for (int n = 0; n < num; n++)
	{
		for (int c = 0; c < channel; c++)
		{
			printf(" n - ch : %d (%d)\n", n, c);
			for (int i = 0; i < height; i++)
			{
				for (int j = 0; j < width; j++)
				{
					printf("%d, ", *(data + n*n_size + c*c_size + i*width + j));
				}
				printf("\n");
			}
		}
	}
	printf(")\n");
}
