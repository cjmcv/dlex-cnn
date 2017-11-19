////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#ifndef DLEX_TIMER_HPP_
#define DLEX_TIMER_HPP_

#include <chrono>

// finish
// DLOG_ERR("Convolution load_param failed\n");添加误差判断，不加日志，在linux训练可加重定向。重定向时用stderr仍会显示在终端，而不会去文件中
// 成员命名规范（函数大小写，变量_小写）, train/test需要可共存
// 添加graph 的node size打印,graph_.reset(new Graph<Dtype>())不要放在构造中-->已放至init
// 添加opParam的映射和字串化，需要由字符串得到opParam -> 
// 在node留一个字符串用于表示对应opParam，
// 模型读写：1、二进制模型（不可视，结构+blob，不含optimizer等，用于输出使用）；-- 延后
//			 2、输出可视结构 + 可视超参数 + 二进制blob等， 用于存储阶段训练模型。  --  完成
// 字符串if else，分支声明对应对象返回
// 写时：先执行param的字串化（用op重写实现返回string），用node写出；读时，将param字串赋值到node中，由node通过inte_op重写的param反字串化
// op参数有默认构造，使用参数时要添加判断是否已设置好参数

// 加模型读写->添加node的字符串化，
// tensor中（num,channels, height, width）中height/width的顺序需要跟opencv图像输入对应确认，待确认


// 待处理问题：
// Convolution group卷积未实现
// 将conv和deconv的参数中kernel num/channels 改成output num进行统一?对外改为output num，对内还是num/channel
// Pooling op中极大值池化max_idx_map_, 在不训练时可以省掉；
// 构建的网络中间带不用的节点会崩溃？
// optimizer修改优化
// op里的inline和返回const


// network中添加输入数据预处理，如归一化和均值

// (优化)规范化函数（， node中较多函数直接调用op的）
// 添加统一错误码


// 只测试不训练时，不需要为diff_和gradient_开辟内存
// softmax_cross_entropy_hop.cpp 的allocOpBuf4Train，子op的创建需要改，改到一起映射？


// 模型加载(延后做)，（多线程，后加多路处理多个输入节点，给节点加依赖判断/计算图模型引擎）
// 线程与内存管理
// 添加注释
// 以宏定义添加第三方日志库？？


// node中data和w/b是否需要隔离--暂不隔离

// 考虑一些node仅对一些数据做简单处理，而不影响网络连接，而caffe中的relu等，在网络中直接删除仍可以继续使用，不会出现节点连接不上的问题
// 可以考虑加一类特殊的node，不影响输入输出的连接，以旁支的方式加入，有添加则对输入数据处理（原地返回数据），不添加不处理。
// 添加network init，将graph的new移到network的init中

// network函数 包含graph，超参数，训练策略控制，optimizer等
// snapshot，lr，optimizer，test_iter, max_iter，CPU/GPU
// lr控制函数放在net里面，设置到optimizer中？

namespace dlex_cnn
{
class Timer {
public:
	typedef std::chrono::high_resolution_clock clock;
	typedef std::chrono::nanoseconds ns;
	Timer() { Start(); }
	/**
	* @brief Starts a timer.
	*/
	inline void Start() { start_time_ = clock::now(); }
	inline float NanoSeconds() {
		return std::chrono::duration_cast<ns>(clock::now() - start_time_).count();
	}
	/**
	* @brief Returns the elapsed time in milliseconds.
	*/
	inline float MilliSeconds() { return NanoSeconds() / 1000000.f; }
	/**
	* @brief Returns the elapsed time in microseconds.
	*/
	inline float MicroSeconds() { return NanoSeconds() / 1000.f; }
	/**
	* @brief Returns the elapsed time in seconds.
	*/
	inline float Seconds() { return NanoSeconds() / 1000000000.f; }

protected:
	std::chrono::time_point<clock> start_time_;
};
}
#endif //DLEX_TIMER_HPP_