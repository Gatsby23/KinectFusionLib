// This is the CPU part of the surface measurement
// Author: Christian Diller, git@christian-diller.de

#include <kinectfusion.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wextra"
#pragma GCC diagnostic ignored "-Weffc++"
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#pragma GCC diagnostic pop

using cv::cuda::GpuMat;

namespace kinectfusion {
    namespace internal {

        namespace cuda { // Forward declare CUDA functions
            // 这里是cuda的前向声明（只是为了声明函数）
            void compute_vertex_map(const GpuMat& depth_map, GpuMat& vertex_map, const float depth_cutoff,
                                    const CameraParameters cam_params);
            void compute_normal_map(const GpuMat& vertex_map, GpuMat& normal_map);
        }

        /**************************************************************
         * @brief 观测数据
         * @param input_frame 输入数据
         * @param camera_params 虚拟相机参数配置
         * @param num_levels 层数
         * @param color_sigma 颜色噪声?
         * @param spatial_sigma 空间噪声?
         *************************************************************/
        FrameData surface_measurement(const cv::Mat_<float>& input_frame,
                                     const CameraParameters& camera_params,
                                      const size_t num_levels, const float depth_cutoff,
                                      const int kernel_size, const float color_sigma, 
                                      const float spatial_sigma)
        {
            // Initialize frame data
            // 初始化图像帧数据->这里只是初始化好内存，并没有进行相关数据生成.
            FrameData data(num_levels);

            // Allocate GPU memory
            /// 分配GPU内存
            for (size_t level = 0; level < num_levels; ++level) {
                // 图像金子塔每一层都包含相关的高度和宽度
                const int width = camera_params.level(level).image_width;
                const int height = camera_params.level(level).image_height;

                //在显存中创建连续矩阵
                data.depth_pyramid[level] = cv::cuda::createContinuous(height, width, CV_32FC1);
                data.smoothed_depth_pyramid[level] = cv::cuda::createContinuous(height, width, CV_32FC1);

                data.color_pyramid[level] = cv::cuda::createContinuous(height, width, CV_8UC3);

                data.vertex_pyramid[level] = cv::cuda::createContinuous(height, width, CV_32FC3);
                data.normal_pyramid[level] = cv::cuda::createContinuous(height, width, CV_32FC3);
            }

            // Start by uploading original frame to GPU
            // 开始将原始数据传给GPU（这里需要注意，只讲深度数据传递给GPU）
            data.depth_pyramid[0].upload(input_frame);

            // Build pyramids and filter bilaterally on GPU 
            // 在GPU上创建金字塔和双向滤波器
            // 首先创建一个cuda流，有cuda流的话，将任务都放在cuda上面，依据放的顺序，依次进行操作
            cv::cuda::Stream stream;
            for (size_t level = 1; level < num_levels; ++level)
                cv::cuda::pyrDown(data.depth_pyramid[level - 1], data.depth_pyramid[level], stream);
            for (size_t level = 0; level < num_levels; ++level) {
                cv::cuda::bilateralFilter(data.depth_pyramid[level], // source
                                          data.smoothed_depth_pyramid[level], // destination
                                          kernel_size,
                                          color_sigma,
                                          spatial_sigma,
                                          cv::BORDER_DEFAULT,
                                          stream);
            }
            // 等到任务完成
            stream.waitForCompletion();

            // Compute vertex and normal maps
            // 计算顶点图和法向量图->注意这里计算完后就放到data里面，这样一次表面测量就完成了
            for (size_t level = 0; level < num_levels; ++level) {
                cuda::compute_vertex_map(data.smoothed_depth_pyramid[level], data.vertex_pyramid[level],
                                         depth_cutoff, camera_params.level(level));
                cuda::compute_normal_map(data.vertex_pyramid[level], data.normal_pyramid[level]);
            }

            return data;
        }
    }
}
