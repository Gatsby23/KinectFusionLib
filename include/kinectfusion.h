// This is the KinectFusion Pipeline
// Author: Christian Diller, git@christian-diller.de

#ifndef KINECTFUSION_H
#define KINECTFUSION_H

#include "data_types.h"

namespace kinectfusion {
    /********************************************************************************************************
     * @brief 这里是KinectFusion运行的整体Pipeline. 对输入的视觉帧处理并将它正和岛同一个Volume中去.
     * 这里便是论文当中所描述的主要四个步骤：
     *  （1）表面测量：计算顶点图和深度图和它们对应的多层金字塔.
     *  （2）位姿观测：用测量得到深度图和反向投影得到的深度图通过ICP的方式来进行位姿配准.
     *   (3) 表面重建：将表面测量融合到整体Volume中.
     *   (4) 表面预测：通过反向投影（RayCasting）的方式得到预测得到的表面.
     * 重建完成后，可以输出整个重建效果.
     ********************************************************************************************************/
    class Pipeline {
    public:
        /*******************************************************
         *  @brief 构建整体流程，设置了整体的模型volume和相机参数
         *  @param _camera_parameters: 整个流程用到的相机参数
         *  @param _configuration:     整个流程中用到的所有参数
         *******************************************************/
        Pipeline(const CameraParameters _camera_parameters,
                 const GlobalConfiguration _configuration);

        ~Pipeline() = default;

        /**
         * Invoke this for every frame you want to fuse into the global volume
         * @param depth_map The depth map for the current frame. Must consist of float values representing the depth in mm
         * @param color_map The RGB color map. Must be a matrix (datatype CV_8UC3)
         * @return Whether the frame has been fused successfully. Will only be false if the ICP failed.
         */

        /*****************************************************************************************
         * @brief 对所有想放到整体volume中的图像帧进行处理.
         * @param depth_map 当前帧对应的深度图,这里必须将所有的depth值单位转换成mm,float类型.
         * @param color_map 颜色图，必须是个矩阵.
         * @return 只有当ICP解算位姿的错的时候，才会返回false，否则无论是否整合到global volume中，都是返回true.
         *****************************************************************************************/
        bool process_frame(const cv::Mat_<float>& depth_map, const cv::Mat_<cv::Vec3b>& color_map);

        /**
         * Retrieve all camera poses computed so far
         * @return A vector for 4x4 camera poses, consisting of rotation and translation
         */
        std::vector<Eigen::Matrix4f> get_poses() const;

        /**
         * Use this to get a visualization of the last raycasting
         * @return The last (colorized) model frame from raycasting the internal volume
         */
        /*****************************************
         * @brief 这里是对raycast后的结果进行可视化
         * @param 这里是最后颜色模型的反向投影
         ******************************************/
        cv::Mat get_last_model_frame() const;

        /**
         * Extract a point cloud
         * @return A PointCloud representation (see description of PointCloud for more information on the data layout)
         */
        PointCloud extract_pointcloud() const;

        /**
         * Extract a dense surface mesh
         * @return A SurfaceMesh representation (see description of SurfaceMesh for more information on the data layout)
         */
        /*********************************************
         * @brief 抽取出来稠密表面
         * @return 返回mesh的表面，更多的信息在data layout里面
         *******************************************/
        SurfaceMesh extract_mesh() const;

    private:
        // Internal parameters, not to be changed after instantiation
        const CameraParameters camera_parameters;
        const GlobalConfiguration configuration;

        // The global volume (containing tsdf and color)
        /// 这里是global volume的值，用来保存tsdf和颜色
        internal::VolumeData volume;

        // The model data for the current frame
        /// 对于当前帧的整体数据
        internal::ModelData model_data;

        // Poses: Current and all previous
        Eigen::Matrix4f current_pose;
        std::vector<Eigen::Matrix4f> poses;

        // Frame ID and raycast result for output purposes
        size_t frame_id;
        cv::Mat last_model_frame;
    };

    /**
     * Store a PointCloud instance as a PLY file.
     * If file cannot be saved, nothing will be done
     * @param filename The path and name of the file to write to; if it does not exists, it will be created and
     *                 if it exists it will be overwritten
     * @param point_cloud The PointCloud instance
     */
    void export_ply(const std::string& filename, const PointCloud& point_cloud);

    /**
     * Store a SurfaceMesh instance as a PLY file.
     * If file cannot be saved, nothing will be done
     * @param filename The path and name of the file to write to; if it does not exists, it will be created and
     *                 if it exists it will be overwritten
     * @param surface_mesh The SurfaceMesh instance
     */
    void export_ply(const std::string& filename, const SurfaceMesh& surface_mesh);


    namespace internal {

        /*
         * Step 1: SURFACE MEASUREMENT
         * Compute vertex and normal maps and their pyramids
         */
        FrameData surface_measurement(const cv::Mat_<float>& input_frame,
                                      const CameraParameters& camera_params,
                                      const size_t num_levels, const float depth_cutoff,
                                      const int kernel_size, const float color_sigma, const float spatial_sigma);


        /*
         * Step 2: POSE ESTIMATION
         * Use ICP with measured depth and predicted surface to localize camera
         */
        bool pose_estimation(Eigen::Matrix4f& pose,
                             const FrameData& frame_data,
                             const ModelData& model_data,
                             const CameraParameters& cam_params,
                             const int pyramid_height,
                             const float distance_threshold, const float angle_threshold,
                             const std::vector<int>& iterations);

        namespace cuda {

            /*
             * Step 3: SURFACE RECONSTRUCTION
             * Integration of surface measurements into a global volume
             */
            void surface_reconstruction(const cv::cuda::GpuMat& depth_image,
                                        const cv::cuda::GpuMat& color_image,
                                        VolumeData& volume,
                                        const CameraParameters& cam_params,
                                        const float truncation_distance,
                                        const Eigen::Matrix4f& model_view);


            /*
             * Step 4: SURFACE PREDICTION
             * Raycast volume in order to compute a surface prediction
             */
            void surface_prediction(const VolumeData& volume,
                                    cv::cuda::GpuMat& model_vertex,
                                    cv::cuda::GpuMat& model_normal,
                                    cv::cuda::GpuMat& model_color,
                                    const CameraParameters& cam_parameters,
                                    const float truncation_distance,
                                    const Eigen::Matrix4f& pose);

            PointCloud extract_points(const VolumeData& volume, const int buffer_size);

            SurfaceMesh marching_cubes(const VolumeData& volume, const int triangles_buffer_size);
        }

    }
}
#endif //KINECTFUSION_H
