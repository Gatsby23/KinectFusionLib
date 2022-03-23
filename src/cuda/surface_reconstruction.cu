// Performs surface reconstruction, i.e. updates the internal volume with data from the current frame
// This is CUDA code; compile with nvcc
// Author: Christian Diller, git@christian-diller.de

#include "include/common.h"

using Vec2ida = Eigen::Matrix<int, 2, 1, Eigen::DontAlign>;

namespace kinectfusion {
    namespace internal {
        namespace cuda {

           /*********************************************************************************
            * @brief 这里就是SDF整合进去，但是有点没明白就是Eigen::DontAlign，这个是用来做啥的
            **********************************************************************************/
            __global__
            void update_tsdf_kernel(const PtrStepSz<float> depth_image, const PtrStepSz<uchar3> color_image,
                                    PtrStepSz<short2> tsdf_volume, PtrStepSz<uchar3> color_volume,
                                    int3 volume_size, float voxel_scale,
                                    CameraParameters cam_params, const float truncation_distance,
                                    Eigen::Matrix<float, 3, 3, Eigen::DontAlign> rotation, Vec3fda translation)
            {
                // 这里表示的事Volume数据里的索引格（这里是x,y平面的索引）
                const int x = blockIdx.x * blockDim.x + threadIdx.x;
                const int y = blockIdx.y * blockDim.y + threadIdx.y;

                // 如果这个点大于volume的边界就直接返回
                if (x >= volume_size.x || y >= volume_size.y)
                    return;

                // 这里是再按z轴来遍历
                for (int z = 0; z < volume_size.z; ++z) {
                    // 需要注意的x, y, z表示的是体素左下角的点，现在等于说将其挪到体素中心位置
                    const Vec3fda position((static_cast<float>(x) + 0.5f) * voxel_scale,
                                           (static_cast<float>(y) + 0.5f) * voxel_scale,
                                           (static_cast<float>(z) + 0.5f) * voxel_scale);
                  
                    // 计算出在相机下的位置（这里实际上是point on camera corrdinates.）
                    const Vec3fda camera_pos = rotation * position + translation;

                    // 如果当前体素的位置z轴是负的，就代表有误，因为不可能从相机看到反面的体素
                    if (camera_pos.z() <= 0)
                        continue;

                    // 这里是构建图像的UV平面
                    const Vec2ida uv(
                            __float2int_rn(camera_pos.x() / camera_pos.z() * cam_params.focal_x + cam_params.principal_x),
                            __float2int_rn(camera_pos.y() / camera_pos.z() * cam_params.focal_y + cam_params.principal_y));

                    // 如果UV平面大于对应的深度图像，也就直接返回，不用继续处理了
                    if (uv.x() < 0 || uv.x() >= depth_image.cols || uv.y() < 0 || uv.y() >= depth_image.rows)
                        continue;

                    // 这里等于说取出depth的值
                    const float depth = depth_image.ptr(uv.y())[uv.x()];

                    // 如果depth的值小于0，则这个点，相机测量有问题，直接返回就好
                    if (depth <= 0)
                        continue;

                    /// 这里计算λ内部数值，对应公式7
                    const Vec3fda xylambda(
                            (uv.x() - cam_params.principal_x) / cam_params.focal_x,
                            (uv.y() - cam_params.principal_y) / cam_params.focal_y,
                            1.f);
                    // 这里是求模
                    const float lambda = xylambda.norm();

                    // 这里对应的应该是F_{Rk}，公式(6)
                    // 只是没仙童，为什么这里t_{g,k} - p就是camera_pos
                    const float sdf = (-1.f) * ((1.f / lambda) * camera_pos.norm() - depth);

                    // 如果大于-的值得话，就代表这个TSDF是可以的，生成然后放进去
                    if (sdf >= -truncation_distance) {
                        const float new_tsdf = fmin(1.f, sdf / truncation_distance);

                        // 获得现在对应的tsdf对
                        short2 voxel_tuple = tsdf_volume.ptr(z * volume_size.y + y)[x];

                        const float current_tsdf = static_cast<float>(voxel_tuple.x) * DIVSHORTMAX;
                        const int current_weight = voxel_tuple.y;

                        const int add_weight = 1;

                        // 这里是对TSDF进行更新
                        const float updated_tsdf = (current_weight * current_tsdf + add_weight * new_tsdf) /
                                                   (current_weight + add_weight);

                        // 生成相关的TSDF的值
                        const int new_weight = min(current_weight + add_weight, MAX_WEIGHT);
                        const int new_value = max(-SHORTMAX, min(SHORTMAX, static_cast<int>(updated_tsdf * SHORTMAX)));

                        tsdf_volume.ptr(z * volume_size.y + y)[x] = make_short2(static_cast<short>(new_value),
                                                                                static_cast<short>(new_weight));

                        // 这里相当于把颜色更新进去
                        if (sdf <= truncation_distance / 2 && sdf >= -truncation_distance / 2) {
                            uchar3& model_color = color_volume.ptr(z * volume_size.y + y)[x];
                            const uchar3 image_color = color_image.ptr(uv.y())[uv.x()];

                            model_color.x = static_cast<uchar>(
                                    (current_weight * model_color.x + add_weight * image_color.x) /
                                    (current_weight + add_weight));
                            model_color.y = static_cast<uchar>(
                                    (current_weight * model_color.y + add_weight * image_color.y) /
                                    (current_weight + add_weight));
                            model_color.z = static_cast<uchar>(
                                    (current_weight * model_color.z + add_weight * image_color.z) /
                                    (current_weight + add_weight));
                        }
                    }
                }
            }


 
            /************************************************************************
             * @brief 这里是针对表面重建，integrate_to_volume来做
               @param model_view是啥: ModelView实际上就是外参，表示位姿
             *************************************************************************/ 
            void surface_reconstruction(const cv::cuda::GpuMat& depth_image, const cv::cuda::GpuMat& color_image,
                                        VolumeData& volume,
                                        const CameraParameters& cam_params, const float truncation_distance,
                                        const Eigen::Matrix4f& model_view)
            {
                const dim3 threads(32, 32);
                const dim3 blocks((volume.volume_size.x + threads.x - 1) / threads.x,
                                  (volume.volume_size.y + threads.y - 1) / threads.y);

                /// 这里来对场景进行重建->只是这里不理解的是volume.voxel_scale是什么意思？
                update_tsdf_kernel<<<blocks, threads>>>(depth_image, color_image,
                        volume.tsdf_volume, volume.color_volume,
                        volume.volume_size, volume.voxel_scale,
                        cam_params, truncation_distance,
                        model_view.block(0, 0, 3, 3), model_view.block(0, 3, 3, 1));

                cudaThreadSynchronize();
            }
        }
    }
}
