#pragma once

#pragma nv_diag_suppress 20012
#pragma nv_diag_suppress 20014

#include <base/Constraint.h>
#include <states/SingleArmStates.h>
#include <util.h>
#include <cuda_runtime.h>

namespace CUDAMPLib
{
    class TaskSpaceConstraint : public BaseConstraint
    {
        public:
            /**
                * @brief Construct a new Task Space Constraint object
                * @param constraint_name The name of the task space constraint.
                * @param task_link_index The index of the task link in the link names.
                * @param offset_pose_in_task_link The offset pose in the task link frame, and it is represented as pose matrix.
                * @param reference_frame The reference pose in the base_link frame, and it is represented as a x, y, z, roll, pitch, yaw.
                * @param tolerance The tolerance of the task space constraint in the x, y, z, roll, pitch, yaw.
             */
            TaskSpaceConstraint(
                const std::string& constraint_name,
                const int task_link_index,
                const Eigen::Matrix4d& offset_pose_in_task_link,
                const std::vector<float>& reference_frame,
                const std::vector<float>& tolerance
            );

            ~TaskSpaceConstraint() override;

            void computeCost(BaseStatesPtr states) override;

            void computeCostFast(BaseStatesPtr states) override;

            void computeCostLarge(BaseStatesPtr states) override;

            void computeGradient(BaseStatesPtr states) override;
            
        private:
            int task_link_index_;
            Eigen::Matrix4d offset_pose_in_task_link_;
            std::vector<float> reference_frame_;
            std::vector<float> tolerance_;

            float * d_offset_pose_in_task_link_;
            float * d_reference_frame_;
            float * d_tolerance_;
    };

    typedef std::shared_ptr<TaskSpaceConstraint> TaskSpaceConstraintPtr;
} // namespace CUDAMPLibs