//////////////////////////////////// math.hh /////////////////////////////////////
#pragma once

#include <algorithm>
#include <cmath>

namespace ppln::collision
{
    using Point = std::array<float, 3>;

    template <typename DataT>
    inline constexpr auto dot_2(const DataT &ax, const DataT &ay, const DataT &bx, const DataT &by) -> DataT
    {
        return (ax * bx) + (ay * by);
    }

    template <typename DataT>
    inline constexpr auto dot_3(
        const DataT &ax,
        const DataT &ay,
        const DataT &az,
        const DataT &bx,
        const DataT &by,
        const DataT &bz) -> DataT
    {
        return (ax * bx) + (ay * by) + (az * bz);
    }

    template <typename DataT>
    inline constexpr auto sql2_3(
        const DataT &ax,
        const DataT &ay,
        const DataT &az,
        const DataT &bx,
        const DataT &by,
        const DataT &bz) -> DataT
    {
        const auto xs = (ax - bx);
        const auto ys = (ay - by);
        const auto zs = (az - bz);

        return dot_3(xs, ys, zs, xs, ys, zs);
    }


    inline constexpr auto clamp(const float &v, const float &lower, const float &upper) -> float
    {
        return std::max(std::min(v, upper), lower);
    }

}  // namespace ppln::collision

///////////////////////////////////////// environment.hh /////////////////////////////////////////
#pragma once

#include <vector>
#include <optional>
#include "shapes.hh"

/* Adapted from https://github.com/KavrakiLab/vamp/blob/main/src/impl/vamp/collision/environment.hh */

namespace ppln::collision
{
    template <typename DataT>
    struct Environment
    {
        Sphere<DataT> *spheres;
        unsigned int num_spheres;

        Capsule<DataT> *capsules;
        unsigned int num_capsules;

        Capsule<DataT> *z_aligned_capsules;
        unsigned int num_z_aligned_capsules;

        Cylinder<DataT> *cylinders;
        unsigned int num_cylinders;

        Cuboid<DataT> *cuboids;
        unsigned int num_cuboids;

        Cuboid<DataT> *z_aligned_cuboids;
        unsigned int num_z_aligned_cuboids;

        // HeightField<DataT> *heightfields;
        // unsigned int num_heightfields;

        Environment() = default;

        ~Environment() {
            delete[] spheres;
            delete[] capsules;
            delete[] cuboids;
            delete[] z_aligned_capsules;
            delete[] cylinders;
            delete[] z_aligned_cuboids;
        }
    };
}  // namespace ppln::collision

///////////////////////////////////////////////////// factory.hh /////////////////////////////////////////////////////////
#pragma once

#include <array>
#include <vector>

#include "shapes.hh"

#include <Eigen/Dense>

/* Adapted from https://github.com/KavrakiLab/vamp/blob/main/src/impl/vamp/collision/factory.hh */

using namespace ppln;

namespace ppln::collision::factory
{
    using ConstEigenRef = const Eigen::Ref<const Eigen::Vector3f> &;
    using ConstEigenRotationRef = const Eigen::Quaternionf &;
    using ConstArrayRef = const std::array<float, 3> &;

    namespace cuboid
    {
        /**
         * Constuct a cuboid given its:
           - Center coordinate
           - Orientation about that center given in Euler XYZ angles
           - Half-extents (radii) from that center along each of the XYZ axes.
         */
        inline static auto flat(
            float center_x,
            float center_y,
            float center_z,
            float rho,
            float theta,
            float phi,
            float half_extent_x,
            float half_extent_y,
            float half_extent_z) noexcept -> collision::Cuboid<float>
        {
            const auto rotation_matrix = Eigen::AngleAxisf(phi, Eigen::Vector3f::UnitZ()) *
                                         Eigen::AngleAxisf(theta, Eigen::Vector3f::UnitY()) *
                                         Eigen::AngleAxisf(rho, Eigen::Vector3f::UnitX());

            const auto axis_1 = rotation_matrix * Eigen::Vector3f::UnitX();
            const auto axis_2 = rotation_matrix * Eigen::Vector3f::UnitY();
            const auto axis_3 = rotation_matrix * Eigen::Vector3f::UnitZ();

            return collision::Cuboid<float>(
                center_x,
                center_y,
                center_z,
                axis_1[0],
                axis_1[1],
                axis_1[2],
                axis_2[0],
                axis_2[1],
                axis_2[2],
                axis_3[0],
                axis_3[1],
                axis_3[2],
                half_extent_x,
                half_extent_y,
                half_extent_z);
        }

        inline static auto eigen(
            ConstEigenRef center,
            ConstEigenRef euler_xyz,
            ConstEigenRef half_extents) noexcept -> collision::Cuboid<float>
        {
            return flat(
                center[0],
                center[1],
                center[2],
                euler_xyz[0],
                euler_xyz[1],
                euler_xyz[2],
                half_extents[0],
                half_extents[1],
                half_extents[2]);
        };

        inline static auto eigen_rot(
            ConstEigenRef center,
            ConstEigenRotationRef rotation,
            ConstEigenRef half_extents) noexcept -> collision::Cuboid<float>
        {
            auto euler = rotation.toRotationMatrix().eulerAngles(0, 1, 2);
            return eigen(center, euler, half_extents);
        };

        inline static auto array(
            ConstArrayRef center,
            ConstArrayRef euler_xyz,
            ConstArrayRef half_extents) noexcept -> collision::Cuboid<float>
        {
            return flat(
                center[0],
                center[1],
                center[2],
                euler_xyz[0],
                euler_xyz[1],
                euler_xyz[2],
                half_extents[0],
                half_extents[1],
                half_extents[2]);
        };
    }  // namespace cuboid

    namespace cylinder
    {

        namespace endpoints
        {
            /**
               Construct a cylinder given two end points (the center of the "caps" of the cylinder)
               and the radius around the line segment connecting these two points.
             */
            inline static auto
            flat(float x1, float y1, float z1, float x2, float y2, float z2, float radius) noexcept
                -> collision::Cylinder<float>
            {
                auto x_v = x2 - x1;
                auto y_v = y2 - y1;
                auto z_v = z2 - z1;
                auto dot = dot_3(x_v, y_v, z_v, x_v, y_v, z_v);

                return collision::Cylinder<float>(
                    x1, y1, z1, x_v, y_v, z_v, radius, static_cast<float>(1.0 / dot));
            }

            inline static auto eigen(ConstEigenRef endpoint1, ConstEigenRef endpoint2, float radius) noexcept
                -> collision::Cylinder<float>
            {
                return flat(
                    endpoint1[0],
                    endpoint1[1],
                    endpoint1[2],
                    endpoint2[0],
                    endpoint2[1],
                    endpoint2[2],
                    radius);
            }

            inline static auto array(ConstArrayRef endpoint1, ConstArrayRef endpoint2, float radius) noexcept
                -> collision::Cylinder<float>
            {
                return flat(
                    endpoint1[0],
                    endpoint1[1],
                    endpoint1[2],
                    endpoint2[0],
                    endpoint2[1],
                    endpoint2[2],
                    radius);
            }
        }  // namespace endpoints

        namespace center
        {
            /**
               Construct a cylinder from a center point and an Euler XYZ orientation about that center.
               End points are each length / 2 away from the center along the local Z axis.
               Radius is the radius around the line segment connecting the points.
             */
            inline static auto flat(
                float center_x,
                float center_y,
                float center_z,
                float rho,
                float theta,
                float phi,
                float radius,
                float length) noexcept -> collision::Cylinder<float>
            {
                const auto rotation_matrix = Eigen::AngleAxisf(phi, Eigen::Vector3f::UnitZ()) *
                                             Eigen::AngleAxisf(theta, Eigen::Vector3f::UnitY()) *
                                             Eigen::AngleAxisf(rho, Eigen::Vector3f::UnitX());

                const auto tf_matrix = Eigen::Translation3f(center_x, center_y, center_z) * rotation_matrix;

                const auto p1 = tf_matrix * Eigen::Vector3f(0, 0, length / 2);
                const auto p2 = tf_matrix * Eigen::Vector3f(0, 0, -length / 2);

                return endpoints::flat(p1[0], p1[1], p1[2], p2[0], p2[1], p2[2], radius);
            }

            inline static auto
            eigen(ConstEigenRef center, ConstEigenRef euler_xyz, float radius, float length) noexcept
                -> collision::Cylinder<float>
            {
                return flat(
                    center[0],
                    center[1],
                    center[2],
                    euler_xyz[0],
                    euler_xyz[1],
                    euler_xyz[2],
                    radius,
                    length);
            }

            inline static auto eigen_rot(
                ConstEigenRef center,
                ConstEigenRotationRef rotation,
                float radius,
                float length) noexcept -> collision::Cylinder<float>
            {
                auto euler = rotation.toRotationMatrix().eulerAngles(0, 1, 2);
                return eigen(center, euler, radius, length);
            }

            inline static auto
            array(ConstArrayRef center, ConstArrayRef euler_xyz, float radius, float length) noexcept
                -> collision::Cylinder<float>
            {
                return flat(
                    center[0],
                    center[1],
                    center[2],
                    euler_xyz[0],
                    euler_xyz[1],
                    euler_xyz[2],
                    radius,
                    length);
            }
        }  // namespace center
    }  // namespace cylinder

    namespace capsule
    {

        namespace endpoints
        {
            /**
               Construct a capsule given two end points (the center of the "caps" of the capsule)
               and the radius around the line segment connecting these two points.
             */
            inline static auto
            flat(float x1, float y1, float z1, float x2, float y2, float z2, float radius) noexcept
                -> collision::Capsule<float>
            {
                auto x_v = x2 - x1;
                auto y_v = y2 - y1;
                auto z_v = z2 - z1;
                auto dot = dot_3(x_v, y_v, z_v, x_v, y_v, z_v);

                return collision::Capsule<float>(
                    x1, y1, z1, x_v, y_v, z_v, radius, static_cast<float>(1.0 / dot));
            }

            inline static auto eigen(ConstEigenRef endpoint1, ConstEigenRef endpoint2, float radius) noexcept
                -> collision::Capsule<float>
            {
                return flat(
                    endpoint1[0],
                    endpoint1[1],
                    endpoint1[2],
                    endpoint2[0],
                    endpoint2[1],
                    endpoint2[2],
                    radius);
            }

            inline static auto array(ConstArrayRef endpoint1, ConstArrayRef endpoint2, float radius) noexcept
                -> collision::Capsule<float>
            {
                return flat(
                    endpoint1[0],
                    endpoint1[1],
                    endpoint1[2],
                    endpoint2[0],
                    endpoint2[1],
                    endpoint2[2],
                    radius);
            }
        }  // namespace endpoints

        namespace center
        {
            /**
               Construct a capsule from a center point and an Euler XYZ orientation about that center.
               End points are each length / 2 away from the center along the local Z axis.
               Radius is the radius around the line segment connecting the points.
             */
            inline static auto flat(
                float center_x,
                float center_y,
                float center_z,
                float rho,
                float theta,
                float phi,
                float radius,
                float length) noexcept -> collision::Capsule<float>
            {
                const auto rotation_matrix = Eigen::AngleAxisf(phi, Eigen::Vector3f::UnitZ()) *
                                             Eigen::AngleAxisf(theta, Eigen::Vector3f::UnitY()) *
                                             Eigen::AngleAxisf(rho, Eigen::Vector3f::UnitX());

                const auto tf_matrix = Eigen::Translation3f(center_x, center_y, center_z) * rotation_matrix;

                const auto p1 = tf_matrix * Eigen::Vector3f(0, 0, length / 2);
                const auto p2 = tf_matrix * Eigen::Vector3f(0, 0, -length / 2);

                return endpoints::flat(p1[0], p1[1], p1[2], p2[0], p2[1], p2[2], radius);
            }

            inline static auto
            eigen(ConstEigenRef center, ConstEigenRef euler_xyz, float radius, float length) noexcept
                -> collision::Capsule<float>
            {
                return flat(
                    center[0],
                    center[1],
                    center[2],
                    euler_xyz[0],
                    euler_xyz[1],
                    euler_xyz[2],
                    radius,
                    length);
            }

            inline static auto eigen_rot(
                ConstEigenRef center,
                ConstEigenRotationRef rotation,
                float radius,
                float length) noexcept -> collision::Capsule<float>
            {
                auto euler = rotation.toRotationMatrix().eulerAngles(0, 1, 2);
                return eigen(center, euler, radius, length);
            }

            inline static auto
            array(ConstArrayRef center, ConstArrayRef euler_xyz, float radius, float length) noexcept
                -> collision::Capsule<float>
            {
                return flat(
                    center[0],
                    center[1],
                    center[2],
                    euler_xyz[0],
                    euler_xyz[1],
                    euler_xyz[2],
                    radius,
                    length);
            }
        }  // namespace center
    }  // namespace capsule

    namespace sphere
    {
        inline static auto flat(float center_x, float center_y, float center_z, float radius) noexcept
            -> collision::Sphere<float>
        {
            return collision::Sphere<float>(center_x, center_y, center_z, radius);
        }

        inline static auto eigen(ConstEigenRef center, float radius) noexcept -> collision::Sphere<float>
        {
            return flat(center[0], center[1], center[2], radius);
        }

        inline static auto array(ConstArrayRef center, float radius) noexcept -> collision::Sphere<float>
        {
            return flat(center[0], center[1], center[2], radius);
        }
    }  // namespace sphere
}  // namespace ppln::collision::factory

///////////////////////////////////////////////// shapes.hh /////////////////////////////////////////////////////////
#pragma once

#include <string>
#include <memory>

#include "math.hh"

/* Adapted from https://github.com/KavrakiLab/vamp/blob/main/src/impl/vamp/collision/shapes.hh */

namespace ppln::collision
{
    template <typename DataT>
    struct Shape
    {
        std::string name;
        DataT min_distance;

        Shape() = default;

        template <typename OtherDataT>
        explicit Shape(const Shape<OtherDataT> &other) : name(other.name), min_distance(other.min_distance)
        {
        }
    };

    // A cuboid, or rectangular prism, shape oriented in an arbitrary frame. The
    // cube is defined by its center (x, y, z), the three axes defining its frame
    // (axis_1_x, axis_1_y, axis_1_z, etc.), and the half-lengths along each axis
    // (axis_1_r, etc.)
    template <typename DataT>
    struct Cuboid : public Shape<DataT>
    {
        DataT x;
        DataT y;
        DataT z;

        DataT axis_1_x;
        DataT axis_1_y;
        DataT axis_1_z;
        DataT axis_2_x;
        DataT axis_2_y;
        DataT axis_2_z;
        DataT axis_3_x;
        DataT axis_3_y;
        DataT axis_3_z;

        DataT axis_1_r;
        DataT axis_2_r;
        DataT axis_3_r;

        inline constexpr auto compute_min_distance() -> DataT
        {
            auto d_1 = dot_3(-x, -y, -z, axis_1_x, axis_1_y, axis_1_z);
            auto d_2 = dot_3(-x, -y, -z, axis_2_x, axis_2_y, axis_2_z);
            auto d_3 = dot_3(-x, -y, -z, axis_3_x, axis_3_y, axis_3_z);

            auto v_1 = clamp(d_1, -axis_1_r, axis_1_r);
            auto v_2 = clamp(d_2, -axis_2_r, axis_2_r);
            auto v_3 = clamp(d_3, -axis_3_r, axis_3_r);

            auto x_n = x + axis_1_x * v_1 + axis_2_x * v_2 + axis_3_x * v_3;
            auto y_n = y + axis_1_y * v_1 + axis_2_y * v_2 + axis_3_y * v_3;
            auto z_n = z + axis_1_z * v_1 + axis_2_z * v_2 + axis_3_z * v_3;

            return sqrt(x_n * x_n + y_n * y_n + z_n * z_n);
        }

        Cuboid() = default;

        explicit Cuboid(
            DataT x,
            DataT y,
            DataT z,  //
            DataT axis_1_x,
            DataT axis_1_y,
            DataT axis_1_z,  //
            DataT axis_2_x,
            DataT axis_2_y,
            DataT axis_2_z,  //
            DataT axis_3_x,
            DataT axis_3_y,
            DataT axis_3_z,  //
            DataT axis_1_r,
            DataT axis_2_r,
            DataT axis_3_r)
          : Shape<DataT>()
          , x(x)
          , y(y)
          , z(z)
          , axis_1_x(axis_1_x)
          , axis_1_y(axis_1_y)
          , axis_1_z(axis_1_z)
          , axis_2_x(axis_2_x)
          , axis_2_y(axis_2_y)
          , axis_2_z(axis_2_z)
          , axis_3_x(axis_3_x)
          , axis_3_y(axis_3_y)
          , axis_3_z(axis_3_z)
          , axis_1_r(axis_1_r)
          , axis_2_r(axis_2_r)
          , axis_3_r(axis_3_r)
        {
            Shape<DataT>::min_distance = compute_min_distance();
        }

        template <typename OtherDataT>
        explicit Cuboid(const Cuboid<OtherDataT> &other)
          : Shape<DataT>(other)
          , x(other.x)
          , y(other.y)
          , z(other.z)
          , axis_1_x(other.axis_1_x)
          , axis_1_y(other.axis_1_y)
          , axis_1_z(other.axis_1_z)
          , axis_2_x(other.axis_2_x)
          , axis_2_y(other.axis_2_y)
          , axis_2_z(other.axis_2_z)
          , axis_3_x(other.axis_3_x)
          , axis_3_y(other.axis_3_y)
          , axis_3_z(other.axis_3_z)
          , axis_1_r(other.axis_1_r)
          , axis_2_r(other.axis_2_r)
          , axis_3_r(other.axis_3_r)
        {
        }
    };

    // A cylinder shape oriented in an arbitrary frame. The cylinder is defined by
    // two endpoints, which we store as the first point (x1, y1, z1) and the vector
    // from the first point to the second point (xv, yv, zv), and its radius (r)
    template <typename DataT>
    struct Cylinder : public Shape<DataT>
    {
        DataT x1;
        DataT y1;
        DataT z1;

        DataT xv;
        DataT yv;
        DataT zv;

        DataT r;

        // This is the reciprocal of the dot product of the vector between the
        // cylinder's endpoints, i.e. rdv = 1 / ([xv yv zv] . [xv yv zv]) We store it
        // for convenience to make some computations faster
        DataT rdv;

        inline constexpr auto x2() -> DataT
        {
            return x1 + xv;
        }

        inline constexpr auto y2() -> DataT
        {
            return y1 + yv;
        }

        inline constexpr auto z2() -> DataT
        {
            return z1 + zv;
        }

        inline constexpr auto compute_min_distance() -> DataT
        {
            auto dot = clamp(dot_3(-x1, -y1, -z1, xv, yv, zv) * rdv, 0.F, 1.F);

            auto xp = x1 + xv * dot;
            auto yp = y1 + yv * dot;
            auto zp = z1 + zv * dot;

            auto xo = -xp;
            auto yo = -yp;
            auto zo = -zp;

            auto ol = sqrt(dot_3(xo, yo, zo, xo, yo, zo));
            xo = xo / ol;
            yo = yo / ol;
            zo = zo / ol;

            auto ro = clamp(ol, 0.F, r);

            auto xn = xp + ro * xo;
            auto yn = yp + ro * yo;
            auto zn = zp + ro * zo;

            return sqrt(xn * xn + yn * yn + zn * zn);
        }

        Cylinder() = default;

        explicit Cylinder(
            DataT x1,
            DataT y1,
            DataT z1,  //
            DataT xv,
            DataT yv,
            DataT zv,  //
            DataT r,
            DataT rdv)
          : Shape<DataT>(), x1(x1), y1(y1), z1(z1), xv(xv), yv(yv), zv(zv), r(r), rdv(rdv)
        {
            Shape<DataT>::min_distance = compute_min_distance();
        }

        template <typename OtherDataT>
        explicit Cylinder(const Cylinder<OtherDataT> &other)
          : Shape<DataT>(other)
          , x1(other.x1)
          , y1(other.y1)
          , z1(other.z1)
          , xv(other.xv)
          , yv(other.yv)
          , zv(other.zv)
          , r(other.r)
          , rdv(other.rdv)
        {
        }
    };

    template <typename DataT>
    using Capsule = Cylinder<DataT>;

    // A sphere shape, represented as its center (x, y, z) and radius (r)
    template <typename DataT>
    struct Sphere : public Shape<DataT>
    {
        DataT x;
        DataT y;
        DataT z;
        DataT r;

        Sphere() = default;

        explicit Sphere(DataT x, DataT y, DataT z, DataT r) : Shape<DataT>(), x(x), y(y), z(z), r(r)
        {
            Shape<DataT>::min_distance = sqrt(x * x + y * y + z * z) - r;
        }

        template <typename OtherDataT>
        explicit Sphere(const Sphere<OtherDataT> &other)
          : Shape<DataT>(other), x(other.x), y(other.y), z(other.z), r(other.r)
        {
        }
    };
}  // namespace ppln::collision

/////////////////////////// scene.yaml ///////////////////////////

world:
  collision_objects:
    - id: Can1
      primitives:
        - type: cylinder
          dimensions: [0.14, 0.03]
      header:
        frame_id: base_link
      primitive_poses:
        - position: [0.6288254387277676, -0.6633360167754826, 1.073041363431408]
          orientation: [0, 0, -0.4062511770401913, 0.9137614465239049]
    - header:
        frame_id: base_link
      primitive_poses:
        - orientation: [0, 0, -0.4062511770401913, 0.9137614465239049]
          position: [0.1002061383045733, -0.8709406815177271, 1.073041363431408]
      id: Can2
      primitives:
        - type: cylinder
          dimensions: [0.14, 0.03]
    - id: Can3
      primitives:
        - type: cylinder
          dimensions: [0.14, 0.03]
      primitive_poses:
        - position: [0.5140673441348387, -0.2281167854724374, 1.073041363431408]
          orientation: [0, 0, -0.4062511770401913, 0.9137614465239049]
      header:
        frame_id: base_link
    - id: shelf_bottom
      primitives:
        - type: box
          dimensions: [1.2, 1, 0.04]
      header:
        frame_id: base_link
      primitive_poses:
        - position: [0.6092300653384217, -0.8157097421788335, 0.993041363431408]
          orientation: [0, 0, -0.4062511770401913, 0.9137614465239049]
    - header:
        frame_id: base_link
      primitive_poses:
        - position: [0.6092300653384217, -0.8157097421788335, 1.293041363431408]
          orientation: [0, 0, -0.4062511770401913, 0.9137614465239049]
      id: shelf_top
      primitives:
        - type: box
          dimensions: [1.2, 1, 0.04]
    - header:
        frame_id: base_link
      id: side_left
      primitives:
        - type: box
          dimensions: [1.2, 0.04, 0.34]
      primitive_poses:
        - orientation: [0, 0, -0.4062511770401913, 0.9137614465239049]
          position: [0.2380134021541376, -1.150669723332293, 1.143041363431408]
    - header:
        frame_id: base_link
      id: side_right
      primitives:
        - type: box
          dimensions: [1.2, 0.04, 0.34]
      primitive_poses:
        - orientation: [0, 0, -0.4062511770401913, 0.9137614465239049]
          position: [0.9804467285227059, -0.4807497610253744, 1.143041363431408]
///////////////////////////////////////////////////////////////////////////////////////////
// Help me to write a function to load the environment from the scene.yaml file and create the environment object
////////////////////////////////////////////////////////////////////////////////////////

goal_constraints:
  - joint_constraints:
      - joint_name: torso_lift_joint
        position: 0.05580749394926036
      - joint_name: shoulder_pan_joint
        position: 0.2319594187719277
      - joint_name: shoulder_lift_joint
        position: -0.7632272745271215
      - position: 0.7273950815863892
        joint_name: upperarm_roll_joint
      - joint_name: elbow_flex_joint
        position: 1.421938462868271
      - joint_name: forearm_roll_joint
        position: 2.57193125373631
      - joint_name: wrist_flex_joint
        position: 0.4549567580598895
      - joint_name: wrist_roll_joint
        position: -3.141592599877235
planner_id: BKPIECEGood
group_name: arm_with_torso
num_planning_attempts: 2
allowed_planning_time: 60
max_velocity_scaling_factor: 0
max_acceleration_scaling_factor: 0
workspace_parameters:
  header:
    frame_id: ""
  min_corner: [-1, -1, -1]
  max_corner: [1, 1, 1]
start_state:
  joint_state:
    name: [l_wheel_joint, r_wheel_joint, torso_lift_joint, bellows_joint, head_pan_joint, head_tilt_joint, shoulder_pan_joint, shoulder_lift_joint, upperarm_roll_joint, elbow_flex_joint, forearm_roll_joint, wrist_flex_joint, wrist_roll_joint, l_gripper_finger_joint, r_gripper_finger_joint]
    position: [0, 0, 0.1, 0.05, 0, 0, 1.32, 1.4, -0.2, 1.72, 0, 1.66, 0, 0.05, 0.05]
    header:
      frame_id: base_link
/////////////////////////////////////////////////////////////////////////////////////////////////
// write a function to load start and goal values. Start values in under start_state's joint_state, while goal values are under goal_constraints
/////////////////////////////////////////////////////////////////////////////////////////////////