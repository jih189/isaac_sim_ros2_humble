#include <rclcpp/rclcpp.hpp>
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/robot_model/robot_model.h>
#include <moveit/robot_state/robot_state.h>
#include <moveit/planning_pipeline/planning_pipeline.h>
#include "moveit/planning_interface/planning_interface.h"
#include "moveit/robot_state/conversions.h"
#include <moveit/kinematic_constraints/utils.h>
#include <moveit/planning_scene_monitor/planning_scene_monitor.h>
#include <moveit_msgs/msg/display_robot_state.hpp>
#include <moveit/robot_trajectory/robot_trajectory.h>

// cudampl include
#include <CUDAMPLib/multiply.h>
#include <CUDAMPLib/kinematics.h>
#include <CUDAMPLib/cost.h>
#include <CUDAMPLib/spaces/SingleArmSpace.h>
#include <CUDAMPLib/constraints/EnvConstraint.h>
#include <CUDAMPLib/constraints/SelfCollisionConstraint.h>
#include <CUDAMPLib/tasks/SingleArmTask.h>
#include <CUDAMPLib/planners/RRG.h>
#include <CUDAMPLib/termination/StepTermination.h>

// ompl include
#include <ompl/base/SpaceInformation.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/geometric/planners/rrt/RRTConnect.h>
#include <ompl/geometric/SimpleSetup.h>

namespace ob = ompl::base;
namespace og = ompl::geometric;

#include <yaml-cpp/yaml.h>

static const rclcpp::Logger LOGGER = rclcpp::get_logger("CUDAMPLib");

/**
A class which take robot model and generate robot information for cudampl.
 */
class RobotInfo
{
    public:
    RobotInfo(const moveit::core::RobotModelPtr& robot_model, const std::string & group_name, const std::string & collision_spheres_file_path, bool debug = false)
    {

        // Initialize all variables
        joint_types.clear();
        joint_poses.clear();
        joint_axes.clear();
        link_maps.clear();
        link_names.clear();
        joint_name_to_parent_link.clear();
        collision_spheres_map.clear();
        collision_spheres_pos.clear();
        collision_spheres_radius.clear();
        self_collision_enabled_map.clear();
        active_joint_map.clear();
        upper_bounds.clear();
        lower_bounds.clear();
        default_joint_values.clear();
        dimension = 0;

        // Get all link names
        link_names = robot_model->getLinkModelNames();

        if (! loadCollisionSpheres(collision_spheres_file_path)) // this requires the link_names is generated.
        {
            RCLCPP_ERROR(LOGGER, "Failed to load collision spheres from file");
        }

        // initialize self_collision_enabled_map with true with size of link_names
        self_collision_enabled_map.resize(link_names.size(), std::vector<bool>(link_names.size(), true));
        auto srdf_model_ptr = robot_model->getSRDF();
        // print distabled collision pair
        for (const auto& disabled_collision_pair : srdf_model_ptr->getDisabledCollisionPairs())
        {
            // get index of link1_ and link2_
            int link1_index = -1;
            for (size_t i = 0; i < link_names.size(); i++)
            {
                if (link_names[i] == disabled_collision_pair.link1_)
                {
                    link1_index = i;
                    break;
                }
            }
            int link2_index = -1;
            for (size_t i = 0; i < link_names.size(); i++)
            {
                if (link_names[i] == disabled_collision_pair.link2_)
                {
                    link2_index = i;
                    break;
                }
            }
            if (link1_index == -1 || link2_index == -1)
            {
                // print error message in red
                std::cout << "\033[1;31mDisabled collision pair link name is not in the link names\033[0m" << std::endl;
                continue;
            }

            self_collision_enabled_map[link1_index][link2_index] = false;
            self_collision_enabled_map[link2_index][link1_index] = false;
        }

        std::vector<double> default_joint_values_double;
        // get default joint values
        robot_model->getVariableDefaultPositions(default_joint_values_double);

        size_t non_fixed_joint_index = 0;

        // Ready the input to kin_forward
        for (const auto& link_name : link_names)
        {
            if (debug)
            {
                std::cout << "link name: " << link_name << std::endl;
            }
            
            // print its parent link name
            const moveit::core::LinkModel* link_model = robot_model->getLinkModel(link_name);
            if (link_model->getParentLinkModel() != nullptr) // if it is not the root link
            {
                std::string parent_link_name = link_model->getParentLinkModel()->getName();
                if (debug)
                    std::cout << "Parent Link name: " << parent_link_name << std::endl;

                // find the index of parent_link_name in the link_names
                int parent_link_index = -1;
                for (size_t i = 0; i < link_names.size(); i++)
                {
                    if (link_names[i] == parent_link_name)
                    {
                        parent_link_index = i;
                        break;
                    }
                }
                link_maps.push_back(parent_link_index);
                
                // find joint name to its parent link
                const moveit::core::JointModel* joint_model = link_model->getParentJointModel();
                if (debug)
                {
                    std::cout << "Joint name: " << joint_model->getName() << std::endl;
                    std::cout << "Joint type: " << joint_model->getType() << std::endl;
                }
                joint_name_to_parent_link.push_back(joint_model->getName());
                joint_types.push_back(joint_model->getType());

                if (joint_model->getType() == moveit::core::JointModel::REVOLUTE)
                {
                    const moveit::core::RevoluteJointModel* revolute_joint_model = dynamic_cast<const moveit::core::RevoluteJointModel*>(joint_model);
                    joint_axes.push_back(revolute_joint_model->getAxis());
                    default_joint_values.push_back((float)default_joint_values_double[non_fixed_joint_index]);
                    non_fixed_joint_index++;
                    if (debug)
                        std::cout << "Joint axis: " << revolute_joint_model->getAxis().transpose() << std::endl;
                }
                else if (joint_model->getType() == moveit::core::JointModel::PRISMATIC)
                {
                    const moveit::core::PrismaticJointModel* prismatic_joint_model = dynamic_cast<const moveit::core::PrismaticJointModel*>(joint_model);
                    joint_axes.push_back(prismatic_joint_model->getAxis());
                    default_joint_values.push_back((float)default_joint_values_double[non_fixed_joint_index]);
                    non_fixed_joint_index++;
                    if (debug)
                        std::cout << "Joint axis: " << prismatic_joint_model->getAxis().transpose() << std::endl;
                }
                else
                {
                    joint_axes.push_back(Eigen::Vector3d::Zero());
                    default_joint_values.push_back(0.0);
                    if (debug)
                        std::cout << "Joint axis: " << Eigen::Vector3d::Zero().transpose() << std::endl;
                }

                // get joint origin transform
                joint_poses.push_back(link_model->getJointOriginTransform());
            }
            else
            {
                joint_name_to_parent_link.push_back("");
                joint_types.push_back(0); // 0 means unknown joint type
                joint_poses.push_back(Eigen::Isometry3d::Identity());
                joint_axes.push_back(Eigen::Vector3d::Zero());
                default_joint_values.push_back(0.0);
                link_maps.push_back(-1);
            }
            if (debug)
                std::cout << " ===================================== " << std::endl;
        }
    
        const moveit::core::JointModelGroup* joint_model_group = robot_model->getJointModelGroup(group_name);
        const std::vector<std::string>& active_joint_model_names = joint_model_group->getActiveJointModelNames();
        
        // get the active joint map
        for (const auto& joint_name : joint_name_to_parent_link)
        {
            bool active = false;
            for (const auto& active_joint_name : active_joint_model_names)
            {
                if (joint_name == active_joint_name)
                {
                    dimension++;
                    active = true;
                    break;
                }
            }
            active_joint_map.push_back(active);
        }

        std::vector<float> lower_bounds_temp;
        std::vector<float> upper_bounds_temp;

        // get joint bounds
        const moveit::core::JointBoundsVector& joint_bounds_vector = joint_model_group->getActiveJointModelsBounds();
        for (const std::vector<moveit::core::VariableBounds>* joint_bounds : joint_bounds_vector)
        {
            for (const moveit::core::VariableBounds & joint_bound : *joint_bounds)
            {
                lower_bounds_temp.push_back((float)(joint_bound.min_position_));
                upper_bounds_temp.push_back((float)(joint_bound.max_position_));
            }
        }

        int active_joint_index = 0;
        for (size_t i = 0; i < active_joint_map.size(); i++)
        {
            if (active_joint_map[i])
            {
                lower_bounds.push_back(lower_bounds_temp[active_joint_index]);
                upper_bounds.push_back(upper_bounds_temp[active_joint_index]);
                active_joint_index++;
            }
            else
            {
                lower_bounds.push_back(0.0);
                upper_bounds.push_back(0.0);
            }
        }

        // std::vector<double> default_joint_values_double;
        // // get default joint values
        // robot_model->getVariableDefaultPositions(default_joint_values_double);

        // // for (size_t i = 0; i < default_joint_values_double.size(); i++)
        // // {
        // //     // default_joint_values.push_back((float)default_joint_values_double[i]);
        // //     default_joint_values.push_back((float)i);
        // // }
    }

    bool loadCollisionSpheres(const std::string & collision_spheres_file_path)
    {
        // load collision spheres from file

        if (collision_spheres_file_path.empty()){
            // print error message in red
            std::cout << "\033[1;31mCollision spheres file path is empty\033[0m" << std::endl;
            return false;
        }

        // load a yaml file
        YAML::Node collision_spheres_yaml = YAML::LoadFile(collision_spheres_file_path);

        // check if the yaml file contains collision_spheres
        if (!collision_spheres_yaml["collision_spheres"]){
            // print error message in red
            std::cout << "\033[1;31mNo collision_spheres in the yaml file\033[0m" << std::endl;
            return false;
        }
        else{
            // std::cout << collision_spheres_yaml["collision_spheres"] << std::endl;

            // print each collision sphere
            for (const auto& collision_sphere : collision_spheres_yaml["collision_spheres"]){
                // std::cout << collision_sphere << std::endl;
                // print each key of the collision sphere
                for (const auto& key : collision_sphere){
                    // std::cout << key.first.as<std::string>() << std::endl;

                    std::string collision_sphere_link_name = key.first.as<std::string>();

                    // get collision_sphere_link_name index in link_names
                    int collision_sphere_link_index = -1;
                    for (size_t i = 0; i < link_names.size(); i++)
                    {
                        if (link_names[i] == collision_sphere_link_name)
                        {
                            collision_sphere_link_index = i;
                            break;
                        }
                    }

                    if (collision_sphere_link_index == -1){
                        // print error message in red
                        std::cout << "\033[1;31mCollision sphere link name is not in the link names\033[0m" << std::endl;
                        return false;
                    }

                    // print each value of the key
                    for (const auto& value : key.second){
                        // std::cout << "center " << value["center"][0] << " " << value["center"][1] << " " << value["center"][2] << " " << value["center"][3] << " radius " << value["radius"] << std::endl;
                        collision_spheres_map.push_back(collision_sphere_link_index);
                        collision_spheres_pos.push_back({value["center"][0].as<float>(), value["center"][1].as<float>(), value["center"][2].as<float>()});
                        collision_spheres_radius.push_back(value["radius"].as<float>());
                    }
                }
                // std::cout << collision_sphere[0].first << std::endl;
            }

            // list all keys of collision_spheres_yaml["collision_spheres"]
            // for (const auto& key : collision_spheres_yaml["collision_spheres"]){
            //     std::cout << key.first.as<std::string>() << std::endl;
            // }
            // std::cout << "type: " << collision_spheres_yaml["collision_spheres"].Type() << std::endl;
            // for (const auto& collision_sphere : collision_spheres_yaml["collision_spheres"]){
            //     // each collision sphere is a map
            //     // print map key of the collision sphere
            //     std::cout << "key: " << collision_sphere.first.as<std::string>() << std::endl;
            // }
        }

        return true;
    }

    std::vector<int> getJointTypes() const
    {
        return joint_types;
    }

    std::vector<Eigen::Isometry3d> getJointPoses() const
    {
        return joint_poses;
    }

    std::vector<Eigen::Vector3d> getJointAxes() const
    {
        return joint_axes;
    }

    std::vector<int> getLinkMaps() const
    {
        return link_maps;
    }

    std::vector<std::string> getLinkNames() const
    {
        return link_names;
    }

    std::vector<int> getCollisionSpheresMap() const
    {
        return collision_spheres_map;
    }

    std::vector<std::vector<float>> getCollisionSpheresPos() const
    {
        return collision_spheres_pos;
    }

    std::vector<float> getCollisionSpheresRadius() const
    {
        return collision_spheres_radius;
    }

    std::vector<std::vector<bool>> getSelfCollisionEnabledMap() const
    {
        return self_collision_enabled_map;
    }

    std::vector<bool> getActiveJointMap() const
    {
        return active_joint_map;
    }

    int getDimension() const
    {
        return dimension;
    }

    std::vector<float> getUpperBounds() const
    {
        return upper_bounds;
    }

    std::vector<float> getLowerBounds() const
    {
        return lower_bounds;
    }

    std::vector<float> getDefaultJointValues() const
    {
        return default_joint_values;
    }

    private:
    std::vector<int> joint_types;
    std::vector<Eigen::Isometry3d> joint_poses;
    std::vector<Eigen::Vector3d> joint_axes;
    std::vector<int> link_maps; // the index of parent link in link_names
    std::vector<std::vector<bool>> self_collision_enabled_map;
    std::vector<std::string> link_names;
    std::vector<std::string> joint_name_to_parent_link;
    std::vector<int> collision_spheres_map; // define which link the collision sphere belongs to
    std::vector<std::vector<float>> collision_spheres_pos; // (x, y, z)
    std::vector<float> collision_spheres_radius; // radius
    std::vector<bool> active_joint_map;
    int dimension;
    std::vector<float> upper_bounds;
    std::vector<float> lower_bounds;
    std::vector<float> default_joint_values;
};

/***
    Randomly generate a set of joint values and pass them to the kin_forward function.
    Then, we use the RobotState object to get the link poses and compare them with the link poses generated by the kin_forward function.
 */
void TEST_KINE_FORWARD(const moveit::core::RobotModelPtr & robot_model, const std::string & group_name, rclcpp::Node::SharedPtr node, bool debug = false){

    std::cout << "TEST kine_forward with robot model " << robot_model->getName() << std::endl;

    std::string collision_spheres_file_path;
    node->get_parameter("collision_spheres_file_path", collision_spheres_file_path);
    RCLCPP_INFO(LOGGER, "collision_spheres_file_path: %s", collision_spheres_file_path.c_str());

    moveit::core::RobotStatePtr robot_state = std::make_shared<moveit::core::RobotState>(robot_model);
    const std::vector<std::string>& joint_names = robot_model->getActiveJointModelNames();

    // Generate test set
    std::vector<std::vector<float>> joint_values_test_set;
    for (size_t t = 0; t < 20; t++)
    {
        // Generate sampled configuration
        robot_state->setToRandomPositions();
        robot_state->update(); 

        std::vector<float> sampled_joint_values;
        for (const auto& joint_name : joint_names)
        {
            sampled_joint_values.push_back((float)(robot_state->getJointPositions(joint_name)[0]));
        }

        joint_values_test_set.push_back(sampled_joint_values);
    }

    // Get robot information
    RobotInfo robot_info(robot_model, group_name, collision_spheres_file_path, debug);

    std::vector<std::vector<Eigen::Isometry3d>> link_poses_from_kin_forward;
    // std::vector<std::vector<Eigen::Isometry3d>> link_poses_from_kin_forward_cuda;
    // CUDAMPLib::kin_forward(
    //     joint_values_test_set,
    //     robot_info.getJointTypes(),
    //     robot_info.getJointPoses(),
    //     robot_info.getJointAxes(),
    //     robot_info.getLinkMaps(),
    //     link_poses_from_kin_forward
    // );

    // test cuda
    CUDAMPLib::kin_forward_cuda(
        joint_values_test_set,
        robot_info.getJointTypes(),
        robot_info.getJointPoses(),
        robot_info.getJointAxes(),
        robot_info.getLinkMaps(),
        link_poses_from_kin_forward
    );

    // print link poses
    for (size_t i = 0; i < link_poses_from_kin_forward.size(); i++)
    {
        std::cout << "Test Set " << i << " with joint configuration: ";
        // compute link poses from robot state
        for (size_t j = 0; j < joint_names.size(); j++)
        {
            std::cout << " " << joint_values_test_set[i][j];
            robot_state->setJointPositions(joint_names[j], std::vector<double>{(double)joint_values_test_set[i][j]});
        }
        std::cout << std::endl;
        robot_state->update();

        bool equal = true;

        for (size_t j = 0; j < link_poses_from_kin_forward[i].size(); j++)
        {
            const Eigen::Isometry3d& link_pose_from_robot_state = robot_state->getGlobalLinkTransform(robot_info.getLinkNames()[j]);

            if (debug)
            {
                std::cout << "link name: " << robot_info.getLinkNames()[j] << std::endl;
                std::cout << "link pose from kin_forward: " << std::endl;
                std::cout << link_poses_from_kin_forward[i][j].matrix() << std::endl;
                std::cout << "link pose from robot state: " << std::endl;
                std::cout << link_pose_from_robot_state.matrix() << std::endl;
            }

            // if (link_poses_from_kin_forward[i][j].isApprox(link_pose_from_robot_state, 1e-4))
            // {
            //     // print above text with green color
            //     // std::cout << "\033[1;32mLink " << robot_info.getLinkNames()[j] << " poses are equal\033[0m" << std::endl;
            // }
            // else
            if (not link_poses_from_kin_forward[i][j].isApprox(link_pose_from_robot_state, 1e-4))
            {
                // print parent link
                const moveit::core::LinkModel* link_model = robot_model->getLinkModel(robot_info.getLinkNames()[j]);
                std::string parent_link_name = link_model->getParentLinkModel()->getName();
                std::cout << "Parent Link name: " << parent_link_name << std::endl;
                // print joint name
                const moveit::core::JointModel* joint_model = link_model->getParentJointModel();
                std::cout << "Joint name: " << joint_model->getName() << std::endl;
                std::cout << "Joint type: " << joint_model->getType() << std::endl;
                std::cout << "Joint value: " << joint_values_test_set[i][j] << std::endl;
                std::cout << "parent pose: " << std::endl;
                std::cout << link_poses_from_kin_forward[i][robot_info.getLinkMaps()[j]].matrix() << std::endl;
                std::cout << "joint pose: " << std::endl;
                std::cout << robot_info.getJointPoses()[j].matrix() << std::endl;
                std::cout << "joint axis: " << std::endl;
                std::cout << robot_info.getJointAxes()[j].transpose() << std::endl;

                // print above text with red color
                std::cout << "\033[1;31mLink " << robot_info.getLinkNames()[j] << " poses are not equal\033[0m" << std::endl;
                
                equal = false;
                break;
            }
        }

        if (equal)
        {
            std::cout << "\033[1;32m Task pass \033[0m" << std::endl;
        }
        else
        {
            std::cout << "\033[1;31m Task fail \033[0m" << std::endl;
        }
    }

    robot_state.reset();
}

/***
    Randomly generate a joint configuration and pass it to the kin_forward_collision_spheres_cuda function.
    This function will return the collision spheres positions in base_link frame, then we will 
    visualize the collision spheres in rviz.
 */
void DISPLAY_ROBOT_STATE_IN_RVIZ(const moveit::core::RobotModelPtr & robot_model, const std::string & group_name, rclcpp::Node::SharedPtr node, bool debug = false)
{

    std::string collision_spheres_file_path;
    node->get_parameter("collision_spheres_file_path", collision_spheres_file_path);
    RCLCPP_INFO(LOGGER, "collision_spheres_file_path: %s", collision_spheres_file_path.c_str());

    RobotInfo robot_info(robot_model, group_name, collision_spheres_file_path, debug);

    moveit::core::RobotStatePtr robot_state = std::make_shared<moveit::core::RobotState>(robot_model);
    const std::vector<std::string>& joint_names = robot_model->getActiveJointModelNames();

    // print robot model name
    RCLCPP_INFO(LOGGER, "Robot model name: %s", robot_model->getName().c_str());

    // random set joint values
    robot_state->setToRandomPositions();
    robot_state->update();

    // Generate test set with one configuration.
    std::vector<std::vector<float>> joint_values_test_set;
    std::vector<float> sampled_joint_values;
    for (const auto& joint_name : joint_names)
    {
        sampled_joint_values.push_back((float)(robot_state->getJointPositions(joint_name)[0]));
    }
    joint_values_test_set.push_back(sampled_joint_values);

    std::vector<std::vector<Eigen::Isometry3d>> link_poses_from_kin_forward;
    std::vector<std::vector<std::vector<float>>> collision_spheres_pos_from_kin_forward;

    // test cuda
    CUDAMPLib::kin_forward_collision_spheres_cuda(
        joint_values_test_set,
        robot_info.getJointTypes(),
        robot_info.getJointPoses(),
        robot_info.getJointAxes(),
        robot_info.getLinkMaps(),
        robot_info.getCollisionSpheresMap(),
        robot_info.getCollisionSpheresPos(),
        link_poses_from_kin_forward,
        collision_spheres_pos_from_kin_forward
    );

    std::vector<std::vector<float>> collision_spheres_pos_of_first_config;
    for (const auto & collision_spheres_in_link : collision_spheres_pos_from_kin_forward[0])
    {
        for (size_t i = 0; i < collision_spheres_in_link.size(); i += 3)
        {
            collision_spheres_pos_of_first_config.push_back({collision_spheres_in_link[i], collision_spheres_in_link[i + 1], collision_spheres_in_link[i + 2]});
            // std::cout << "cs pos: " << collision_spheres_in_link[i] << " " << collision_spheres_in_link[i + 1] << " " << collision_spheres_in_link[i + 2] << " radius " << robot_info.getCollisionSpheresRadius()[i] << std::endl;
        }
    }

    // Create marker publisher
    auto marker_publisher = node->create_publisher<visualization_msgs::msg::MarkerArray>("collision_spheres", 1);

    // Create a MarkerArray message
    visualization_msgs::msg::MarkerArray robot_collision_spheres_marker_array;
    for (size_t i = 0; i < collision_spheres_pos_of_first_config.size(); i++)
    {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = "base_link";
        marker.header.stamp = node->now();
        marker.ns = "collision_spheres";
        marker.id = i;
        marker.type = visualization_msgs::msg::Marker::SPHERE;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.pose.position.x = collision_spheres_pos_of_first_config[i][0];
        marker.pose.position.y = collision_spheres_pos_of_first_config[i][1];
        marker.pose.position.z = collision_spheres_pos_of_first_config[i][2];
        marker.pose.orientation.x = 0.0;
        marker.pose.orientation.y = 0.0;
        marker.pose.orientation.z = 0.0;
        marker.pose.orientation.w = 1.0;
        marker.scale.x = 2 * robot_info.getCollisionSpheresRadius()[i];
        marker.scale.y = 2 * robot_info.getCollisionSpheresRadius()[i];
        marker.scale.z = 2 * robot_info.getCollisionSpheresRadius()[i];
        marker.color.a = 1.0; // Don't forget to set the alpha!
        marker.color.r = 0.0;
        marker.color.g = 1.0;
        marker.color.b = 0.0;
        robot_collision_spheres_marker_array.markers.push_back(marker);
    }

    // Create a robot state publisher
    auto robot_state_publisher = node->create_publisher<moveit_msgs::msg::DisplayRobotState>("display_robot_state", 1);

    // Create a DisplayRobotState message
    moveit_msgs::msg::DisplayRobotState display_robot_state;
    moveit::core::robotStateToRobotStateMsg(*robot_state, display_robot_state.state);

    // use loop to publish the trajectory
    while (rclcpp::ok())
    {
        // Publish the message
        robot_state_publisher->publish(display_robot_state);
        marker_publisher->publish(robot_collision_spheres_marker_array);
        
        rclcpp::spin_some(node);

        // sleep for 1 second
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    robot_state.reset();
}

/***
    Testing collision
    Here is the steps:
    1. Randomly generate some balls in base_link frame.
    2. Sample an arm configuration and pass it to kin_forward_collision_spheres_cuda function.
    3. Check self collision.
    4. Check collision with the balls.
    5. Visualize the balls with different color based on collision reasons(self collision or env collision).
 */
void TEST_COLLISIONS(const moveit::core::RobotModelPtr & robot_model, const std::string & group_name, rclcpp::Node::SharedPtr node, bool debug = false)
{

    std::string collision_spheres_file_path;
    node->get_parameter("collision_spheres_file_path", collision_spheres_file_path);
    RCLCPP_INFO(LOGGER, "collision_spheres_file_path: %s", collision_spheres_file_path.c_str());

    RobotInfo robot_info(robot_model, group_name, collision_spheres_file_path, debug);

    moveit::core::RobotStatePtr robot_state = std::make_shared<moveit::core::RobotState>(robot_model);
    const std::vector<std::string>& joint_names = robot_model->getActiveJointModelNames();

    // print robot model name
    RCLCPP_INFO(LOGGER, "Robot model name: %s", robot_model->getName().c_str());

    // // random set joint values
    // robot_state->setToRandomPositions();
    // robot_state->update();

    float obstacle_spheres_radius = 0.06;
    // randomly generate some (obstacle_spheres_radius, obstacle_spheres_radius, obstacle_spheres_radius) size balls in base_link frame in range
    // x range [0.3, 0.6], y range [-0.5, 0.5], z range [0.5, 1.5]
    int num_of_obstacle_spheres = 20;
    std::vector<std::vector<float>> balls_pos;
    std::vector<float> ball_radius;
    for (int i = 0; i < num_of_obstacle_spheres; i++)
    {
        float x = 0.3 * ((float)rand() / RAND_MAX) + 0.3;
        float y = 2.0 * 0.5 * ((float)rand() / RAND_MAX) - 0.5;
        float z = 1.0 * ((float)rand() / RAND_MAX) + 0.5;
        balls_pos.push_back({x, y, z});
        ball_radius.push_back(obstacle_spheres_radius);
    }

    /***************************************** test */

    // Generate test set with one configuration.
    std::vector<std::vector<float>> joint_values_test_set;

    int num_of_sampled_configurations = 100;

    for (int t = 0; t < num_of_sampled_configurations; t++)
    {
        // Generate sampled configuration
        robot_state->setToRandomPositions();
        robot_state->update();

        std::vector<float> sampled_joint_values;
        for (const auto& joint_name : joint_names)
        {
            sampled_joint_values.push_back((float)(robot_state->getJointPositions(joint_name)[0]));
        }
        joint_values_test_set.push_back(sampled_joint_values);
    }

    // create collision cost as a shared pointer
    CUDAMPLib::CollisionCostPtr collision_cost = std::make_shared<CUDAMPLib::CollisionCost>(
        balls_pos,
        ball_radius
    );
    // create self collision cost as a shared pointer
    CUDAMPLib::SelfCollisionCostPtr self_collision_cost = std::make_shared<CUDAMPLib::SelfCollisionCost>(
        robot_info.getCollisionSpheresMap(),
        robot_info.getSelfCollisionEnabledMap()
    );

    std::vector<CUDAMPLib::CostBasePtr> cost_set;
    std::vector<float> cost_of_configurations;
    cost_set.push_back(collision_cost);
    cost_set.push_back(self_collision_cost);
    std::vector<std::vector<std::vector<float>>> collision_spheres_pos_in_base_link_for_debug;

    CUDAMPLib::evaluation_cuda(
        joint_values_test_set,
        robot_info.getJointTypes(),
        robot_info.getJointPoses(),
        robot_info.getJointAxes(),
        robot_info.getLinkMaps(),
        robot_info.getCollisionSpheresMap(),
        robot_info.getCollisionSpheresPos(),
        robot_info.getCollisionSpheresRadius(),
        cost_set,
        cost_of_configurations,
        collision_spheres_pos_in_base_link_for_debug
    );

    int collision_free_configuration_index = -1;
    for (int i = 0; i < num_of_sampled_configurations; i++)
    {
        if (cost_of_configurations[i] == 0.0)
        {
            collision_free_configuration_index = i;
            break;
        }
    }

    if (collision_free_configuration_index == -1)
    {
        std::cout << "=========================No collision free configuration==============================" << std::endl;
        collision_free_configuration_index = 0;
        std::cout << "cost of the first configuration: " << cost_of_configurations[0] << std::endl;
    }

    std::vector<std::vector<float>> collision_spheres_pos_of_selected_config;
    for (const auto & collision_spheres_in_link : collision_spheres_pos_in_base_link_for_debug[collision_free_configuration_index])
    {
        for (size_t i = 0; i < collision_spheres_in_link.size(); i += 3)
        {
            collision_spheres_pos_of_selected_config.push_back({collision_spheres_in_link[i], collision_spheres_in_link[i + 1], collision_spheres_in_link[i + 2]});
        }
    }

    // set the robot state to the collision free configuration
    for (size_t j = 0; j < joint_names.size(); j++)
    {
        robot_state->setJointPositions(joint_names[j], std::vector<double>{(double)joint_values_test_set[collision_free_configuration_index][j]});
    }
    robot_state->update();

    // print collision_free_configuration_index
    std::cout << "collision_free_configuration_index: " << collision_free_configuration_index << std::endl;

    /********************************************** */

    // Create marker publisher
    auto obstacle_marker_publisher = node->create_publisher<visualization_msgs::msg::MarkerArray>("obstacle_collision_spheres", 1);

    // Create a robot state publisher
    auto robot_state_publisher = node->create_publisher<moveit_msgs::msg::DisplayRobotState>("display_robot_state", 1);

    // Create marker publisher
    auto self_marker_publisher = node->create_publisher<visualization_msgs::msg::MarkerArray>("self_collision_spheres", 1);

    // Create a DisplayRobotState message
    moveit_msgs::msg::DisplayRobotState display_robot_state;
    moveit::core::robotStateToRobotStateMsg(*robot_state, display_robot_state.state);

    // Create a self MarkerArray message
    visualization_msgs::msg::MarkerArray robot_collision_spheres_marker_array;
    for (size_t i = 0; i < collision_spheres_pos_of_selected_config.size(); i++)
    {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = "base_link";
        marker.header.stamp = node->now();
        marker.ns = "self_collision_spheres";
        marker.id = i;
        marker.type = visualization_msgs::msg::Marker::SPHERE;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.pose.position.x = collision_spheres_pos_of_selected_config[i][0];
        marker.pose.position.y = collision_spheres_pos_of_selected_config[i][1];
        marker.pose.position.z = collision_spheres_pos_of_selected_config[i][2];
        marker.pose.orientation.x = 0.0;
        marker.pose.orientation.y = 0.0;
        marker.pose.orientation.z = 0.0;
        marker.pose.orientation.w = 1.0;
        marker.scale.x = 2 * robot_info.getCollisionSpheresRadius()[i];
        marker.scale.y = 2 * robot_info.getCollisionSpheresRadius()[i];
        marker.scale.z = 2 * robot_info.getCollisionSpheresRadius()[i];
        marker.color.a = 1.0; // Don't forget to set the alpha!
        marker.color.r = 0.0;
        marker.color.g = 1.0;
        marker.color.b = 0.0;
        robot_collision_spheres_marker_array.markers.push_back(marker);
    }

    // Create a obstacle MarkerArray message
    visualization_msgs::msg::MarkerArray obstacle_collision_spheres_marker_array;
    for (size_t i = 0; i < balls_pos.size(); i++)
    {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = "base_link";
        marker.header.stamp = node->now();
        marker.ns = "obstacle_collision_spheres";
        marker.id = i;
        marker.type = visualization_msgs::msg::Marker::SPHERE;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.pose.position.x = balls_pos[i][0];
        marker.pose.position.y = balls_pos[i][1];
        marker.pose.position.z = balls_pos[i][2];
        marker.pose.orientation.x = 0.0;
        marker.pose.orientation.y = 0.0;
        marker.pose.orientation.z = 0.0;
        marker.pose.orientation.w = 1.0;
        marker.scale.x = 2 * obstacle_spheres_radius;
        marker.scale.y = 2 * obstacle_spheres_radius;
        marker.scale.z = 2 * obstacle_spheres_radius;
        marker.color.a = 1.0; // Don't forget to set the alpha!
        marker.color.r = 1.0;
        marker.color.g = 0.0;
        marker.color.b = 0.0;
        obstacle_collision_spheres_marker_array.markers.push_back(marker);
    }

    // use loop to publish the trajectory
    while (rclcpp::ok())
    {
        // Publish the message
        obstacle_marker_publisher->publish(obstacle_collision_spheres_marker_array);
        self_marker_publisher->publish(robot_collision_spheres_marker_array);
        robot_state_publisher->publish(display_robot_state);
        
        rclcpp::spin_some(node);

        // sleep for 1 second
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    robot_state.reset();
}

void TEST_CUDAMPLib(const moveit::core::RobotModelPtr & robot_model, const std::string & group_name, rclcpp::Node::SharedPtr node, bool debug = false)
{
    std::string collision_spheres_file_path;
    node->get_parameter("collision_spheres_file_path", collision_spheres_file_path);
    RobotInfo robot_info(robot_model, group_name, collision_spheres_file_path, debug);

    // Prepare obstacle constraint
    float obstacle_spheres_radius = 0.06;
    // randomly generate some (obstacle_spheres_radius, obstacle_spheres_radius, obstacle_spheres_radius) size balls in base_link frame in range
    // x range [0.3, 0.6], y range [-0.5, 0.5], z range [0.5, 1.5]
    int num_of_obstacle_spheres = 20;
    std::vector<std::vector<float>> balls_pos;
    std::vector<float> ball_radius;
    for (int i = 0; i < num_of_obstacle_spheres; i++)
    {
        float x = 0.3 * ((float)rand() / RAND_MAX) + 0.3;
        float y = 2.0 * 0.5 * ((float)rand() / RAND_MAX) - 0.5;
        float z = 1.0 * ((float)rand() / RAND_MAX) + 0.5;
        balls_pos.push_back({x, y, z});
        ball_radius.push_back(obstacle_spheres_radius);
    }

    std::vector<CUDAMPLib::BaseConstraintPtr> constraints;

    CUDAMPLib::EnvConstraintPtr env_constraint = std::make_shared<CUDAMPLib::EnvConstraint>(
        "obstacle_constraint",
        balls_pos,
        ball_radius
    );
    constraints.push_back(env_constraint);

    CUDAMPLib::SelfCollisionConstraintPtr self_collision_constraint = std::make_shared<CUDAMPLib::SelfCollisionConstraint>(
        "self_collision_constraint",
        robot_info.getSelfCollisionEnabledMap()
    );
    constraints.push_back(self_collision_constraint);

    // Create space
    CUDAMPLib::SingleArmSpacePtr single_arm_space = std::make_shared<CUDAMPLib::SingleArmSpace>(
        robot_info.getDimension(),
        constraints,
        robot_info.getJointTypes(),
        robot_info.getJointPoses(),
        robot_info.getJointAxes(),
        robot_info.getLinkMaps(),
        robot_info.getCollisionSpheresMap(),
        robot_info.getCollisionSpheresPos(),
        robot_info.getCollisionSpheresRadius(),
        robot_info.getActiveJointMap(),
        robot_info.getLowerBounds(),
        robot_info.getUpperBounds(),
        robot_info.getDefaultJointValues()
    );

    // sample a set of states
    CUDAMPLib::SingleArmStatesPtr sampled_states = std::static_pointer_cast<CUDAMPLib::SingleArmStates>(single_arm_space->sample(10));
    sampled_states->update();

    std::vector<bool> state_feasibility;

    // check states
    single_arm_space->checkStates(sampled_states, state_feasibility);

    std::vector<std::vector<float>> states_joint_values = sampled_states->getJointStatesHost();
    for (size_t i = 0; i < states_joint_values.size(); i++)
    {
        for (size_t j = 0; j < states_joint_values[i].size(); j++)
        {
            std::cout << states_joint_values[i][j] << " ";
        }

        if (state_feasibility[i])
        {
            std::cout << " feasible" << std::endl;
        }
        else
        {
            std::cout << " infeasible" << std::endl;
        }
    }

    std::vector<std::vector<std::vector<float>>> self_collision_spheres_pos =  sampled_states->getSelfCollisionSpheresPosInBaseLinkHost();

    std::vector<std::vector<float>> collision_spheres_pos_of_selected_config = self_collision_spheres_pos[0];
    bool is_selected_config_feasible = state_feasibility[0];

    // Create marker publisher
    auto self_marker_publisher = node->create_publisher<visualization_msgs::msg::MarkerArray>("self_collision_spheres", 1);
    
    auto obstacle_marker_publisher = node->create_publisher<visualization_msgs::msg::MarkerArray>("obstacle_collision_spheres", 1);

    // Create a self MarkerArray message
    visualization_msgs::msg::MarkerArray robot_collision_spheres_marker_array;
    for( size_t i = 0 ; i < collision_spheres_pos_of_selected_config.size(); i++)
    {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = "base_link";
        marker.header.stamp = node->now();
        marker.ns = "self_collision_spheres";
        marker.id = i;
        marker.type = visualization_msgs::msg::Marker::SPHERE;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.pose.position.x = collision_spheres_pos_of_selected_config[i][0];
        marker.pose.position.y = collision_spheres_pos_of_selected_config[i][1];
        marker.pose.position.z = collision_spheres_pos_of_selected_config[i][2];
        marker.pose.orientation.x = 0.0;
        marker.pose.orientation.y = 0.0;
        marker.pose.orientation.z = 0.0;
        marker.pose.orientation.w = 1.0;
        marker.scale.x = 2 * robot_info.getCollisionSpheresRadius()[i];
        marker.scale.y = 2 * robot_info.getCollisionSpheresRadius()[i];
        marker.scale.z = 2 * robot_info.getCollisionSpheresRadius()[i];
        marker.color.a = 1.0; // Don't forget to set the alpha!
        if (is_selected_config_feasible)
        {
            marker.color.r = 0.0;
            marker.color.g = 1.0;
            marker.color.b = 0.0;
        }
        else
        {
            marker.color.r = 1.0;
            marker.color.g = 0.0;
            marker.color.b = 0.0;
        }
        robot_collision_spheres_marker_array.markers.push_back(marker);
    }

    // Create a obstacle MarkerArray message
    visualization_msgs::msg::MarkerArray obstacle_collision_spheres_marker_array;
    for (size_t i = 0; i < balls_pos.size(); i++)
    {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = "base_link";
        marker.header.stamp = node->now();
        marker.ns = "obstacle_collision_spheres";
        marker.id = i;
        marker.type = visualization_msgs::msg::Marker::SPHERE;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.pose.position.x = balls_pos[i][0];
        marker.pose.position.y = balls_pos[i][1];
        marker.pose.position.z = balls_pos[i][2];
        marker.pose.orientation.x = 0.0;
        marker.pose.orientation.y = 0.0;
        marker.pose.orientation.z = 0.0;
        marker.pose.orientation.w = 1.0;
        marker.scale.x = 2 * ball_radius[i];
        marker.scale.y = 2 * ball_radius[i];
        marker.scale.z = 2 * ball_radius[i];
        marker.color.a = 1.0; // Don't forget to set the alpha!
        marker.color.r = 0.5;
        marker.color.g = 0.5;
        marker.color.b = 0.0;
        obstacle_collision_spheres_marker_array.markers.push_back(marker);
    }

    // use loop to publish the trajectory
    while (rclcpp::ok())
    {
        // Publish the message
        self_marker_publisher->publish(robot_collision_spheres_marker_array);

        obstacle_marker_publisher->publish(obstacle_collision_spheres_marker_array);
        
        rclcpp::spin_some(node);

        // sleep for 1 second
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

void TEST_Planner(const moveit::core::RobotModelPtr & robot_model, const std::string & group_name, rclcpp::Node::SharedPtr node, bool debug = false)
{
    std::string collision_spheres_file_path;
    node->get_parameter("collision_spheres_file_path", collision_spheres_file_path);
    RobotInfo robot_info(robot_model, group_name, collision_spheres_file_path, debug);

    // Prepare obstacle constraint
    float obstacle_spheres_radius = 0.06;
    // randomly generate some (obstacle_spheres_radius, obstacle_spheres_radius, obstacle_spheres_radius) size balls in base_link frame in range
    // x range [0.3, 0.6], y range [-0.5, 0.5], z range [0.5, 1.5]
    int num_of_obstacle_spheres = 20;
    std::vector<std::vector<float>> balls_pos;
    std::vector<float> ball_radius;
    for (int i = 0; i < num_of_obstacle_spheres; i++)
    {
        float x = 0.3 * ((float)rand() / RAND_MAX) + 0.3;
        float y = 2.0 * 0.5 * ((float)rand() / RAND_MAX) - 0.5;
        float z = 1.0 * ((float)rand() / RAND_MAX) + 0.5;
        balls_pos.push_back({x, y, z});
        ball_radius.push_back(obstacle_spheres_radius);
    }

    // create planning scene
    auto world = std::make_shared<collision_detection::World>();
    auto planning_scene = std::make_shared<planning_scene::PlanningScene>(robot_model, world);

    // add those balls to the planning scene
    for (int i = 0; i < num_of_obstacle_spheres; i++)
    {
        Eigen::Isometry3d sphere_pose = Eigen::Isometry3d::Identity();
        sphere_pose.translation() = Eigen::Vector3d(balls_pos[i][0], balls_pos[i][1], balls_pos[i][2]);
        planning_scene->getWorldNonConst()->addToObject("obstacle_" + std::to_string(i), shapes::ShapeConstPtr(new shapes::Sphere(ball_radius[i])), sphere_pose);
    }

    moveit::core::RobotStatePtr robot_state = std::make_shared<moveit::core::RobotState>(robot_model);
    const moveit::core::JointModelGroup* joint_model_group = robot_model->getJointModelGroup(group_name);
    
    // generate random state on the joint model group
    robot_state->setToRandomPositions(joint_model_group);
    robot_state->update();

    while(not planning_scene->isStateValid(*robot_state, group_name))
    {
        robot_state->setToRandomPositions(joint_model_group);
        robot_state->update();
    }

    std::vector<double> start_joint_values_double;
    std::vector<float> start_joint_values;
    robot_state->copyJointGroupPositions(joint_model_group, start_joint_values_double);
    for (size_t i = 0; i < start_joint_values_double.size(); i++)
    {
        start_joint_values.push_back((float)start_joint_values_double[i]);
    }

    // set start state
    moveit_msgs::msg::RobotState start_state_msg;
    moveit::core::robotStateToRobotStateMsg(*robot_state, start_state_msg);
    
    // generate random state on the joint model group
    robot_state->setToRandomPositions(joint_model_group);
    robot_state->update();

    while(not planning_scene->isStateValid(*robot_state, group_name))
    {
        robot_state->setToRandomPositions(joint_model_group);
        robot_state->update();
    }

    std::vector<double> goal_joint_values_double;
    std::vector<float> goal_joint_values;
    robot_state->copyJointGroupPositions(joint_model_group, goal_joint_values_double);
    for (size_t i = 0; i < goal_joint_values_double.size(); i++)
    {
        goal_joint_values.push_back((float)goal_joint_values_double[i]);
    }

    // set goal state
    moveit_msgs::msg::RobotState goal_state_msg;
    moveit::core::robotStateToRobotStateMsg(*robot_state, goal_state_msg);

    /**************************************************************************************************** */

    std::vector<CUDAMPLib::BaseConstraintPtr> constraints;

    CUDAMPLib::EnvConstraintPtr env_constraint = std::make_shared<CUDAMPLib::EnvConstraint>(
        "obstacle_constraint",
        balls_pos,
        ball_radius
    );
    // skip it for now.
    constraints.push_back(env_constraint);

    CUDAMPLib::SelfCollisionConstraintPtr self_collision_constraint = std::make_shared<CUDAMPLib::SelfCollisionConstraint>(
        "self_collision_constraint",
        robot_info.getSelfCollisionEnabledMap()
    );
    constraints.push_back(self_collision_constraint);

    // Create space
    CUDAMPLib::SingleArmSpacePtr single_arm_space = std::make_shared<CUDAMPLib::SingleArmSpace>(
        robot_info.getDimension(),
        constraints,
        robot_info.getJointTypes(),
        robot_info.getJointPoses(),
        robot_info.getJointAxes(),
        robot_info.getLinkMaps(),
        robot_info.getCollisionSpheresMap(),
        robot_info.getCollisionSpheresPos(),
        robot_info.getCollisionSpheresRadius(),
        robot_info.getActiveJointMap(),
        robot_info.getLowerBounds(),
        robot_info.getUpperBounds(),
        robot_info.getDefaultJointValues()
    );

    std::vector<std::vector<float>> start_joint_values_set;
    start_joint_values_set.push_back(start_joint_values);

    std::vector<std::vector<float>> goal_joint_values_set;
    goal_joint_values_set.push_back(goal_joint_values);
    // create the task
    CUDAMPLib::SingleArmTaskPtr task = std::make_shared<CUDAMPLib::SingleArmTask>(
        start_joint_values_set,
        goal_joint_values_set
    );

    // create the planner
    CUDAMPLib::RRGPtr planner = std::make_shared<CUDAMPLib::RRG>(single_arm_space);

    // set the task
    planner->setMotionTask(task);

    // create termination condition
    CUDAMPLib::StepTerminationPtr termination_condition = std::make_shared<CUDAMPLib::StepTermination>(1);

    // solve the task
    planner->solve(termination_condition);

    moveit_msgs::msg::DisplayTrajectory display_trajectory;
    moveit_msgs::msg::RobotTrajectory robot_trajectory_msg;
    auto solution_robot_trajectory = robot_trajectory::RobotTrajectory(robot_model, joint_model_group);

    if (task->hasSolution())
    {
        // print "Task solved" in green color
        std::cout << "\033[1;32m" << "Task solved" << "\033[0m" << std::endl;

        std::vector<std::vector<float>> solution_path = task->getSolution();

        moveit_msgs::msg::RobotState start_state_msg;

        // generate robot trajectory msg
        for (size_t i = 0; i < solution_path.size(); i++)
        {
            // convert solution_path[i] to double vector
            std::vector<double> solution_path_i_double = std::vector<double>(solution_path[i].begin(), solution_path[i].end());
            robot_state->setJointGroupPositions(joint_model_group, solution_path_i_double);
            solution_robot_trajectory.addSuffixWayPoint(*robot_state, 10.0);

            if (i == 0)
            {
                // set start state
                moveit::core::robotStateToRobotStateMsg(*robot_state, start_state_msg);
            }
        }
        // Create a DisplayTrajectory message
        solution_robot_trajectory.getRobotTrajectoryMsg(robot_trajectory_msg);

        display_trajectory.trajectory_start = start_state_msg;
        display_trajectory.trajectory.push_back(robot_trajectory_msg);
    }
    else
    {
        // print "Task not solved" in red color
        std::cout << "\033[1;31m" << "Task not solved" << "\033[0m" << std::endl;
    }

    /**************************************************************************************************** */

    // Create a start robot state publisher
    auto start_robot_state_publisher = node->create_publisher<moveit_msgs::msg::DisplayRobotState>("start_robot_state", 1);

    // Create a goal robot state publisher
    auto goal_robot_state_publisher = node->create_publisher<moveit_msgs::msg::DisplayRobotState>("goal_robot_state", 1);

    auto obstacle_marker_publisher = node->create_publisher<visualization_msgs::msg::MarkerArray>("obstacle_collision_spheres", 1);

    std::shared_ptr<rclcpp::Publisher<moveit_msgs::msg::DisplayTrajectory>> display_publisher =
        node->create_publisher<moveit_msgs::msg::DisplayTrajectory>("/display_planned_path", 1);

    // Create a DisplayRobotState message
    moveit_msgs::msg::DisplayRobotState start_display_robot_state;
    start_display_robot_state.state = start_state_msg;

    // Create a DisplayRobotState message
    moveit_msgs::msg::DisplayRobotState goal_display_robot_state;
    goal_display_robot_state.state = goal_state_msg;

    // Create a obstacle MarkerArray message
    visualization_msgs::msg::MarkerArray obstacle_collision_spheres_marker_array;
    for (size_t i = 0; i < balls_pos.size(); i++)
    {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = "base_link";
        marker.header.stamp = node->now();
        marker.ns = "obstacle_collision_spheres";
        marker.id = i;
        marker.type = visualization_msgs::msg::Marker::SPHERE;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.pose.position.x = balls_pos[i][0];
        marker.pose.position.y = balls_pos[i][1];
        marker.pose.position.z = balls_pos[i][2];
        marker.pose.orientation.x = 0.0;
        marker.pose.orientation.y = 0.0;
        marker.pose.orientation.z = 0.0;
        marker.pose.orientation.w = 1.0;
        marker.scale.x = 2 * ball_radius[i];
        marker.scale.y = 2 * ball_radius[i];
        marker.scale.z = 2 * ball_radius[i];
        marker.color.a = 1.0; // Don't forget to set the alpha!
        marker.color.r = 0.5;
        marker.color.g = 0.5;
        marker.color.b = 0.0;
        obstacle_collision_spheres_marker_array.markers.push_back(marker);
    }

    std::cout << "publishing start and goal robot state" << std::endl;

    // Publish the message in a loop
    while (rclcpp::ok())
    {
        // Publish the message
        start_robot_state_publisher->publish(start_display_robot_state);
        goal_robot_state_publisher->publish(goal_display_robot_state);
        obstacle_marker_publisher->publish(obstacle_collision_spheres_marker_array);

        if (task->hasSolution())
        {
            display_publisher->publish(display_trajectory);
        }
        
        rclcpp::spin_some(node);

        // sleep for 1 second
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    // clear the robot state
    robot_state.reset();
}

std::vector<std::vector<double>> interpolate(const std::vector<double> & start, const std::vector<double> & goal, int num_points)
{
    std::vector<std::vector<double>> trajectory;
    for (int i = 0; i < num_points; ++i)
    {
        std::vector<double> point;
        for (size_t j = 0; j < start.size(); ++j)
        {
            double value = start[j] + (goal[j] - start[j]) * static_cast<double>(i) / static_cast<double>(num_points - 1);
            point.push_back(value);
        }
        trajectory.push_back(point);
    }
    return trajectory;
}

void TEST_OMPL(const moveit::core::RobotModelPtr & robot_model, const std::string & group_name, rclcpp::Node::SharedPtr node, bool debug = false)
{
    std::string collision_spheres_file_path;
    node->get_parameter("collision_spheres_file_path", collision_spheres_file_path);
    RobotInfo robot_info(robot_model, group_name, collision_spheres_file_path, debug);

    // Prepare obstacle constraint
    float obstacle_spheres_radius = 0.06;
    // randomly generate some (obstacle_spheres_radius, obstacle_spheres_radius, obstacle_spheres_radius) size balls in base_link frame in range
    // x range [0.3, 0.6], y range [-0.5, 0.5], z range [0.5, 1.5]
    int num_of_obstacle_spheres = 20;
    std::vector<std::vector<float>> balls_pos;
    std::vector<float> ball_radius;
    for (int i = 0; i < num_of_obstacle_spheres; i++)
    {
        float x = 0.3 * ((float)rand() / RAND_MAX) + 0.3;
        float y = 2.0 * 0.5 * ((float)rand() / RAND_MAX) - 0.5;
        float z = 1.0 * ((float)rand() / RAND_MAX) + 0.5;
        balls_pos.push_back({x, y, z});
        ball_radius.push_back(obstacle_spheres_radius);
    }

    // create planning scene
    auto world = std::make_shared<collision_detection::World>();
    auto planning_scene = std::make_shared<planning_scene::PlanningScene>(robot_model, world);

    // add those balls to the planning scene
    for (int i = 0; i < num_of_obstacle_spheres; i++)
    {
        Eigen::Isometry3d sphere_pose = Eigen::Isometry3d::Identity();
        sphere_pose.translation() = Eigen::Vector3d(balls_pos[i][0], balls_pos[i][1], balls_pos[i][2]);
        planning_scene->getWorldNonConst()->addToObject("obstacle_" + std::to_string(i), shapes::ShapeConstPtr(new shapes::Sphere(ball_radius[i])), sphere_pose);
    }

    moveit::core::RobotStatePtr robot_state = std::make_shared<moveit::core::RobotState>(robot_model);
    const moveit::core::JointModelGroup* joint_model_group = robot_model->getJointModelGroup(group_name);

    // generate random state on the joint model group
    robot_state->setToRandomPositions(joint_model_group);
    robot_state->update();

    while(not planning_scene->isStateValid(*robot_state, group_name))
    {
        robot_state->setToRandomPositions(joint_model_group);
        robot_state->update();
    }

    std::vector<double> start_joint_values_double;
    std::vector<float> start_joint_values;
    robot_state->copyJointGroupPositions(joint_model_group, start_joint_values_double);
    for (size_t i = 0; i < start_joint_values_double.size(); i++)
    {
        start_joint_values.push_back((float)start_joint_values_double[i]);
    }

    // set start state
    moveit_msgs::msg::RobotState start_state_msg;
    moveit::core::robotStateToRobotStateMsg(*robot_state, start_state_msg);
    
    // generate random state on the joint model group
    robot_state->setToRandomPositions(joint_model_group);
    robot_state->update();

    while(not planning_scene->isStateValid(*robot_state, group_name))
    {
        robot_state->setToRandomPositions(joint_model_group);
        robot_state->update();
    }

    std::vector<double> goal_joint_values_double;
    std::vector<float> goal_joint_values;
    robot_state->copyJointGroupPositions(joint_model_group, goal_joint_values_double);
    for (size_t i = 0; i < goal_joint_values_double.size(); i++)
    {
        goal_joint_values.push_back((float)goal_joint_values_double[i]);
    }

    // set goal state
    moveit_msgs::msg::RobotState goal_state_msg;
    moveit::core::robotStateToRobotStateMsg(*robot_state, goal_state_msg);

    // print "start and goal state"
    std::cout << "start state: ";
    for (size_t i = 0; i < start_joint_values.size(); i++)
    {
        std::cout << start_joint_values[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "goal state: ";
    for (size_t i = 0; i < goal_joint_values.size(); i++)
    {
        std::cout << goal_joint_values[i] << " ";
    }
    std::cout << std::endl;

    // set group dimension
    int dim = robot_model->getJointModelGroup(group_name)->getActiveJointModels().size();
    std::cout << "dim: " << dim << std::endl;

    // create ompl states for start and goal
    ompl::base::ScopedState<> start_state(ompl::base::StateSpacePtr(new ompl::base::RealVectorStateSpace(dim)));
    ompl::base::ScopedState<> goal_state(ompl::base::StateSpacePtr(new ompl::base::RealVectorStateSpace(dim)));

    for (int i = 0; i < dim; i++)
    {
        start_state[i] = start_joint_values[i];
        goal_state[i] = goal_joint_values[i];
    }

    // create ompl space
    ompl::base::StateSpacePtr space(new ompl::base::RealVectorStateSpace(dim));

    std::vector<float> upper_bounds_of_active_joints;
    std::vector<float> lower_bounds_of_active_joints;

    // based on active joint map, set the bounds
    for (size_t i = 0; i < robot_info.getActiveJointMap().size(); i++)
    {
        if (robot_info.getActiveJointMap()[i])
        {
            upper_bounds_of_active_joints.push_back(robot_info.getUpperBounds()[i]);
            lower_bounds_of_active_joints.push_back(robot_info.getLowerBounds()[i]);
        }
    }

    // set bounds
    ompl::base::RealVectorBounds bounds(dim);
    for (int i = 0; i < dim; i++)
    {
        bounds.setLow(i, lower_bounds_of_active_joints[i]);
        bounds.setHigh(i, upper_bounds_of_active_joints[i]);
    }
    space->as<ompl::base::RealVectorStateSpace>()->setBounds(bounds);

    // create space information
    ompl::base::SpaceInformationPtr si(new ompl::base::SpaceInformation(space));

    // set state validity checker
    si->setStateValidityChecker([&](const ompl::base::State * state) {
        // convert ompl state to robot state
        std::vector<double> joint_values_double;
        for (int i = 0; i < dim; i++)
        {
            joint_values_double.push_back(state->as<ompl::base::RealVectorStateSpace::StateType>()->values[i]);
        }
        robot_state->setJointGroupPositions(joint_model_group, joint_values_double);
        robot_state->update();
        return planning_scene->isStateValid(*robot_state, group_name);
    });

    // set problem definition
    ompl::base::ProblemDefinitionPtr pdef(new ompl::base::ProblemDefinition(si));
    pdef->setStartAndGoalStates(start_state, goal_state);

    // create planner
    auto planner(std::make_shared<og::RRTConnect>(si));
    planner->setProblemDefinition(pdef);
    planner->setup();

    // solve the problem
    ompl::base::PlannerStatus solved = planner->ob::Planner::solve(1.0);

    // generate robot trajectory msg
    moveit_msgs::msg::DisplayTrajectory display_trajectory;
    moveit_msgs::msg::RobotTrajectory robot_trajectory_msg;
    auto solution_robot_trajectory = robot_trajectory::RobotTrajectory(robot_model, joint_model_group);

    if (solved)
    {
        // print "Task solved" in green color
        std::cout << "\033[1;32m" << "Task solved" << "\033[0m" << std::endl;

        // get the path from the planner
        ob::PathPtr path = pdef->getSolutionPath();
        const auto *path_ = path.get()->as<og::PathGeometric>();
        // convert the path to robot trajectory, and do interpolation between states
        const ob::State * state = path_->getState(0);
        std::vector<double> previous_joint_values_double;
        for (int j = 0; j < dim; j++)
        {
            previous_joint_values_double.push_back(state->as<ompl::base::RealVectorStateSpace::StateType>()->values[j]);
        }

        for (size_t i = 1; i < path_->getStateCount(); i++)
        {
            const ob::State * state = path_->getState(i);
            std::vector<double> joint_values_double;
            for (int j = 0; j < dim; j++)
            {
                joint_values_double.push_back(state->as<ompl::base::RealVectorStateSpace::StateType>()->values[j]);
            }
            // interpolate between previous_joint_values_double and joint_values_double
            std::vector<std::vector<double>> interpolated_joint_values_double = interpolate(previous_joint_values_double, joint_values_double, 10);

            // print the interpolated joint values only
            for (size_t k = 0; k < interpolated_joint_values_double.size(); k++)
            {
                robot_state->setJointGroupPositions(joint_model_group, interpolated_joint_values_double[k]);
                solution_robot_trajectory.addSuffixWayPoint(*robot_state, 10.0);
            }

            previous_joint_values_double = joint_values_double;
        }

        // set start state
        moveit::core::robotStateToRobotStateMsg(*robot_state, start_state_msg);
        // Create a DisplayTrajectory message
        solution_robot_trajectory.getRobotTrajectoryMsg(robot_trajectory_msg);
        display_trajectory.trajectory_start = start_state_msg;
        display_trajectory.trajectory.push_back(robot_trajectory_msg);
    }
    else
    {
        // print "Task not solved" in red color
        std::cout << "\033[1;31m" << "Task not solved" << "\033[0m" << std::endl;
    }

    // Create a start robot state publisher
    auto start_robot_state_publisher = node->create_publisher<moveit_msgs::msg::DisplayRobotState>("start_robot_state", 1);
    // Create a goal robot state publisher
    auto goal_robot_state_publisher = node->create_publisher<moveit_msgs::msg::DisplayRobotState>("goal_robot_state", 1);
    auto obstacle_marker_publisher = node->create_publisher<visualization_msgs::msg::MarkerArray>("obstacle_collision_spheres", 1);
    std::shared_ptr<rclcpp::Publisher<moveit_msgs::msg::DisplayTrajectory>> display_publisher =
        node->create_publisher<moveit_msgs::msg::DisplayTrajectory>("/display_planned_path", 1);
    // Create a DisplayRobotState message
    moveit_msgs::msg::DisplayRobotState start_display_robot_state;
    start_display_robot_state.state = start_state_msg;
    // Create a DisplayRobotState message
    moveit_msgs::msg::DisplayRobotState goal_display_robot_state;
    goal_display_robot_state.state = goal_state_msg;
    // Create a obstacle MarkerArray message
    visualization_msgs::msg::MarkerArray obstacle_collision_spheres_marker_array;
    for (size_t i = 0; i < balls_pos.size(); i++)
    {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = "base_link";
        marker.header.stamp = node->now();
        marker.ns = "obstacle_collision_spheres";
        marker.id = i;
        marker.type = visualization_msgs::msg::Marker::SPHERE;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.pose.position.x = balls_pos[i][0];
        marker.pose.position.y = balls_pos[i][1];
        marker.pose.position.z = balls_pos[i][2];
        marker.pose.orientation.x = 0.0;
        marker.pose.orientation.y = 0.0;
        marker.pose.orientation.z = 0.0;
        marker.pose.orientation.w = 1.0;
        marker.scale.x = 2 * ball_radius[i];
        marker.scale.y = 2 * ball_radius[i];
        marker.scale.z = 2 * ball_radius[i];
        marker.color.a = 1.0; // Don't forget to set the alpha!
        marker.color.r = 0.5;
        marker.color.g = 0.5;
        marker.color.b = 0.0;
        obstacle_collision_spheres_marker_array.markers.push_back(marker);
    }

    std::cout << "publishing start and goal robot state" << std::endl;
    // Publish the message in a loop
    while (rclcpp::ok())
    {
        // Publish the message
        start_robot_state_publisher->publish(start_display_robot_state);
        goal_robot_state_publisher->publish(goal_display_robot_state);
        obstacle_marker_publisher->publish(obstacle_collision_spheres_marker_array);

        if (solved)
        {
            display_publisher->publish(display_trajectory);
        }
        
        rclcpp::spin_some(node);

        // sleep for 1 second
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    
    robot_state.reset();
}

int main(int argc, char** argv)
{
    const std::string GROUP_NAME = "arm";

    rclcpp::init(argc, argv);
    rclcpp::NodeOptions node_options;
    node_options.automatically_declare_parameters_from_overrides(true);
    auto cuda_test_node = rclcpp::Node::make_shared("cuda_test_node", node_options);

    // print out the node name
    RCLCPP_INFO(LOGGER, "Node name: %s", cuda_test_node->get_name());

    // Create a robot model
    robot_model_loader::RobotModelLoaderPtr robot_model_loader(
      new robot_model_loader::RobotModelLoader(cuda_test_node, "robot_description"));
    const moveit::core::RobotModelPtr& kinematic_model = robot_model_loader->getModel();
    if (kinematic_model == nullptr)
    {
        RCLCPP_ERROR(cuda_test_node->get_logger(), "Failed to load robot model");
        return 1;
    }

    // =========================================================================================

    // // print collision_spheres_file_path from ros parameter server
    // std::string collision_spheres_file_path;
    // cuda_test_node->get_parameter("collision_spheres_file_path", collision_spheres_file_path);
    // RCLCPP_INFO(cuda_test_node->get_logger(), "collision_spheres_file_path: %s", collision_spheres_file_path.c_str());

    // TEST_KINE_FORWARD(kinematic_model, GROUP_NAME, cuda_test_node);
    
    // DISPLAY_ROBOT_STATE_IN_RVIZ(kinematic_model, GROUP_NAME, cuda_test_node);

    // TEST_COLLISIONS(kinematic_model, GROUP_NAME, cuda_test_node);

    // TEST_CUDAMPLib(kinematic_model, GROUP_NAME, cuda_test_node);

    TEST_Planner(kinematic_model, GROUP_NAME, cuda_test_node);

    // TEST_OMPL(kinematic_model, GROUP_NAME, cuda_test_node);

    // list ros parameters
    // RCLCPP_INFO(cuda_test_node->get_logger(), "List all parameters");
    // auto parameters = cuda_test_node->list_parameters({""}, 5);
    // for (const auto& parameter : parameters.names)
    // {
    //     RCLCPP_INFO(cuda_test_node->get_logger(), "Parameter name: %s", parameter.c_str());
    // }

    // stop the node
    rclcpp::shutdown();

    return 0;
}