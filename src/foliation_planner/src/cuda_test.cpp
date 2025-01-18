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


#include <CUDAMPLib/multiply.h>
#include <CUDAMPLib/kinematics.h>

#include <yaml-cpp/yaml.h>

static const rclcpp::Logger LOGGER = rclcpp::get_logger("CUDAMPLib");

/**
A class which take robot model and generate robot information for cudampl.
 */
class RobotInfo
{
    public:
    RobotInfo(const moveit::core::RobotModelPtr& robot_model, const std::string & collision_spheres_file_path, bool debug = false)
    {

        

        // Initialize all variables
        joint_types.clear();
        joint_poses.clear();
        joint_axes.clear();
        link_maps.clear();
        link_names.clear();

        // Get all link names
        link_names = robot_model->getLinkModelNames();

        if (! loadCollisionSpheres(collision_spheres_file_path)) // this requires the link_names is generated.
        {
            RCLCPP_ERROR(LOGGER, "Failed to load collision spheres from file");
        }

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
                joint_types.push_back(joint_model->getType());

                if (joint_model->getType() == moveit::core::JointModel::REVOLUTE)
                {
                    const moveit::core::RevoluteJointModel* revolute_joint_model = dynamic_cast<const moveit::core::RevoluteJointModel*>(joint_model);
                    joint_axes.push_back(revolute_joint_model->getAxis());
                    if (debug)
                        std::cout << "Joint axis: " << revolute_joint_model->getAxis().transpose() << std::endl;
                }
                else if (joint_model->getType() == moveit::core::JointModel::PRISMATIC)
                {
                    const moveit::core::PrismaticJointModel* prismatic_joint_model = dynamic_cast<const moveit::core::PrismaticJointModel*>(joint_model);
                    joint_axes.push_back(prismatic_joint_model->getAxis());
                    if (debug)
                        std::cout << "Joint axis: " << prismatic_joint_model->getAxis().transpose() << std::endl;
                }
                else
                {
                    joint_axes.push_back(Eigen::Vector3d::Zero());
                    if (debug)
                        std::cout << "Joint axis: " << Eigen::Vector3d::Zero().transpose() << std::endl;
                }

                // get joint origin transform
                joint_poses.push_back(link_model->getJointOriginTransform());
            }
            else
            {
                joint_types.push_back(0); // 0 means unknown joint type
                joint_poses.push_back(Eigen::Isometry3d::Identity());
                joint_axes.push_back(Eigen::Vector3d::Zero());
                link_maps.push_back(-1);
            }
            if (debug)
                std::cout << " ===================================== " << std::endl;
        }
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
                        collision_spheres.push_back({value["center"][0].as<float>(), value["center"][1].as<float>(), value["center"][2].as<float>(), value["radius"].as<float>()});
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

    void loadCollisionSpheres()
    {

    }

    private:
    std::vector<int> joint_types;
    std::vector<Eigen::Isometry3d> joint_poses;
    std::vector<Eigen::Vector3d> joint_axes;
    std::vector<int> link_maps;
    std::vector<std::string> link_names;
    std::vector<int> collision_spheres_map; // define which link the collision sphere belongs to
    std::vector<std::vector<float>> collision_spheres; // (x, y, z, radius)
};

/***
Randomly generate a set of joint values and pass them to the kin_forward function.
Then, we use the RobotState object to get the link poses and compare them with the link poses generated by the kin_forward function.
 */
void TEST_KINE_FORWARD(const moveit::core::RobotModelPtr & robot_model, rclcpp::Node::SharedPtr node, bool debug = false){

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
    RobotInfo robot_info(robot_model, collision_spheres_file_path, debug);

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

void DISPLAY_ROBOT_STATE_IN_RVIZ(const moveit::core::RobotModelPtr & robot_model, rclcpp::Node::SharedPtr node, bool debug = false)
{

    std::string collision_spheres_file_path;
    node->get_parameter("collision_spheres_file_path", collision_spheres_file_path);
    RCLCPP_INFO(LOGGER, "collision_spheres_file_path: %s", collision_spheres_file_path.c_str());

    RobotInfo robot_info(robot_model, collision_spheres_file_path, debug);

    moveit::core::RobotStatePtr robot_state = std::make_shared<moveit::core::RobotState>(robot_model);

    // print robot model name
    RCLCPP_INFO(LOGGER, "Robot model name: %s", robot_model->getName().c_str());

    //************************************* */
    // // print link names of the robot model
    // const std::vector<std::string>& link_names = robot_model->getLinkModelNames();
    // std::cout << "Link names: " << std::endl;
    // for (const auto& link_name : link_names)
    // {
    //     std::cout << link_name << std::endl;
    //     // get link model
    //     const moveit::core::LinkModel* link_model = robot_model->getLinkModel(link_name);
    //     // get shape of the link
    //     const std::vector<shapes::ShapeConstPtr> shapes = link_model->getShapes();
    //     for (const auto& shape : shapes)
    //     {
    //         std::cout << "shape type: " << shape->type << std::endl;
    //     }
    // }
    //************************************* */

    // random set joint values
    robot_state->setToRandomPositions();
    robot_state->update();

    // Create a robot state publisher
    auto robot_state_publisher = node->create_publisher<moveit_msgs::msg::DisplayRobotState>("cuda_test_robot_state", 1);

    // Create a DisplayRobotState message
    moveit_msgs::msg::DisplayRobotState display_robot_state;
    moveit::core::robotStateToRobotStateMsg(*robot_state, display_robot_state.state);

    // use loop to publish the trajectory
    while (rclcpp::ok())
    {
        // Publish the message
        robot_state_publisher->publish(display_robot_state);
        
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

    // Using the RobotModelLoader, we can construct a planning scene monitor that
    // will create a planning scene, monitors planning scene diffs, and apply the diffs to it's
    // internal planning scene. We then call startSceneMonitor, startWorldGeometryMonitor and
    // startStateMonitor to fully initialize the planning scene monitor
    planning_scene_monitor::PlanningSceneMonitorPtr psm(
        new planning_scene_monitor::PlanningSceneMonitor(cuda_test_node, robot_model_loader));

    /* listen for planning scene messages on topic /XXX and apply them to the internal planning scene
                       the internal planning scene accordingly */
    psm->startSceneMonitor();
    /* listens to changes of world geometry, collision objects, and (optionally) octomaps
                                    world geometry, collision objects and optionally octomaps */
    psm->startWorldGeometryMonitor();
    /* listen to joint state updates as well as changes in attached collision objects
                            and update the internal planning scene accordingly*/
    psm->startStateMonitor();



    // =========================================================================================

    // // print collision_spheres_file_path from ros parameter server
    // std::string collision_spheres_file_path;
    // cuda_test_node->get_parameter("collision_spheres_file_path", collision_spheres_file_path);
    // RCLCPP_INFO(cuda_test_node->get_logger(), "collision_spheres_file_path: %s", collision_spheres_file_path.c_str());

    // TEST_KINE_FORWARD(kinematic_model, cuda_test_node);
    
    DISPLAY_ROBOT_STATE_IN_RVIZ(kinematic_model, cuda_test_node);

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