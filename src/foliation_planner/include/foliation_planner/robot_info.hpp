#pragma once // include only once

#include <yaml-cpp/yaml.h>

#include "moveit/robot_model/robot_model.h"

/**
A class which take robot model and generate robot information for cudampl.
 */
class RobotInfo
{
    public:
    RobotInfo(
        const moveit::core::RobotModelConstPtr& robot_model, 
        const std::string & group_name, 
        const std::string & collision_spheres_file_path, 
        const std::map<std::string, float> default_joint_values_map = std::map<std::string, float>()
    )
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

        // get the end effector link names
        const std::vector< const moveit::core::JointModelGroup * > end_effector_joint_model_groups = robot_model->getEndEffectors();
        for (size_t i = 0; i < end_effector_joint_model_groups.size(); i++)
        {
            for (std::string link_name : end_effector_joint_model_groups[i]->getLinkModelNames())
            {
                end_effector_link_names.push_back(link_name);
            }
        }

        if (! loadCollisionSpheres(collision_spheres_file_path)) // this requires the link_names is generated.
        {
            // RCLCPP_ERROR(LOGGER, "Failed to load collision spheres from file");
            std::cout << "\033[1;31mFailed to load collision spheres from file\033[0m" << std::endl;
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
            const moveit::core::LinkModel* link_model = robot_model->getLinkModel(link_name);
            if (link_model->getParentLinkModel() != nullptr) // if it is not the root link
            {
                std::string parent_link_name = link_model->getParentLinkModel()->getName();

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
                joint_name_to_parent_link.push_back(joint_model->getName());
                joint_types.push_back(joint_model->getType());

                if (joint_model->getType() == moveit::core::JointModel::REVOLUTE)
                {
                    const moveit::core::RevoluteJointModel* revolute_joint_model = dynamic_cast<const moveit::core::RevoluteJointModel*>(joint_model);
                    joint_axes.push_back(revolute_joint_model->getAxis());
                    if (default_joint_values_map.find(joint_model->getName()) != default_joint_values_map.end())
                    {
                        default_joint_values.push_back(default_joint_values_map.at(joint_model->getName()));
                    }
                    else
                    {
                        default_joint_values.push_back((float)default_joint_values_double[non_fixed_joint_index]);
                    }
                    non_fixed_joint_index++;
                }
                else if (joint_model->getType() == moveit::core::JointModel::PRISMATIC)
                {
                    const moveit::core::PrismaticJointModel* prismatic_joint_model = dynamic_cast<const moveit::core::PrismaticJointModel*>(joint_model);
                    joint_axes.push_back(prismatic_joint_model->getAxis());
                    if (default_joint_values_map.find(joint_model->getName()) != default_joint_values_map.end())
                    {
                        default_joint_values.push_back(default_joint_values_map.at(joint_model->getName()));
                    }
                    else
                    {
                        default_joint_values.push_back((float)default_joint_values_double[non_fixed_joint_index]);
                    }
                    non_fixed_joint_index++;
                }
                else
                {
                    joint_axes.push_back(Eigen::Vector3d::Zero());
                    default_joint_values.push_back(0.0);
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
        }
    
        const moveit::core::JointModelGroup* joint_model_group = robot_model->getJointModelGroup(group_name);
        const std::vector<std::string>& active_joint_model_names = joint_model_group->getActiveJointModelNames();
        end_link_name_of_model = joint_model_group->getLinkModelNames().back();
        
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
    }

    // Delegating constructor for non-const pointer.
    RobotInfo(
        const moveit::core::RobotModelPtr& robot_model,
        const std::string & group_name,
        const std::string & collision_spheres_file_path,
        const std::map<std::string, float> default_joint_values_map = std::map<std::string, float>()
    )
    : RobotInfo(moveit::core::RobotModelConstPtr(robot_model), group_name, collision_spheres_file_path, default_joint_values_map)
    {
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
                // print each collision_sphere of the collision sphere
                // std::cout << collision_sphere.first.as<std::string>() << std::endl;

                std::string collision_sphere_link_name = collision_sphere.first.as<std::string>();

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
                    std::cout << "\033[1;31mCollision sphere link " << collision_sphere_link_name << " is not in the link names\033[0m" << std::endl;
                    // print all link names
                    std::cout << "\033[1;31mLink names: \033[0m" << std::endl;
                    for (const auto& link_name : link_names){
                        std::cout << link_name << " ";
                    }
                    std::cout << std::endl;
                    continue;
                }

                // print each value of the collision_sphere
                for (const auto& value : collision_sphere.second){
                    // std::cout << "center " << value["center"][0] << " " << value["center"][1] << " " << value["center"][2] << " " << value["center"][3] << " radius " << value["radius"] << std::endl;
                    collision_spheres_map.push_back(collision_sphere_link_index);
                    collision_spheres_pos.push_back({value["center"][0].as<float>(), value["center"][1].as<float>(), value["center"][2].as<float>()});
                    collision_spheres_radius.push_back(value["radius"].as<float>());
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

    std::vector<std::string> getJointNames() const
    {
        return joint_name_to_parent_link;
    }

    std::string getEndEffectorLinkName() const
    {
        // return the last link of link_names
        return end_link_name_of_model;
    }

    std::vector<std::string> getEndEffectorLinkNames() const
    {
        return end_effector_link_names;
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
    std::string end_link_name_of_model;
    std::vector<std::string> end_effector_link_names;
};