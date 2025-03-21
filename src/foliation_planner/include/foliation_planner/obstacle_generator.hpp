#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <urdf/model.h>
#include <Eigen/Dense>
#include <cmath>

#include <moveit/robot_model/robot_model.h>
#include <moveit/robot_model/joint_model_group.h>
#include <geometric_shapes/shapes.h>

// Workspace bounds for sampling (can be made parameters if needed)
const float WORKSPACE_X_MIN = -0.5f;
const float WORKSPACE_X_MAX =  1.0f;
const float WORKSPACE_Y_MIN = -0.6f;
const float WORKSPACE_Y_MAX =  0.6f;
const float WORKSPACE_Z_MIN =  0.0f;
const float WORKSPACE_Z_MAX =  1.5f;


// Recursive helper function to accumulate successor link names.
void getSuccessorLinkNames(const urdf::LinkConstSharedPtr& link, std::vector<std::string>& successor_links)
{
    // Iterate over each child link of the current link.
    for(const auto& child_link: link->child_links)
    {
        // Add the child link name to the list of successor links.
        successor_links.push_back(child_link->name);
        // Recursively call the function on the child link.
        getSuccessorLinkNames(child_link, successor_links);
    }
}

// Main function to get all successor link names given a link name and a URDF model.
std::vector<std::string> getSuccessorLinkNames(const urdf::ModelInterfaceSharedPtr& model, const std::string& link_name)
{
    std::vector<std::string> successor_links;
    // Get the starting link from the model.
    urdf::LinkConstSharedPtr link = model->getLink(link_name);
    if (!link)
    {
        std::cerr << "Link " << link_name << " not found in the model." << std::endl;
        return successor_links;
    }
    // Start the recursion.
    getSuccessorLinkNames(link, successor_links);
    return successor_links;
}

/**
    Get the names of the links that are not moveable by the group.
 */
std::vector<std::string> getUnmoveableLinkNames(const moveit::core::RobotModelPtr & robot_model, const std::string & group_name)
{
    std::vector<std::string> link_names = robot_model->getLinkModelNames();
    const moveit::core::JointModelGroup* joint_model_group = robot_model->getJointModelGroup(group_name);

    // Get the moveable link names.
    std::string root_link_name_of_group = joint_model_group->getLinkModelNames().front();
    std::vector<std::string> successor_link_names = getSuccessorLinkNames(robot_model->getURDF(), root_link_name_of_group);
    // successor_link_names.push_back(root_link_name_of_group);

    // remove the moveable link names from the link names
    for (size_t i = 0; i < successor_link_names.size(); i++)
    {
        link_names.erase(std::remove(link_names.begin(), link_names.end(), successor_link_names[i]), link_names.end());
    }

    return link_names;
}

/**
    Get shape and transforms of link model.
 */
void getLinkShapesAndTransforms(const moveit::core::LinkModel* link_model, std::vector<shapes::ShapeConstPtr>& shapes, EigenSTL::vector_Affine3d& shape_poses)
{
    // Get shapes of this link.
    shapes.clear();
    for(auto s: link_model->getShapes())
    {
        shapes.push_back(s);
    }
    // Get transforms of each shape.
    shape_poses.clear();
    for (auto t: link_model->getCollisionOriginTransforms())
    {
        shape_poses.push_back(t);
    }
}

void generate_sphere_obstacles(std::vector<std::vector<float>> & balls_pos, std::vector<float> & ball_radius, const std::string & group_name, int num_of_obstacle_spheres, float obstacle_spheres_radius)
{
    if (group_name == "arm"){
        for (int i = 0; i < num_of_obstacle_spheres; i++)
        {
            float x = 0.3 * ((float)rand() / RAND_MAX) + 0.3;
            float y = 2.0 * 0.5 * ((float)rand() / RAND_MAX) - 0.5;
            float z = 1.0 * ((float)rand() / RAND_MAX) + 0.5;
            balls_pos.push_back({x, y, z});
            ball_radius.push_back(obstacle_spheres_radius);
        }
    }
    else if (group_name == "fr3_arm")
    {
        for (int i = 0; i < num_of_obstacle_spheres; i++)
        {
            float x = 1.4 * ((float)rand() / RAND_MAX) - 0.7;
            float y = 1.4 * ((float)rand() / RAND_MAX) - 0.7;
            float z = 1.0 * ((float)rand() / RAND_MAX) + 0.0;

            if (
                x > -0.2 && x < 0.2 &&
                y > -0.2 && y < 0.2 &&
                z > 0.0 && z < 0.6
            )
                continue;

            balls_pos.push_back({x, y, z});
            ball_radius.push_back(obstacle_spheres_radius);
        }
    }
    else
    {
        std::cout << "Group name is not supported!" << std::endl;
    }
}

struct BoundingBox
{
    float x_min;
    float x_max;
    float y_min;
    float y_max;
    float z_min;
    float z_max;

    // pose
    float x;
    float y;
    float z;
    float roll;
    float pitch;
    float yaw;
};

/**
    Generate a map of bounding boxes for each link in the robot model.
 */
std::map<std::string, std::vector<BoundingBox>> getBoundingBoxForLinks(const moveit::core::RobotModelPtr & robot_model, const std::vector<std::string> & link_names)
{
    // create a map from link name to a list of bounding boxes
    std::map<std::string, std::vector<BoundingBox>> link_name_to_bounding_boxes;

    for (size_t i = 0; i < link_names.size(); i++)
    {
        std::vector<shapes::ShapeConstPtr> shapes;
        EigenSTL::vector_Affine3d shape_poses;
        auto link_model = robot_model->getLinkModel(link_names[i]);
        getLinkShapesAndTransforms(link_model, shapes, shape_poses);

        std::vector<BoundingBox> bounding_boxes;
        BoundingBox bounding_box;
        bounding_box.x_min = -1 * link_model->getShapeExtentsAtOrigin()[0] / 2;
        bounding_box.x_max = link_model->getShapeExtentsAtOrigin()[0] / 2;
        bounding_box.y_min = -1 * link_model->getShapeExtentsAtOrigin()[1] / 2;
        bounding_box.y_max = link_model->getShapeExtentsAtOrigin()[1] / 2;
        bounding_box.z_min = -1 * link_model->getShapeExtentsAtOrigin()[2] / 2;
        bounding_box.z_max = link_model->getShapeExtentsAtOrigin()[2] / 2;

        bounding_box.x = link_model->getCenteredBoundingBoxOffset()[0];
        bounding_box.y = link_model->getCenteredBoundingBoxOffset()[1];
        bounding_box.z = link_model->getCenteredBoundingBoxOffset()[2];

        bounding_box.roll = 0.0;
        bounding_box.pitch = 0.0;
        bounding_box.yaw = 0.0;

        bounding_boxes.push_back(bounding_box);

        link_name_to_bounding_boxes[link_names[i]] = bounding_boxes;
    }
    return link_name_to_bounding_boxes;
}

/**
    Generate unmoveable bounding boxes for the robot model.
 */
std::vector<BoundingBox> getUnmoveableBoundingBoxes(const moveit::core::RobotModelPtr & robot_model, const std::string & group_name, float extra_safety_distance = 0.0)
{
    moveit::core::RobotStatePtr robot_state = std::make_shared<moveit::core::RobotState>(robot_model);
    // set robot state to default state
    robot_state->setToDefaultValues();

    // update the robot state
    robot_state->update();

    std::vector<std::string> unmoveable_link_names = getUnmoveableLinkNames(robot_model, group_name);
    std::map<std::string, std::vector<BoundingBox>> link_name_to_bounding_boxes = getBoundingBoxForLinks(robot_model, unmoveable_link_names);
    std::vector<BoundingBox> bounding_boxes_in_base_link; 

    // print the bounding boxes of the links
    for (const auto & link_name_to_bounding_box : link_name_to_bounding_boxes)
    {
        // get the link pose
        Eigen::Isometry3d link_pose = robot_state->getGlobalLinkTransform(link_name_to_bounding_box.first);

        for (const auto & bounding_box : link_name_to_bounding_box.second)
        {

            // calculate the bounding box in the base link
            Eigen::Isometry3d bounding_box_pose = Eigen::Isometry3d::Identity();
            bounding_box_pose.translation() = Eigen::Vector3d(bounding_box.x, bounding_box.y, bounding_box.z);
            Eigen::AngleAxisd roll_angle(bounding_box.roll, Eigen::Vector3d::UnitX());
            Eigen::AngleAxisd pitch_angle(bounding_box.pitch, Eigen::Vector3d::UnitY());
            Eigen::AngleAxisd yaw_angle(bounding_box.yaw, Eigen::Vector3d::UnitZ());

            bounding_box_pose.rotate(roll_angle * pitch_angle * yaw_angle);

            Eigen::Isometry3d bounding_box_in_base_link = link_pose * bounding_box_pose;
            BoundingBox bounding_box_in_base_link_struct;
            bounding_box_in_base_link_struct.x_min = bounding_box.x_min - extra_safety_distance;
            bounding_box_in_base_link_struct.x_max = bounding_box.x_max + extra_safety_distance;
            bounding_box_in_base_link_struct.y_min = bounding_box.y_min - extra_safety_distance;
            bounding_box_in_base_link_struct.y_max = bounding_box.y_max + extra_safety_distance;
            bounding_box_in_base_link_struct.z_min = bounding_box.z_min - extra_safety_distance;
            bounding_box_in_base_link_struct.z_max = bounding_box.z_max + extra_safety_distance;

            bounding_box_in_base_link_struct.x = bounding_box_in_base_link.translation().x();
            bounding_box_in_base_link_struct.y = bounding_box_in_base_link.translation().y();
            bounding_box_in_base_link_struct.z = bounding_box_in_base_link.translation().z();

            Eigen::Vector3d rpy = bounding_box_in_base_link.rotation().eulerAngles(0, 1, 2);
            bounding_box_in_base_link_struct.roll = rpy[0];
            bounding_box_in_base_link_struct.pitch = rpy[1];
            bounding_box_in_base_link_struct.yaw = rpy[2];

            bounding_boxes_in_base_link.push_back(bounding_box_in_base_link_struct);
        }
    }

    return bounding_boxes_in_base_link;
}

/**
    Check if a point is inside a bounding box of a vector of bounding boxes.
    For each bounding box, we first need to transform the point to the bounding box frame, then check if the point is inside the bounding box.
 */
bool isPointInsideBoundingBoxes(const Eigen::Vector3d & point, const std::vector<BoundingBox> & bounding_boxes)
{
    for (const auto& box : bounding_boxes)
    {
        // Translation of the bounding box
        Eigen::Vector3d box_translation(box.x, box.y, box.z);

        // Construct rotation matrix from roll, pitch, yaw (ZYX order)
        Eigen::AngleAxisd roll_angle(box.roll, Eigen::Vector3d::UnitX());
        Eigen::AngleAxisd pitch_angle(box.pitch, Eigen::Vector3d::UnitY());
        Eigen::AngleAxisd yaw_angle(box.yaw, Eigen::Vector3d::UnitZ());
        // Eigen::Matrix3d box_rotation = yaw_angle * pitch_angle * roll_angle;
        Eigen::Matrix3d box_rotation = (yaw_angle * pitch_angle * roll_angle).toRotationMatrix();

        // Transform the point into the bounding box's local frame
        Eigen::Vector3d local_point = box_rotation.transpose() * (point - box_translation);

        // Check if the local point lies within the AABB bounds in local frame
        if (local_point.x() >= box.x_min && local_point.x() <= box.x_max &&
            local_point.y() >= box.y_min && local_point.y() <= box.y_max &&
            local_point.z() >= box.z_min && local_point.z() <= box.z_max)
        {
            return true;
        }
    }
    return false;
}

bool isSphereCollidingWithBoundingBoxes(const Eigen::Vector3d& center,
                                        float radius,
                                        const std::vector<BoundingBox>& bounding_boxes)
{
    for (const auto& box : bounding_boxes)
    {
        // Transform center to local frame of bounding box
        Eigen::Vector3d box_translation(box.x, box.y, box.z);
        Eigen::AngleAxisd roll(box.roll, Eigen::Vector3d::UnitX());
        Eigen::AngleAxisd pitch(box.pitch, Eigen::Vector3d::UnitY());
        Eigen::AngleAxisd yaw(box.yaw, Eigen::Vector3d::UnitZ());
        Eigen::Matrix3d box_rotation = (yaw * pitch * roll).toRotationMatrix();
        Eigen::Vector3d local_center = box_rotation.transpose() * (center - box_translation);

        // Compute closest point on box to sphere center (clamp)
        Eigen::Vector3d closest_point;
        closest_point.x() = std::min(std::max(local_center.x(), static_cast<double>(box.x_min)), static_cast<double>(box.x_max));
        closest_point.y() = std::min(std::max(local_center.y(), static_cast<double>(box.y_min)), static_cast<double>(box.y_max));
        closest_point.z() = std::min(std::max(local_center.z(), static_cast<double>(box.z_min)), static_cast<double>(box.z_max));

        // If the distance from center to closest point is less than radius → collision
        double dist_sq = (closest_point - local_center).squaredNorm();
        if (dist_sq <= static_cast<double>(radius * radius))
        {
            return true;
        }
    }
    return false;
}

/**
    Sample a number of spheres which is not in the bounding boxes of the robot model.
    */
void genSphereObstacles(
    int num_of_obstacles, 
    float max_radius, 
    float min_radius, 
    const std::vector<BoundingBox> & bounding_boxes, 
    std::vector<std::vector<float>> & obstacle_positions,
    std::vector<float> & obstacle_radius
)
{
    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_real_distribution<float> dist_x(WORKSPACE_X_MIN, WORKSPACE_X_MAX);
    std::uniform_real_distribution<float> dist_y(WORKSPACE_Y_MIN, WORKSPACE_Y_MAX);
    std::uniform_real_distribution<float> dist_z(WORKSPACE_Z_MIN, WORKSPACE_Z_MAX);

    obstacle_positions.clear();
    obstacle_radius.clear();
    int max_attempts = 1000;
    int attempts = 0;

    while (obstacle_positions.size() < static_cast<size_t>(num_of_obstacles) && attempts < max_attempts)
    {
        Eigen::Vector3d position_candidate(dist_x(gen), dist_y(gen), dist_z(gen));
        // Random radius
        float radius_candidate = min_radius + (max_radius - min_radius) * ((float)rand() / RAND_MAX);

        // Check if it's inside any bounding box
        // if (!isPointInsideBoundingBoxes(candidate, bounding_boxes))
        if (!isSphereCollidingWithBoundingBoxes(position_candidate, radius_candidate, bounding_boxes))
        {
            std::vector<double> obstacle_position_double = {position_candidate.x(), position_candidate.y(), position_candidate.z()}; 
            // convert c from double to float
            std::vector<float> obstacle_position(obstacle_position_double.begin(), obstacle_position_double.end());
            obstacle_positions.push_back(obstacle_position);
            obstacle_radius.push_back(radius_candidate);
        }
        attempts++;
    }

    if (obstacle_positions.size() < static_cast<size_t>(num_of_obstacles))
    {
        std::cerr << "[genSphereObstacles] Warning: Only generated " << obstacle_positions.size()
                  << " obstacles after " << attempts << " attempts." << std::endl;
    }
}