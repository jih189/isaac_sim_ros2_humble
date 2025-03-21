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

#include <visualization_msgs/msg/marker_array.hpp>
#include <rclcpp/rclcpp.hpp>

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

struct Sphere
{
    float x;
    float y;
    float z;
    float radius;
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
    Sample a number of spheres which do not collide with the bounding boxes of the robot model.
    */
void genSphereObstacles(
    int num_of_obstacles, 
    float max_radius, 
    float min_radius, 
    const std::vector<BoundingBox> & bounding_boxes, 
    std::vector<Sphere> & obstacle_spheres
)
{
    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_real_distribution<float> dist_x(WORKSPACE_X_MIN, WORKSPACE_X_MAX);
    std::uniform_real_distribution<float> dist_y(WORKSPACE_Y_MIN, WORKSPACE_Y_MAX);
    std::uniform_real_distribution<float> dist_z(WORKSPACE_Z_MIN, WORKSPACE_Z_MAX);

    obstacle_spheres.clear();

    int max_attempts = 1000;
    int attempts = 0;

    while (obstacle_spheres.size() < static_cast<size_t>(num_of_obstacles) && attempts < max_attempts)
    {
        Eigen::Vector3d position_candidate(dist_x(gen), dist_y(gen), dist_z(gen));
        // Random radius
        float radius_candidate = min_radius + (max_radius - min_radius) * ((float)rand() / RAND_MAX);

        // Check if it collides with any bounding box
        if (!isSphereCollidingWithBoundingBoxes(position_candidate, radius_candidate, bounding_boxes))
        {
            std::vector<double> obstacle_position_double = {position_candidate.x(), position_candidate.y(), position_candidate.z()}; 
            // convert c from double to float
            std::vector<float> obstacle_position(obstacle_position_double.begin(), obstacle_position_double.end());

            Sphere obstacle_sphere;
            obstacle_sphere.x = (float)(position_candidate.x());
            obstacle_sphere.y = (float)(position_candidate.y());
            obstacle_sphere.z = (float)(position_candidate.z());
            obstacle_sphere.radius = radius_candidate;

            obstacle_spheres.push_back(obstacle_sphere);
        }
        attempts++;
    }

    if (static_cast<int>(obstacle_spheres.size()) < num_of_obstacles)
    {
        // print in red
        std::cout << "\033[1;31mFailed to generate " << num_of_obstacles << " spheres after " << max_attempts << " attempts.\033[0m" << std::endl;
    }
}

visualization_msgs::msg::MarkerArray generateSpheresMarkers(
    const std::vector<Sphere> & spheres,
    rclcpp::Node::SharedPtr node)
{
     // Create a obstacle MarkerArray message
    visualization_msgs::msg::MarkerArray obstacle_collision_spheres_marker_array;
    for (size_t i = 0; i < spheres.size(); i++)
    {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = "base_link";
        marker.header.stamp = node->now();
        marker.ns = "obstacle_collision_spheres";
        marker.id = i;
        marker.type = visualization_msgs::msg::Marker::SPHERE;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.pose.position.x = spheres[i].x;
        marker.pose.position.y = spheres[i].y;
        marker.pose.position.z = spheres[i].z;

        marker.pose.orientation.x = 0.0;
        marker.pose.orientation.y = 0.0;
        marker.pose.orientation.z = 0.0;
        marker.pose.orientation.w = 1.0;
        marker.scale.x = 2 * spheres[i].radius;
        marker.scale.y = 2 * spheres[i].radius;
        marker.scale.z = 2 * spheres[i].radius;
        marker.color.a = 1.0; // Don't forget to set the alpha!
        marker.color.r = 0.5;
        marker.color.g = 0.5;
        marker.color.b = 0.0;
        obstacle_collision_spheres_marker_array.markers.push_back(marker);
    }
    return obstacle_collision_spheres_marker_array;
}

bool isBoundingBoxesCollidingWithBoundingBoxes(
    const BoundingBox& box1,
    const std::vector<BoundingBox>& other_boxes)
{
    for (const auto& box2 : other_boxes)
    {
        // Build rotation matrices
        Eigen::AngleAxisd roll1(box1.roll, Eigen::Vector3d::UnitX());
        Eigen::AngleAxisd pitch1(box1.pitch, Eigen::Vector3d::UnitY());
        Eigen::AngleAxisd yaw1(box1.yaw, Eigen::Vector3d::UnitZ());
        Eigen::Matrix3d R1 = (yaw1 * pitch1 * roll1).toRotationMatrix();

        Eigen::AngleAxisd roll2(box2.roll, Eigen::Vector3d::UnitX());
        Eigen::AngleAxisd pitch2(box2.pitch, Eigen::Vector3d::UnitY());
        Eigen::AngleAxisd yaw2(box2.yaw, Eigen::Vector3d::UnitZ());
        Eigen::Matrix3d R2 = (yaw2 * pitch2 * roll2).toRotationMatrix();

        // Half-dimensions of box1 and box2 in local frame
        Eigen::Vector3d half1(
            0.5 * (box1.x_max - box1.x_min),
            0.5 * (box1.y_max - box1.y_min),
            0.5 * (box1.z_max - box1.z_min));

        Eigen::Vector3d half2(
            0.5 * (box2.x_max - box2.x_min),
            0.5 * (box2.y_max - box2.y_min),
            0.5 * (box2.z_max - box2.z_min));

        // Translation vector between box centers in box1's frame
        Eigen::Vector3d t = R1.transpose() * (Eigen::Vector3d(box2.x, box2.y, box2.z) - Eigen::Vector3d(box1.x, box1.y, box1.z));

        // Rotation matrix from box2 to box1 frame
        Eigen::Matrix3d R = R1.transpose() * R2;

        // Absolute value with epsilon to handle near-zero cases
        Eigen::Matrix3d AbsR;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                AbsR(i, j) = std::abs(R(i, j)) + 1e-6;

        // SAT: Test all 15 separating axes
        for (int i = 0; i < 3; ++i) {
            double ra = half1[i];
            double rb = half2[0] * AbsR(i, 0) + half2[1] * AbsR(i, 1) + half2[2] * AbsR(i, 2);
            if (std::abs(t[i]) > ra + rb)
                goto no_collision;
        }

        for (int i = 0; i < 3; ++i) {
            double ra = half1[0] * AbsR(0, i) + half1[1] * AbsR(1, i) + half1[2] * AbsR(2, i);
            double rb = half2[i];
            if (std::abs(t[0] * R(0, i) + t[1] * R(1, i) + t[2] * R(2, i)) > ra + rb)
                goto no_collision;
        }

        // Cross products of axes
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                double ra = half1[(i+1)%3] * AbsR((i+2)%3, j) + half1[(i+2)%3] * AbsR((i+1)%3, j);
                double rb = half2[(j+1)%3] * AbsR(i, (j+2)%3) + half2[(j+2)%3] * AbsR(i, (j+1)%3);
                double dist = std::abs(t((i+2)%3) * R((i+1)%3, j) - t((i+1)%3) * R((i+2)%3, j));
                if (dist > ra + rb)
                    goto no_collision;
            }
        }

        // No separating axis found → collision
        return true;

    no_collision:
        continue;
    }
    return false;
}

/**
    Sample a number of cuboid, represented as bounding box, which do not collide with the bounding boxes of the robot model.
    */
void genCuboidObstacles(
    int num_of_obstacles,
    float max_side_length,
    float min_side_length,
    const std::vector<BoundingBox>& bounding_boxes,
    std::vector<BoundingBox>& obstacle_bounding_boxes)
{
    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_real_distribution<float> dist_x(WORKSPACE_X_MIN, WORKSPACE_X_MAX);
    std::uniform_real_distribution<float> dist_y(WORKSPACE_Y_MIN, WORKSPACE_Y_MAX);
    std::uniform_real_distribution<float> dist_z(WORKSPACE_Z_MIN, WORKSPACE_Z_MAX);
    std::uniform_real_distribution<float> dist_length(min_side_length, max_side_length);
    std::uniform_real_distribution<float> dist_angle(-M_PI, M_PI);

    obstacle_bounding_boxes.clear();

    int max_attempts = 1000;
    int attempts = 0;

    while (static_cast<int>(obstacle_bounding_boxes.size()) < num_of_obstacles && attempts < max_attempts)
    {
        float dx = dist_length(gen);
        float dy = dist_length(gen);
        float dz = dist_length(gen);

        BoundingBox candidate;
        candidate.x_min = -dx / 2.0f;
        candidate.x_max = dx / 2.0f;
        candidate.y_min = -dy / 2.0f;
        candidate.y_max = dy / 2.0f;
        candidate.z_min = -dz / 2.0f;
        candidate.z_max = dz / 2.0f;

        candidate.x = dist_x(gen);
        candidate.y = dist_y(gen);
        candidate.z = dist_z(gen);
        candidate.roll  = dist_angle(gen);
        candidate.pitch = dist_angle(gen);
        candidate.yaw   = dist_angle(gen);

        std::vector<BoundingBox> existing_boxes = bounding_boxes;
        existing_boxes.insert(existing_boxes.end(), obstacle_bounding_boxes.begin(), obstacle_bounding_boxes.end());

        if (!isBoundingBoxesCollidingWithBoundingBoxes(candidate, existing_boxes))
        {
            obstacle_bounding_boxes.push_back(candidate);
        }

        attempts++;
    }

    if (static_cast<int>(obstacle_bounding_boxes.size()) < num_of_obstacles)
    {
        std::cout << "\033[1;31mGenerated only " << obstacle_bounding_boxes.size()
                  << " out of " << num_of_obstacles
                  << " cuboids after " << max_attempts << " attempts.\033[0m" << std::endl;
    }
}

visualization_msgs::msg::MarkerArray generateBoundingBoxesMarkers(
    const std::vector<BoundingBox>& boxes,
    rclcpp::Node::SharedPtr node)
{
    visualization_msgs::msg::MarkerArray marker_array;

    for (size_t i = 0; i < boxes.size(); ++i)
    {
        const auto& box = boxes[i];

        // Compute dimensions
        float dx = box.x_max - box.x_min;
        float dy = box.y_max - box.y_min;
        float dz = box.z_max - box.z_min;

        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = "base_link";
        marker.header.stamp = node->now();
        marker.ns = "obstacle_collision_boxes";
        marker.id = static_cast<int>(i);
        marker.type = visualization_msgs::msg::Marker::CUBE;
        marker.action = visualization_msgs::msg::Marker::ADD;

        // Center position
        marker.pose.position.x = box.x;
        marker.pose.position.y = box.y;
        marker.pose.position.z = box.z;

        // Orientation from RPY
        tf2::Quaternion q;
        q.setRPY(box.roll, box.pitch, box.yaw);
        marker.pose.orientation.x = q.x();
        marker.pose.orientation.y = q.y();
        marker.pose.orientation.z = q.z();
        marker.pose.orientation.w = q.w();

        // Size
        marker.scale.x = dx;
        marker.scale.y = dy;
        marker.scale.z = dz;

        // Color
        marker.color.r = 0.5;
        marker.color.g = 0.5;
        marker.color.b = 0.0;
        marker.color.a = 1.0;

        marker_array.markers.push_back(marker);
    }

    return marker_array;
}