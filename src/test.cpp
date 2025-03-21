struct Sphere
{
    float x;
    float y;
    float z;
    float radius;
};

// Add spheres as obstacles to the planning scene
for (size_t i = 0; i < collision_spheres.size(); i++)
{
    Eigen::Isometry3d sphere_pose = Eigen::Isometry3d::Identity();
    sphere_pose.translation() = Eigen::Vector3d(collision_spheres[i].x, collision_spheres[i].y, collision_spheres[i].z);
    planning_scene->getWorldNonConst()->addToObject("obstacle_" + std::to_string(i), shapes::ShapeConstPtr(new shapes::Sphere(collision_spheres[i].radius)), sphere_pose);
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

// x, y, z are not necessary at the center of the bounding box