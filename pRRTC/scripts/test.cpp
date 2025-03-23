#include <fstream>
#include <iostream>

#include "src/collision/environment.hh"
#include "src/collision/factory.hh"
#include "src/planning/Planners.hh"
#include "src/planning/pRRTC_settings.hh"

#include <yaml-cpp/yaml.h>
#include <sstream>
#include <iomanip>

using namespace ppln::collision;

// Function to load the environment from a YAML file and create the Environment object.
Environment<float> load_environment(const YAML::Node &scene)
{
    // Vectors to temporarily hold the shapes.
    std::vector<Sphere<float>> spheres;
    std::vector<Capsule<float>> capsules;
    std::vector<Cylinder<float>> cylinders;
    std::vector<Cuboid<float>> cuboids;

    YAML::Node objects = scene["world"]["collision_objects"];

    // Loop through each collision object.
    for (std::size_t i = 0; i < objects.size(); ++i) {
        YAML::Node obj = objects[i];

        // Get the header if needed (e.g., frame_id).
        YAML::Node header = obj["header"];
        std::string frame_id = header ? header["frame_id"].as<std::string>() : "";

        // Get the first primitive pose (assumes at least one is provided).
        YAML::Node poses = obj["primitive_poses"];
        Eigen::Vector3f position(0.f, 0.f, 0.f);
        Eigen::Quaternionf orientation(1.f, 0.f, 0.f, 0.f); // identity quaternion

        if (poses && poses.size() > 0) {
            YAML::Node pose = poses[0];

            // Extract position if available.
            if (pose["position"]) {
                std::vector<float> pos_vec = pose["position"].as<std::vector<float>>();
                if (pos_vec.size() >= 3) {
                    position = Eigen::Vector3f(pos_vec[0], pos_vec[1], pos_vec[2]);
                }
            }

            // Extract orientation if available.
            if (pose["orientation"]) {
                std::vector<float> orient_vec = pose["orientation"].as<std::vector<float>>();
                if (orient_vec.size() >= 4) {
                    // YAML gives [x, y, z, w] but Eigen expects (w, x, y, z)
                    orientation = Eigen::Quaternionf(orient_vec[3], orient_vec[0], orient_vec[1], orient_vec[2]);
                }
            }
        }

        // Loop over the primitives for this object.
        YAML::Node primitives = obj["primitives"];
        for (std::size_t j = 0; j < primitives.size(); ++j) {
            YAML::Node prim = primitives[j];
            std::string type = prim["type"].as<std::string>();

            if (type == "cylinder") {
                // Expect dimensions: [length, radius]
                std::vector<float> dims = prim["dimensions"].as<std::vector<float>>();
                if (dims.size() >= 2) {
                    float length = dims[0];
                    float radius = dims[1];
                    // Use the cylinder factory using center-based construction.
                    cylinders.push_back(
                        factory::cylinder::center::eigen_rot(position, orientation, radius, length)
                    );
                }
            } else if (type == "box") {
                // Expect dimensions: [length, width, height] (full extents).
                std::vector<float> dims = prim["dimensions"].as<std::vector<float>>();
                if (dims.size() >= 3) {
                    // Convert full extents to half-extents.
                    Eigen::Vector3f half_extents(dims[0] / 2.0f, dims[1] / 2.0f, dims[2] / 2.0f);
                    cuboids.push_back(
                        factory::cuboid::eigen_rot(position, orientation, half_extents)
                    );
                }
            } else if (type == "sphere") {
                // If a sphere primitive is defined (dimensions: [radius]).
                std::vector<float> dims = prim["dimensions"].as<std::vector<float>>();
                if (!dims.empty()) {
                    float radius = dims[0];
                    spheres.push_back(
                        factory::sphere::eigen(position, radius)
                    );
                }
            }
            // You can add further handling for "capsule" or other types here.
        }
    }

    // print the number of objects
    std::cout << "Loaded environment with:\n"
              << "Spheres: " << spheres.size() << "\n"
              << "Cylinders: " << cylinders.size() << "\n"
              << "Cuboids: " << cuboids.size() << "\n"
              << std::endl;

    // // Allocate the Environment object and copy the shapes into dynamically allocated arrays.
    Environment<float> env{};

    if (!spheres.empty()) {
        env.spheres = new Sphere<float>[spheres.size()];
        std::copy(spheres.begin(), spheres.end(), env.spheres);
        env.num_spheres = static_cast<unsigned int>(spheres.size());
    }

    if (!capsules.empty()) {
        env.capsules = new Capsule<float>[capsules.size()];
        std::copy(capsules.begin(), capsules.end(), env.capsules);
        env.num_capsules = static_cast<unsigned int>(capsules.size());
    }

    if (!cylinders.empty()) {
        env.cylinders = new Cylinder<float>[cylinders.size()];
        std::copy(cylinders.begin(), cylinders.end(), env.cylinders);
        env.num_cylinders = static_cast<unsigned int>(cylinders.size());
    }

    if (!cuboids.empty()) {
        env.cuboids = new Cuboid<float>[cuboids.size()];
        std::copy(cuboids.begin(), cuboids.end(), env.cuboids);
        env.num_cuboids = static_cast<unsigned int>(cuboids.size());
    }

    return env;
}

void loadStartAndGoal(const YAML::Node &config,
                      std::map<std::string, double>& start,
                      std::map<std::string, double>& goal)
{
    // ====== Extract start state ======
    // Expecting structure: start_state -> joint_state -> { name: [...], position: [...] }
    YAML::Node jointState = config["start_state"]["joint_state"];
    if (!jointState) {
        throw std::runtime_error("start_state->joint_state not found in the YAML file.");
    }
    
    // Get names and positions as vectors.
    std::vector<std::string> startNames = jointState["name"].as<std::vector<std::string>>();
    std::vector<double> startPositions = jointState["position"].as<std::vector<double>>();
    
    // Check that the sizes match.
    if (startNames.size() != startPositions.size()) {
        throw std::runtime_error("Mismatch between number of joint names and positions in start_state.");
    }
    
    // Fill the start map.
    for (std::size_t i = 0; i < startNames.size(); ++i) {
        start[startNames[i]] = startPositions[i];
    }

    // ====== Extract goal constraints ======
    // Expecting structure: goal_constraints: [ { joint_constraints: [ { joint_name, position }, ... ] }, ... ]
    YAML::Node goalConstraints = config["goal_constraints"];
    if (!goalConstraints || !goalConstraints.IsSequence()) {
        throw std::runtime_error("goal_constraints not found or not a sequence.");
    }
    
    // Loop through each goal constraint entry.
    for (std::size_t i = 0; i < goalConstraints.size(); ++i) {
        YAML::Node constraint = goalConstraints[i];
        if (!constraint["joint_constraints"] || !constraint["joint_constraints"].IsSequence()) {
            continue;  // skip if there is no joint_constraints list
        }
        YAML::Node jointConstraints = constraint["joint_constraints"];
        // For each joint constraint, get the joint name and position.
        for (std::size_t j = 0; j < jointConstraints.size(); ++j) {
            YAML::Node jointConstraint = jointConstraints[j];
            std::string jointName = jointConstraint["joint_name"].as<std::string>();
            double position = jointConstraint["position"].as<double>();
            goal[jointName] = position;
        }
    }
}

int main(int argc, char* argv[]) {

    std::string problem_dir = "/home/ros/problems";
    std::string problem_name = "bookshelf_small";
    std::string robot_name = "fetch";
    int problem_idx = 1;
    std::ostringstream oss;
    // Set the width to 5 and fill with '0'
    oss << std::setw(4) << std::setfill('0') << problem_idx;
    std::string task_index_str = oss.str();
    std::string scene_name = "scene" + task_index_str + ".yaml";

    std::string scene_path = problem_dir + "/" + problem_name + "_" + robot_name + "/" + scene_name;

    std::cout << "scene path : " << scene_path << std::endl;

    // load the file with yaml
    YAML::Node scene = YAML::LoadFile(scene_path);

    // Environment env = load_environment(scene);
    Environment<float> env{};

    // load the request
    std::string request_name = "request" + task_index_str + ".yaml";
    std::string request_path = problem_dir + "/" + problem_name + "_" + robot_name + "/" + request_name;

    std::cout << "request path : " << request_path << std::endl;

    // load the file with yaml
    YAML::Node request = YAML::LoadFile(request_path);

    std::map<std::string, double> start;
    std::map<std::string, double> goal;
    loadStartAndGoal(request, start, goal);

    // convert goal to vector
    std::vector<float> goal_vector;
    for (const auto& [key, value] : goal) {
        goal_vector.push_back((float)value);
    }

    // convert start to vector but only the key where goal has
    std::vector<float> start_vector;
    for (const auto& [key, value] : goal) {
        if (start.find(key) != start.end()) {
            start_vector.push_back((float)(start[key]));
        }
    }

    // print start_vector and goal_vector
    std::cout << "start: ";
    for (const auto& val : start_vector) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
    std::cout << "goal: ";
    for (const auto& val : goal_vector) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    // create the robot configuration based on start_vector
    ppln::robots::Fetch::Configuration startConfig;
    for (std::size_t i = 0; i < start_vector.size(); ++i) {
        startConfig[i] = start_vector[i];
    }

    // create the robot configuration based on goal_vector
    ppln::robots::Fetch::Configuration goalConfig;
    for (std::size_t i = 0; i < goal_vector.size(); ++i) {
        goalConfig[i] = goal_vector[i];
    }
    std::vector<ppln::robots::Fetch::Configuration> goalConfigs;
    goalConfigs.push_back(goalConfig);

    // Define the settings for the pRRTC planner
    struct pRRTC_settings settings;
    settings.num_new_configs = 512;
    settings.max_iters = 10;
    settings.granularity = 64;
    settings.range = 0.5;
    settings.balance = 2;
    settings.tree_ratio = 1.0;
    settings.dynamic_domain = true;
    settings.dd_radius = 6.0;
    settings.dd_min_radius = 1.0;
    settings.dd_alpha = 0.0001;

    // print startConfig and goalConfig
    ppln::robots::Fetch::print_robot_config(startConfig);
    ppln::robots::Fetch::print_robot_config(goalConfig);

    // run the planner
    auto result = pRRTC::solve<ppln::robots::Fetch>(startConfig, goalConfigs, env, settings);

    for (auto& cfg: result.path) {
        print_cfg<ppln::robots::Fetch>(cfg);
    }

    if (not result.solved) {
        std::cout << "failed!" << std::endl;
    }

    std::cout << "cost: " << result.cost << "\n";
    std::cout << "time (us): " << result.kernel_ns/1000.0f << "\n";

    return 0;
}