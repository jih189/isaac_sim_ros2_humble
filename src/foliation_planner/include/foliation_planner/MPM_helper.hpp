#pragma once

#include <iostream>
#include <yaml-cpp/yaml.h>
#include <vector>
#include <string>

// This function loads joint values for the start state and goal constraints,
// storing the results in the provided output vectors (in the order of joint_names).
void loadJointValues(const YAML::Node &root, 
                     const std::vector<std::string>& joint_names, 
                     std::vector<double>& startValues, 
                     std::vector<double>& goalValues)
{
    // Resize the output vectors to match the number of joints and initialize with default value (0.0).
    startValues.resize(joint_names.size(), 0.0);
    goalValues.resize(joint_names.size(), 0.0);

    // --- Load Start State Values ---
    YAML::Node startNames = root["start_state"]["joint_state"]["name"];
    YAML::Node startPositions = root["start_state"]["joint_state"]["position"];
    if (startNames && startPositions && startNames.IsSequence() && startPositions.IsSequence() &&
        startNames.size() == startPositions.size())
    {
        // For each joint in the provided joint_names vector,
        // search for a match in the YAML start state.
        for (size_t i = 0; i < joint_names.size(); ++i)
        {
            const std::string &joint = joint_names[i];
            for (size_t j = 0; j < startNames.size(); ++j)
            {
                if (startNames[j].as<std::string>() == joint)
                {
                    startValues[i] = startPositions[j].as<double>();
                    break; // Joint found, move on to the next joint.
                }
            }
        }
    }

    // --- Load Goal Constraints Values ---
    YAML::Node goalConstraints = root["goal_constraints"];
    if (goalConstraints && goalConstraints.IsSequence() && !goalConstraints.IsNull())
    {
        YAML::Node jointConstraints = goalConstraints[0]["joint_constraints"];
        if (jointConstraints && jointConstraints.IsSequence())
        {
            // For each joint in the provided joint_names vector,
            // search for a matching joint in the YAML goal constraints.
            for (size_t i = 0; i < joint_names.size(); ++i)
            {
                const std::string &joint = joint_names[i];
                for (size_t j = 0; j < jointConstraints.size(); ++j)
                {
                    if (jointConstraints[j]["joint_name"].as<std::string>() == joint)
                    {
                        goalValues[i] = jointConstraints[j]["position"].as<double>();
                        break; // Joint found.
                    }
                }
            }
        }
    }
}


// Function to load the joint state from a YAML file into a map
std::map<std::string, double> loadStartStateJointState(const YAML::Node &config) {
    
    // Check if the expected nodes exist
    if (!config["start_state"] || !config["start_state"]["joint_state"]) {
        throw std::runtime_error("YAML does not contain 'start_state' or 'joint_state' node.");
    }
    
    YAML::Node jointState = config["start_state"]["joint_state"];
    YAML::Node names = jointState["name"];
    YAML::Node positions = jointState["position"];

    // Validate the presence and size of the name and position lists
    if (!names || !positions || names.size() != positions.size()) {
        throw std::runtime_error("Invalid YAML format: missing or mismatched 'name' and 'position' lists.");
    }
    
    // Create the map by pairing each joint name with its corresponding position
    std::map<std::string, double> jointMap;
    for (std::size_t i = 0; i < names.size(); ++i) {
        std::string jointName = names[i].as<std::string>();
        double pos = positions[i].as<double>();
        jointMap[jointName] = pos;
    }
    
    return jointMap;
}