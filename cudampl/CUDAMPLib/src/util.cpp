#include "util.h"

/**
* Flatten a 2D vector of floats into a 1D vector.
*/
std::vector<float> CUDAMPLib::floatVectorFlatten(const std::vector<std::vector<float>>& input)
{
    // 1. Calculate the total number of elements
    size_t totalSize = 0;
    for (const auto& sub : input) {
        totalSize += sub.size();
    }

    // 2. Create the output vector, reserving the exact needed capacity
    std::vector<float> output;
    output.reserve(totalSize);

    // 3. Copy each element from the sub-vectors into 'output'
    for (const auto& sub : input) {
        for (float value : sub) {
            output.push_back(value);
        }
    }

    return output;
}

std::vector<float> CUDAMPLib::IsometryVectorFlatten(const std::vector<Eigen::Isometry3d>& transforms)
{
    // Each Isometry3d is effectively a 4x4 matrix (16 elements).
    // Reserve enough space for all transforms upfront.
    std::vector<float> output;
    output.reserve(transforms.size() * 4 * 4);

    // Copy each transform's 4x4 matrix into 'output', converting double->float.
    for (const auto& iso : transforms)
    {
        // Extract as a 4x4 double matrix
        Eigen::Matrix4d mat = iso.matrix();

        // Push each element (row-major) as float
        for (int row = 0; row < 4; ++row) {
            for (int col = 0; col < 4; ++col) {
                output.push_back(static_cast<float>(mat(row, col)));
            }
        }
    }

    return output;
}

std::vector<std::vector<float>> CUDAMPLib::FromFloatVectorToVec3(const std::vector<float>& data, size_t size)
{
    // Check if the data size is a multiple of the given size
    if (data.size() % size != 0)
    {
        std::cerr << "Invalid data size for conversion." << std::endl;
        return {};
    }

    // Create the output vector
    std::vector<std::vector<float>> output;
    output.reserve(data.size() / size);

    // Iterate over the data, extracting vectors of the given size
    for (size_t i = 0; i < data.size(); i += size)
    {
        output.push_back(std::vector<float>(data.begin() + i, data.begin() + i + size));
    }

    return output;
}

std::vector<Eigen::Isometry3d> CUDAMPLib::fromFloatVector(const std::vector<float>& data)
{
    // Check if the data size is a multiple of 16
    if (data.size() % 16 != 0)
    {
        std::cerr << "Invalid data size for Isometry3d conversion." << std::endl;
        return {};
    }

    // Create the output vector
    std::vector<Eigen::Isometry3d> output;
    output.reserve(data.size() / 16);

    // Iterate over the data, extracting 4x4 matrices
    for (size_t i = 0; i < data.size(); i += 16)
    {
        // Create a 4x4 matrix from the data
        Eigen::Matrix4d mat;
        for (int row = 0; row < 4; ++row)
        {
            for (int col = 0; col < 4; ++col)
            {
                mat(row, col) = static_cast<double>(data[i + row * 4 + col]);
            }
        }

        // Convert the matrix to an Isometry3d and add it to the output
        output.push_back(Eigen::Isometry3d(mat));
    }

    return output;
}

std::vector<float> CUDAMPLib::Vector3dflatten(const std::vector<Eigen::Vector3d>& vectors)
{
    // Each Vector3d has 3 elements (double precision).
    // Reserve space to avoid multiple allocations.
    std::vector<float> output;
    output.reserve(vectors.size() * 3);

    // Copy each coordinate (convert from double to float).
    for (const auto& v : vectors)
    {
        output.push_back(static_cast<float>(v.x()));
        output.push_back(static_cast<float>(v.y()));
        output.push_back(static_cast<float>(v.z()));
    }

    return output;
}

std::vector<int> CUDAMPLib::boolVectorFlatten(const std::vector<std::vector<bool>>& input)
{
    // 1. Calculate the total number of elements
    size_t totalSize = 0;
    for (const auto& sub : input) {
        totalSize += sub.size();
    }

    // 2. Create the output vector, reserving the exact needed capacity
    std::vector<int> output;
    output.reserve(totalSize);

    // 3. Copy each element from the sub-vectors into 'output'
    for (const auto& sub : input) {
        for (bool value : sub) {
            // Convert bool to int
            output.push_back(value ? 1 : 0);
        }
    }

    return output;
}