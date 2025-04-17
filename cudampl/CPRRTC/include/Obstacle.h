#pragma once
namespace CPRRTC
{
    struct Cuboid
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

    struct Cylinder
    {
        float x;
        float y;
        float z;

        float roll;
        float pitch;
        float yaw;

        float radius;
        float height;
    };
}