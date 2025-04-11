
#ifndef FLT_MAX
#define FLT_MAX __int_as_float(0x7f7fffff)    // 3.40282347e+38f
#endif

extern "C" {
    __device__ int startTreeCounter = 0;
    __device__ int goalTreeCounter = 0;
    __device__ int sampledCounter = 0;
}
__constant__ float joint_poses[480] = 
{
1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 
1.000000, 0.000000, 0.000000, -0.124650, 0.000000, -0.000004, -1.000000, 0.238920, 0.000000, 1.000000, -0.000004, 0.311270, 0.000000, 0.000000, 0.000000, 1.000000, 
1.000000, -0.000000, 0.000000, -0.213830, 0.000000, 1.000000, 0.000000, 0.187380, -0.000000, -0.000000, 1.000000, 0.055325, 0.000000, 0.000000, 0.000000, 1.000000, 
1.000000, -0.000000, 0.000000, 0.213830, 0.000000, 1.000000, 0.000000, 0.187380, -0.000000, -0.000000, 1.000000, 0.055325, 0.000000, 0.000000, 0.000000, 1.000000, 
1.000000, -0.000000, 0.000000, 0.001291, 0.000000, 1.000000, 0.000000, 0.187380, -0.000000, -0.000000, 1.000000, 0.055325, 0.000000, 0.000000, 0.000000, 1.000000, 
1.000000, 0.000000, -0.000000, 0.235000, 0.000000, -1.000000, 0.000000, 0.000000, 0.000000, -0.000000, -1.000000, 0.287800, 0.000000, 0.000000, 0.000000, 1.000000, 
1.000000, -0.000000, 0.000000, -0.213830, 0.000000, 1.000000, 0.000000, -0.187380, -0.000000, -0.000000, 1.000000, 0.055325, 0.000000, 0.000000, 0.000000, 1.000000, 
1.000000, -0.000000, 0.000000, 0.213830, 0.000000, 1.000000, 0.000000, -0.187380, -0.000000, -0.000000, 1.000000, 0.055325, 0.000000, 0.000000, 0.000000, 1.000000, 
1.000000, -0.000000, 0.000000, 0.001291, 0.000000, 1.000000, 0.000000, -0.187380, -0.000000, -0.000000, 1.000000, 0.055325, 0.000000, 0.000000, 0.000000, 1.000000, 
1.000000, -0.000000, 0.000000, -0.086875, 0.000000, 1.000000, 0.000000, 0.000000, -0.000000, -0.000000, 1.000000, 0.377425, 0.000000, 0.000000, 0.000000, 1.000000, 
1.000000, -0.000000, 0.000000, -0.086875, 0.000000, 1.000000, 0.000000, 0.000000, -0.000000, -0.000000, 1.000000, 0.377430, 0.000000, 0.000000, 0.000000, 1.000000, 
1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 
1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 
1.000000, 0.000000, 0.000000, 0.053125, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.603001, 0.000000, 0.000000, 0.000000, 1.000000, 
1.000000, 0.000000, 0.000000, 0.142530, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.057999, 0.000000, 0.000000, 0.000000, 1.000000, 
1.000000, 0.000000, 0.000000, 0.055000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.022500, 0.000000, 0.000000, 0.000000, 1.000000, 
1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000, 0.045000, 0.000000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 
0.000000, 0.000000, 1.000000, 0.000000, -1.000000, 0.000000, 0.000000, 0.000000, 0.000000, -1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 
1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000, 0.020000, 0.000000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 
0.000000, 0.000000, 1.000000, 0.000000, -1.000000, 0.000000, 0.000000, 0.000000, 0.000000, -1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 
1.000000, 0.000000, 0.000000, 0.119525, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.348580, 0.000000, 0.000000, 0.000000, 1.000000, 
1.000000, 0.000000, 0.000000, 0.117000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.060000, 0.000000, 0.000000, 0.000000, 1.000000, 
1.000000, 0.000000, 0.000000, 0.219000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 
1.000000, 0.000000, 0.000000, 0.133000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 
1.000000, 0.000000, 0.000000, 0.197000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 
1.000000, 0.000000, 0.000000, 0.124500, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 
1.000000, 0.000000, 0.000000, 0.138500, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 
1.000000, 0.000000, 0.000000, 0.166450, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 
1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000, -0.015425, 0.000000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 
1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000, 0.015425, 0.000000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000
};
__constant__ float joint_axes[90] = 
{
0.000000, 0.000000, 0.000000, 
0.000000, 0.000000, 0.000000, 
0.000000, 1.000000, 0.000000, 
0.000000, 1.000000, 0.000000, 
0.000000, 1.000000, 0.000000, 
0.000000, 0.000000, 0.000000, 
0.000000, 1.000000, 0.000000, 
0.000000, 1.000000, 0.000000, 
0.000000, 1.000000, 0.000000, 
0.000000, 0.000000, 0.000000, 
0.000000, 0.000000, 1.000000, 
0.000000, 0.000000, -1.000000, 
0.000000, 0.000000, 0.000000, 
0.000000, 0.000000, 1.000000, 
0.000000, 1.000000, 0.000000, 
0.000000, 0.000000, 0.000000, 
0.000000, 0.000000, 0.000000, 
0.000000, 0.000000, 0.000000, 
0.000000, 0.000000, 0.000000, 
0.000000, 0.000000, 0.000000, 
0.000000, 0.000000, 1.000000, 
0.000000, 1.000000, 0.000000, 
1.000000, 0.000000, 0.000000, 
0.000000, 1.000000, 0.000000, 
1.000000, 0.000000, 0.000000, 
0.000000, 1.000000, 0.000000, 
1.000000, 0.000000, 0.000000, 
0.000000, 0.000000, 0.000000, 
0.000000, -1.000000, 0.000000, 
0.000000, 1.000000, 0.000000
};
__constant__ float self_collision_spheres_pos_in_link[909] = 
{
-0.102000, -0.023000, 0.439000, 
-0.102000, 0.042000, 0.128000, 
-0.101000, -0.004000, 0.290000, 
-0.095000, -0.091000, 0.143000, 
-0.097000, 0.078000, 0.454000, 
-0.098000, -0.074000, 0.500000, 
-0.095000, 0.085000, 0.322000, 
-0.095000, -0.085000, 0.323000, 
-0.099000, 0.042000, 0.524000, 
-0.096000, 0.084000, 0.233000, 
-0.094000, 0.094000, 0.106000, 
-0.099000, -0.063000, 0.203000, 
0.064000, 0.006000, 0.228000, 
-0.093000, -0.055000, 0.102000, 
-0.092000, -0.095000, 0.394000, 
-0.080000, 0.088000, 0.538000, 
0.037000, 0.003000, 0.530000, 
-0.085000, -0.085000, 0.533000, 
-0.086000, -0.101000, 0.100000, 
-0.100000, 0.063000, 0.382000, 
-0.093000, 0.093000, 0.164000, 
-0.103000, 0.017000, 0.111000, 
-0.098000, -0.013000, 0.513000, 
0.009000, 0.036000, 0.065000, 
-0.081000, -0.106000, 0.282000, 
0.009000, 0.034000, 0.430000, 
0.041000, 0.015000, 0.567000, 
0.143000, 0.001000, 0.256000, 
0.009000, -0.056000, 0.350000, 
-0.084000, 0.104000, 0.498000, 
0.009000, 0.032000, 0.325000, 
-0.101000, -0.002000, 0.221000, 
-0.068000, 0.120000, 0.383000, 
0.008000, -0.075000, 0.433000, 
0.035000, -0.032000, 0.547000, 
0.008000, 0.071000, 0.166000, 
0.010000, -0.044000, 0.106000, 
0.008000, 0.081000, 0.285000, 
-0.070000, -0.118000, 0.202000, 
0.008000, 0.091000, 0.055000, 
0.009000, 0.038000, 0.120000, 
0.008000, -0.084000, 0.255000, 
0.008000, 0.087000, 0.374000, 
0.008000, -0.092000, 0.175000, 
0.008000, -0.091000, 0.071000, 
0.033000, 0.039000, 0.522000, 
-0.090000, -0.096000, 0.511000, 
0.134000, 0.022000, 0.207000, 
-0.101000, -0.026000, 0.335000, 
0.055000, -0.011000, 0.245000, 
0.035000, -0.032000, 0.507000, 
0.009000, -0.076000, 0.306000, 
-0.120000, -0.053000, 0.092000, 
0.004000, -0.028000, 0.157000, 
0.009000, -0.049000, 0.057000, 
-0.075000, 0.112000, 0.280000, 
0.009000, -0.040000, 0.425000, 
0.003000, 0.110000, 0.205000, 
-0.090000, 0.097000, 0.428000, 
0.008000, -0.093000, 0.389000, 
-0.078000, 0.110000, 0.087000, 
0.111000, -0.030000, 0.205000, 
0.102000, 0.017000, 0.583000, 
0.003000, 0.110000, 0.513000, 
0.013000, 0.095000, 0.424000, 
0.002000, -0.117000, 0.486000, 
-0.049000, -0.138000, 0.058000, 
0.009000, 0.055000, 0.372000, 
0.119000, 0.022000, 0.256000, 
-0.097000, -0.073000, 0.271000, 
-0.115000, -0.130000, 0.066000, 
-0.055000, -0.133000, 0.424000, 
0.002000, -0.113000, 0.348000, 
0.002000, 0.118000, 0.462000, 
-0.053000, 0.014000, 0.566000, 
0.003000, -0.111000, 0.228000, 
-0.060000, 0.128000, 0.227000, 
-0.100000, 0.027000, 0.485000, 
0.042000, -0.055000, 0.578000, 
0.099000, -0.022000, 0.583000, 
0.059000, 0.069000, 0.030000, 
0.053000, -0.075000, 0.030000, 
0.038000, 0.116000, 0.029000, 
-0.030000, -0.087000, 0.065000, 
0.063000, 0.005000, 0.032000, 
-0.018000, 0.008000, 0.065000, 
-0.024000, 0.071000, 0.066000, 
0.034000, -0.115000, 0.028000, 
-0.017000, -0.049000, 0.066000, 
0.041000, -0.038000, 0.032000, 
-0.012000, -0.111000, 0.067000, 
0.035000, 0.037000, 0.031000, 
-0.013000, 0.115000, 0.064000, 
-0.049000, 0.043000, 0.062000, 
-0.049000, -0.039000, 0.062000, 
0.004000, -0.092000, 0.024000, 
0.002000, 0.084000, 0.025000, 
0.048000, -0.120000, 0.077000, 
0.080000, -0.123000, 0.031000, 
0.015000, 0.051000, 0.071000, 
0.010000, 0.010000, 0.027000, 
-0.065000, 0.082000, 0.059000, 
-0.069000, 0.007000, 0.058000, 
0.076000, 0.123000, 0.039000, 
0.021000, 0.099000, 0.072000, 
-0.041000, -0.100000, 0.021000, 
0.146000, 0.125000, 0.078000, 
0.111000, -0.124000, 0.080000, 
-0.100000, 0.015000, 0.021000, 
-0.077000, 0.070000, 0.021000, 
0.169000, -0.125000, 0.081000, 
-0.075000, -0.066000, 0.022000, 
-0.087000, -0.024000, 0.022000, 
0.021000, -0.002000, 0.072000, 
0.165000, -0.124000, 0.034000, 
-0.023000, 0.106000, 0.021000, 
0.028000, -0.059000, 0.073000, 
0.000000, -0.043000, 0.025000, 
0.080000, 0.124000, 0.076000, 
0.071000, -0.040000, 0.028000, 
0.131000, -0.124000, 0.024000, 
0.157000, 0.124000, 0.023000, 
0.108000, 0.124000, 0.031000, 
-0.034000, 0.011000, 0.023000, 
-0.005000, 0.053000, 0.025000, 
-0.070000, -0.061000, 0.058000, 
0.056000, 0.052000, 0.075000, 
0.060000, -0.028000, 0.075000, 
-0.008000, -0.120000, 0.021000, 
-0.035000, -0.065000, 0.024000, 
0.075000, -0.124000, 0.093000, 
0.038000, 0.126000, 0.096000, 
-0.048000, 0.076000, 0.022000, 
0.009000, -0.082000, 0.070000, 
0.064000, 0.083000, 0.074000, 
0.072000, -0.059000, 0.074000, 
-0.058000, 0.037000, 0.023000, 
0.055000, 0.022000, 0.074000, 
0.142000, -0.125000, 0.076000, 
-0.051000, -0.018000, 0.024000, 
-0.028000, -0.029000, -0.001000, 
-0.027000, 0.053000, 0.004000, 
0.026000, 0.012000, -0.010000, 
-0.028000, -0.078000, 0.002000, 
0.025000, -0.078000, -0.014000, 
0.026000, 0.066000, -0.014000, 
-0.027000, 0.012000, 0.004000, 
0.025000, -0.027000, -0.013000, 
-0.025000, 0.080000, -0.016000, 
0.021000, -0.077000, 0.033000, 
0.021000, -0.006000, 0.029000, 
0.020000, 0.047000, 0.032000, 
-0.026000, 0.079000, 0.013000, 
0.025000, 0.079000, -0.003000, 
-0.024000, 0.018000, -0.018000, 
-0.022000, -0.065000, -0.023000, 
-0.022000, -0.034000, 0.024000, 
0.020000, -0.043000, 0.017000, 
-0.019000, -0.024000, -0.030000, 
0.039000, 0.089000, 0.039000, 
0.026000, 0.035000, -0.010000, 
-0.021000, 0.039000, 0.027000, 
-0.025000, -0.082000, 0.021000, 
-0.023000, 0.049000, -0.023000, 
0.023000, -0.049000, -0.021000, 
-0.022000, 0.003000, 0.024000, 
0.021000, -0.030000, 0.036000, 
0.016000, 0.075000, 0.039000, 
-0.017000, -0.086000, -0.033000, 
0.019000, 0.031000, 0.036000, 
0.021000, 0.080000, -0.026000, 
0.019000, 0.033000, -0.030000, 
0.020000, -0.011000, -0.028000, 
0.040000, -0.062000, 0.040000, 
-0.019000, -0.067000, 0.031000, 
-0.002000, -0.012000, 0.035000, 
0.144000, -0.031000, 0.052000, 
0.081000, -0.031000, 0.060000, 
0.121000, -0.034000, 0.091000, 
0.115000, -0.033000, 0.030000, 
0.125000, -0.000000, 0.000000, 
0.129000, -0.028000, 0.003000, 
0.024000, -0.026000, -0.005000, 
0.103000, -0.027000, -0.002000, 
0.151000, -0.027000, -0.011000, 
0.023000, 0.001000, -0.002000, 
0.141000, -0.031000, 0.024000, 
0.117000, 0.000000, 0.000000, 
-0.001000, 0.032000, -0.016000, 
0.036000, 0.032000, 0.005000, 
-0.021000, 0.033000, 0.015000, 
0.123000, -0.009000, -0.001000, 
0.087000, 0.023000, -0.001000, 
0.016000, 0.031000, 0.020000, 
0.020000, -0.017000, -0.002000, 
0.076000, -0.042000, 0.002000, 
0.118000, -0.055000, -0.007000, 
-0.002000, 0.022000, 0.002000, 
0.110000, -0.055000, 0.021000, 
0.052000, -0.031000, -0.005000, 
-0.110000, 0.018000, 0.183000, 
0.145000, -0.026000, 0.130000, 
-0.087000, -0.085000, 0.165000, 
0.064000, 0.134000, 0.134000, 
0.101000, -0.122000, 0.132000, 
-0.065000, 0.108000, 0.155000, 
0.148000, 0.075000, 0.121000, 
-0.077000, 0.130000, 0.230000, 
-0.068000, -0.140000, 0.236000, 
-0.174000, -0.062000, 0.256000, 
-0.180000, -0.050000, 0.098000, 
-0.035000, -0.130000, 0.139000, 
-0.160000, 0.068000, 0.112000, 
-0.173000, 0.081000, 0.263000, 
-0.020000, 0.191000, 0.289000, 
0.136000, 0.115000, 0.152000, 
0.204000, 0.041000, 0.182000, 
0.187000, -0.090000, 0.182000, 
-0.015000, -0.188000, 0.293000, 
0.029000, 0.050000, 0.131000, 
-0.178000, -0.126000, 0.289000, 
0.170000, -0.087000, 0.100000, 
-0.037000, -0.007000, 0.272000, 
0.056000, -0.185000, 0.181000, 
-0.155000, -0.122000, 0.091000, 
-0.160000, 0.140000, 0.080000, 
-0.150000, 0.126000, 0.267000, 
-0.206000, 0.026000, 0.283000, 
0.114000, 0.208000, 0.212000, 
0.109000, 0.169000, 0.100000, 
0.162000, -0.002000, 0.128000, 
-0.021000, 0.084000, 0.287000, 
0.079000, -0.164000, 0.112000, 
-0.103000, -0.120000, 0.231000, 
-0.106000, -0.200000, 0.080000, 
-0.201000, 0.044000, 0.082000, 
0.167000, -0.175000, 0.215000, 
0.217000, 0.014000, 0.333000, 
0.137000, -0.010000, 0.331000, 
0.180000, 0.174000, 0.331000, 
0.055000, -0.044000, 0.130000, 
0.114000, -0.083000, 0.333000, 
0.079000, -0.228000, 0.331000, 
0.068000, -0.149000, 0.331000, 
0.099000, 0.091000, 0.331000, 
0.178000, -0.103000, 0.333000, 
0.040000, -0.069000, 0.331000, 
0.240000, 0.086000, 0.331000, 
0.159000, 0.074000, 0.331000, 
0.096000, 0.182000, 0.333000, 
0.236000, -0.102000, 0.331000, 
0.119000, -0.173000, 0.331000, 
0.232000, -0.042000, 0.331000, 
-0.080000, 0.189000, 0.093000, 
0.243000, -0.020000, 0.218000, 
0.013000, 0.204000, 0.211000, 
0.168000, -0.166000, 0.333000, 
0.100000, 0.041000, 0.333000, 
0.003000, 0.220000, 0.070000, 
0.200000, 0.136000, 0.215000, 
0.211000, 0.142000, 0.333000, 
0.130000, 0.226000, 0.334000, 
-0.113000, -0.020000, 0.186000, 
0.101000, -0.038000, 0.331000, 
0.129000, 0.117000, 0.333000, 
-0.100000, 0.064000, 0.181000, 
0.133000, -0.205000, 0.218000, 
-0.013000, -0.134000, 0.296000, 
0.063000, 0.103000, 0.331000, 
-0.226000, -0.024000, 0.304000, 
0.036000, -0.000000, -0.001000, 
-0.000000, 0.002000, -0.000000, 
0.066000, 0.008000, 0.001000, 
-0.001000, 0.017000, 0.002000, 
-0.002000, -0.009000, -0.009000, 
-0.002000, 0.018000, 0.018000, 
-0.002000, 0.018000, -0.018000, 
-0.002000, -0.011000, 0.023000, 
-0.003000, -0.033000, 0.005000, 
0.069000, 0.011000, 0.001000, 
0.093000, -0.024000, 0.002000, 
0.053000, -0.025000, 0.002000, 
0.098000, 0.028000, 0.000000, 
0.049000, 0.026000, 0.001000, 
0.098000, 0.004000, 0.002000, 
0.105000, -0.034000, -0.004000, 
0.038000, -0.011000, -0.014000, 
-0.010000, 0.009000, -0.004000, 
0.015000, 0.009000, 0.005000, 
0.005000, 0.009000, -0.006000, 
-0.020000, 0.009000, 0.003000, 
-0.001000, 0.009000, 0.006000, 
0.019000, 0.009000, -0.005000, 
-0.022000, 0.008000, -0.007000, 
0.024000, 0.009000, 0.004000, 
-0.010000, -0.009000, -0.004000, 
0.015000, -0.009000, 0.005000, 
0.005000, -0.009000, -0.006000, 
-0.020000, -0.009000, 0.003000, 
-0.001000, -0.009000, 0.006000, 
0.019000, -0.009000, -0.005000, 
-0.022000, -0.009000, -0.007000, 
0.024000, -0.008000, 0.004000
};
__constant__ int self_collision_spheres_to_link_map[303] = 
{
10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 20, 20, 20, 20, 20, 21, 22, 22, 22, 22, 22, 22, 23, 23, 23, 23, 23, 23, 23, 24, 24, 24, 24, 24, 24, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25, 25, 25, 25, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 28, 28, 28, 28, 28, 28, 28, 28, 29, 29, 29, 29, 29, 29, 29, 29
};

 // Multiply two 4x4 matrices (row-major order)
__device__ __forceinline__ void multiply4x4(const float* __restrict__ A, 
                                            const float* __restrict__ B, 
                                            float* __restrict__ C)
{
    C[0] = A[0] * B[0] + A[1] * B[4] + A[2] * B[8]  + A[3] * B[12];
    C[1] = A[0] * B[1] + A[1] * B[5] + A[2] * B[9]  + A[3] * B[13];
    C[2] = A[0] * B[2] + A[1] * B[6] + A[2] * B[10] + A[3] * B[14];
    C[3] = A[0] * B[3] + A[1] * B[7] + A[2] * B[11] + A[3] * B[15];

    C[4] = A[4] * B[0] + A[5] * B[4] + A[6] * B[8]  + A[7] * B[12];
    C[5] = A[4] * B[1] + A[5] * B[5] + A[6] * B[9]  + A[7] * B[13];
    C[6] = A[4] * B[2] + A[5] * B[6] + A[6] * B[10] + A[7] * B[14];
    C[7] = A[4] * B[3] + A[5] * B[7] + A[6] * B[11] + A[7] * B[15];

    C[8]  = A[8]  * B[0] + A[9]  * B[4] + A[10] * B[8]  + A[11] * B[12];
    C[9]  = A[8]  * B[1] + A[9]  * B[5] + A[10] * B[9]  + A[11] * B[13];
    C[10] = A[8]  * B[2] + A[9]  * B[6] + A[10] * B[10] + A[11] * B[14];
    C[11] = A[8]  * B[3] + A[9]  * B[7] + A[10] * B[11] + A[11] * B[15];

    // Last row is fixed as [0, 0, 0, 1]
    C[12] = 0.f; C[13] = 0.f; C[14] = 0.f; C[15] = 1.f;
}

// Fixed joint: multiply parent's pose with joint's fixed pose.
__device__ __forceinline__ void fixed_joint_fn_cuda(const float* parent_link_pose,
                                                    const float* joint_pose,
                                                    float* link_pose)
{
    multiply4x4(parent_link_pose, joint_pose, link_pose);
}

// Create a rotation matrix from an axis-angle representation.
__device__ __forceinline__ void make_rotation_axis_angle(float angle, float x, float y, float z, float* R)
{
    float length = sqrtf(x * x + y * y + z * z);
    const float thresh = 1e-12f;
    float valid = (length >= thresh) ? 1.f : 0.f;
    float inv_length = 1.f / fmaxf(length, thresh);
    float nx = x * inv_length * valid;
    float ny = y * inv_length * valid;
    float nz = z * inv_length * valid;
    float c = cosf(angle);
    float s = sinf(angle);
    float one_c = 1.f - c;

    float r0  = c + nx * nx * one_c;
    float r1  = nx * ny * one_c - nz * s;
    float r2  = nx * nz * one_c + ny * s;
    float r4  = ny * nx * one_c + nz * s;
    float r5  = c + ny * ny * one_c;
    float r6  = ny * nz * one_c - nx * s;
    float r8  = nz * nx * one_c - ny * s;
    float r9  = nz * ny * one_c + nx * s;
    float r10 = c + nz * nz * one_c;

    R[0]  = r0 * valid + (1.f - valid) * 1.f; R[1]  = r1 * valid;           R[2]  = r2 * valid;           R[3]  = 0.f;
    R[4]  = r4 * valid;           R[5]  = r5 * valid + (1.f - valid) * 1.f; R[6]  = r6 * valid;           R[7]  = 0.f;
    R[8]  = r8 * valid;           R[9]  = r9 * valid;           R[10] = r10 * valid + (1.f - valid) * 1.f; R[11] = 0.f;
    R[12] = 0.f; R[13] = 0.f; R[14] = 0.f; R[15] = 1.f;
}

// Revolute joint: compute rotation transformation then multiply with parent's pose.
__device__ __forceinline__ void revolute_joint_fn_cuda(const float* parent_link_pose,
                                                        const float* joint_pose,
                                                        const float* joint_axis,
                                                        float joint_value,
                                                        float* link_pose)
{
    float joint_transform[16];
    make_rotation_axis_angle(joint_value, joint_axis[0], joint_axis[1], joint_axis[2], joint_transform);
    
    float temp[16];
    multiply4x4(parent_link_pose, joint_pose, temp);
    multiply4x4(temp, joint_transform, link_pose);
}

// Prismatic joint: create a translation matrix and combine with parent's pose.
__device__ __forceinline__ void prism_joint_fn_cuda(const float* parent_link_pose,
                                                    const float* joint_pose,
                                                    const float* joint_axis,
                                                    float joint_value,
                                                    float* link_pose)
{
    float x = joint_axis[0], y = joint_axis[1], z = joint_axis[2];
    float T[16] = {
        1.0f, 0.0f, 0.0f, x * joint_value,
        0.0f, 1.0f, 0.0f, y * joint_value,
        0.0f, 0.0f, 1.0f, z * joint_value,
        0.0f, 0.0f, 0.0f, 1.0f
    };
    float joint_pose_T[16];
    multiply4x4(joint_pose, T, joint_pose_T);
    multiply4x4(parent_link_pose, joint_pose_T, link_pose);
}       

__device__ __forceinline__ void kin_forward(float * configuration, float * self_collision_spheres)
{
    // based on the default value and fixed joint, calculate the full joint values.
    float full_joint_values[30];
    full_joint_values[0] = 0.000000;
    full_joint_values[1] = 0.000000;
    full_joint_values[2] = 0.000000;
    full_joint_values[3] = 0.000000;
    full_joint_values[4] = 0.000000;
    full_joint_values[5] = 0.000000;
    full_joint_values[6] = 0.000000;
    full_joint_values[7] = 0.000000;
    full_joint_values[8] = 0.000000;
    full_joint_values[9] = 0.000000;
    full_joint_values[10] = 0.000000;
    full_joint_values[11] = 0.000000;
    full_joint_values[12] = 0.000000;
    full_joint_values[13] = 0.000000;
    full_joint_values[14] = 0.000000;
    full_joint_values[15] = 0.000000;
    full_joint_values[16] = 0.000000;
    full_joint_values[17] = 0.000000;
    full_joint_values[18] = 0.000000;
    full_joint_values[19] = 0.000000;
    full_joint_values[20] = configuration[0];
    full_joint_values[21] = configuration[1];
    full_joint_values[22] = configuration[2];
    full_joint_values[23] = configuration[3];
    full_joint_values[24] = configuration[4];
    full_joint_values[25] = configuration[5];
    full_joint_values[26] = configuration[6];
    full_joint_values[27] = 0.000000;
    full_joint_values[28] = 0.000000;
    full_joint_values[29] = 0.000000;
    float link_poses[480];
    // set the base link pose to identity
    link_poses[0] = 1.0f; link_poses[1] = 0.0f; link_poses[2] = 0.0f; link_poses[3] = 0.0f;
    link_poses[4] = 0.0f; link_poses[5] = 1.0f; link_poses[6] = 0.0f; link_poses[7] = 0.0f;
    link_poses[8] = 0.0f; link_poses[9] = 0.0f; link_poses[10] = 1.0f; link_poses[11] = 0.0f;
    link_poses[12] = 0.0f; link_poses[13] = 0.0f; link_poses[14] = 0.0f; link_poses[15] = 1.0f;
    // Unrolled joint 1
    // fixed joint
    fixed_joint_fn_cuda( &link_poses[0], 
        &joint_poses[16], 
        &link_poses[32] 
    );
    // Unrolled joint 2
    // revolute joint
    revolute_joint_fn_cuda( &link_poses[0], 
        &joint_poses[32], 
        &joint_axes[6], 
        full_joint_values[2], 
        &link_poses[48] 
    );
    // Unrolled joint 3
    // revolute joint
    revolute_joint_fn_cuda( &link_poses[0], 
        &joint_poses[48], 
        &joint_axes[9], 
        full_joint_values[3], 
        &link_poses[64] 
    );
    // Unrolled joint 4
    // revolute joint
    revolute_joint_fn_cuda( &link_poses[0], 
        &joint_poses[64], 
        &joint_axes[12], 
        full_joint_values[4], 
        &link_poses[80] 
    );
    // Unrolled joint 5
    // fixed joint
    fixed_joint_fn_cuda( &link_poses[0], 
        &joint_poses[80], 
        &link_poses[96] 
    );
    // Unrolled joint 6
    // revolute joint
    revolute_joint_fn_cuda( &link_poses[0], 
        &joint_poses[96], 
        &joint_axes[18], 
        full_joint_values[6], 
        &link_poses[112] 
    );
    // Unrolled joint 7
    // revolute joint
    revolute_joint_fn_cuda( &link_poses[0], 
        &joint_poses[112], 
        &joint_axes[21], 
        full_joint_values[7], 
        &link_poses[128] 
    );
    // Unrolled joint 8
    // revolute joint
    revolute_joint_fn_cuda( &link_poses[0], 
        &joint_poses[128], 
        &joint_axes[24], 
        full_joint_values[8], 
        &link_poses[144] 
    );
    // Unrolled joint 9
    // fixed joint
    fixed_joint_fn_cuda( &link_poses[0], 
        &joint_poses[144], 
        &link_poses[160] 
    );
    // Unrolled joint 10
    // prismatic joint
    prism_joint_fn_cuda( &link_poses[0], 
        &joint_poses[160], 
        &joint_axes[30], 
        full_joint_values[10], 
        &link_poses[176] 
    );
    // Unrolled joint 11
    // prismatic joint
    prism_joint_fn_cuda( &link_poses[160], 
        &joint_poses[176], 
        &joint_axes[33], 
        full_joint_values[11], 
        &link_poses[192] 
    );
    // Unrolled joint 12
    // fixed joint
    fixed_joint_fn_cuda( &link_poses[160], 
        &joint_poses[192], 
        &link_poses[208] 
    );
    // Unrolled joint 13
    // revolute joint
    revolute_joint_fn_cuda( &link_poses[160], 
        &joint_poses[208], 
        &joint_axes[39], 
        full_joint_values[13], 
        &link_poses[224] 
    );
    // Unrolled joint 14
    // revolute joint
    revolute_joint_fn_cuda( &link_poses[208], 
        &joint_poses[224], 
        &joint_axes[42], 
        full_joint_values[14], 
        &link_poses[240] 
    );
    // Unrolled joint 15
    // fixed joint
    fixed_joint_fn_cuda( &link_poses[224], 
        &joint_poses[240], 
        &link_poses[256] 
    );
    // Unrolled joint 16
    // fixed joint
    fixed_joint_fn_cuda( &link_poses[240], 
        &joint_poses[256], 
        &link_poses[272] 
    );
    // Unrolled joint 17
    // fixed joint
    fixed_joint_fn_cuda( &link_poses[256], 
        &joint_poses[272], 
        &link_poses[288] 
    );
    // Unrolled joint 18
    // fixed joint
    fixed_joint_fn_cuda( &link_poses[240], 
        &joint_poses[288], 
        &link_poses[304] 
    );
    // Unrolled joint 19
    // fixed joint
    fixed_joint_fn_cuda( &link_poses[288], 
        &joint_poses[304], 
        &link_poses[320] 
    );
    // Unrolled joint 20
    // revolute joint
    revolute_joint_fn_cuda( &link_poses[160], 
        &joint_poses[320], 
        &joint_axes[60], 
        full_joint_values[20], 
        &link_poses[336] 
    );
    // Unrolled joint 21
    // revolute joint
    revolute_joint_fn_cuda( &link_poses[320], 
        &joint_poses[336], 
        &joint_axes[63], 
        full_joint_values[21], 
        &link_poses[352] 
    );
    // Unrolled joint 22
    // revolute joint
    revolute_joint_fn_cuda( &link_poses[336], 
        &joint_poses[352], 
        &joint_axes[66], 
        full_joint_values[22], 
        &link_poses[368] 
    );
    // Unrolled joint 23
    // revolute joint
    revolute_joint_fn_cuda( &link_poses[352], 
        &joint_poses[368], 
        &joint_axes[69], 
        full_joint_values[23], 
        &link_poses[384] 
    );
    // Unrolled joint 24
    // revolute joint
    revolute_joint_fn_cuda( &link_poses[368], 
        &joint_poses[384], 
        &joint_axes[72], 
        full_joint_values[24], 
        &link_poses[400] 
    );
    // Unrolled joint 25
    // revolute joint
    revolute_joint_fn_cuda( &link_poses[384], 
        &joint_poses[400], 
        &joint_axes[75], 
        full_joint_values[25], 
        &link_poses[416] 
    );
    // Unrolled joint 26
    // revolute joint
    revolute_joint_fn_cuda( &link_poses[400], 
        &joint_poses[416], 
        &joint_axes[78], 
        full_joint_values[26], 
        &link_poses[432] 
    );
    // Unrolled joint 27
    // fixed joint
    fixed_joint_fn_cuda( &link_poses[416], 
        &joint_poses[432], 
        &link_poses[448] 
    );
    // Unrolled joint 28
    // prismatic joint
    prism_joint_fn_cuda( &link_poses[432], 
        &joint_poses[448], 
        &joint_axes[84], 
        full_joint_values[28], 
        &link_poses[464] 
    );
    // Unrolled joint 29
    // prismatic joint
    prism_joint_fn_cuda( &link_poses[432], 
        &joint_poses[464], 
        &joint_axes[87], 
        full_joint_values[29], 
        &link_poses[480] 
    );
    // calculate the self collision spheres positions in the base link frame
    #pragma unroll
    for (int i = 0; i < 303; i++)
    {
        float * T = &link_poses[self_collision_spheres_to_link_map[i] * 16];
        float sx = self_collision_spheres_pos_in_link[i * 3 + 0];
        float sy = self_collision_spheres_pos_in_link[i * 3 + 1];
        float sz = self_collision_spheres_pos_in_link[i * 3 + 2];
        float x = T[0] * sx + T[1] * sy + T[2] * sz + T[3];
        float y = T[4] * sx + T[5] * sy + T[6] * sz + T[7];
        float z = T[8] * sx + T[9] * sy + T[10] * sz + T[11];
        self_collision_spheres[i * 3 + 0] = x;
        self_collision_spheres[i * 3 + 1] = y;
        self_collision_spheres[i * 3 + 2] = z;
    }

}
        
extern "C" __global__ void cRRTCKernel(float * d_start_tree_configurations, float * d_goal_tree_configurations, int * d_start_tree_parent_indexs, int * d_goal_tree_parent_indexs, float * d_sampled_configurations) {
    __shared__ float * tree_to_expand;
    __shared__ int * tree_to_expand_parent_indexs;
    __shared__ int localTargetTreeCounter;
    __shared__ int localSampledCounter;
    __shared__ int localStartTreeCounter;
    __shared__ int localGoalTreeCounter;
    __shared__ float partial_distance_cost_from_nn[32];
    __shared__ int partial_nn_index[32];
    __shared__ float local_sampled_configuration[7];
    __shared__ float local_parent_configuration[7];
    __shared__ float local_delta_motion[7];
    __shared__ int local_parent_index;
    __shared__ float local_nearest_neighbor_distance;
    __shared__ float local_motion_configurations[224]; 
    __shared__ int motion_step;
    const int tid = threadIdx.x;
    float self_collision_spheres_pos_in_base[909];

    // run for loop with max_interations_ iterations
    for (int i = 0; i < 1; i++) {

        // Need to decide which tree to expand based on their sizes. The smaller tree will be expanded.
        if (tid == 0)
        {
            // increase the sampledCounter with atomic operation
            localSampledCounter = atomicAdd(&sampledCounter, 1);
            localStartTreeCounter = startTreeCounter;
            localGoalTreeCounter = goalTreeCounter;

            if (localStartTreeCounter < localGoalTreeCounter) {
                tree_to_expand = d_start_tree_configurations;
                tree_to_expand_parent_indexs = d_start_tree_parent_indexs;
                localTargetTreeCounter = localStartTreeCounter;
            } else {
                tree_to_expand = d_goal_tree_configurations;
                tree_to_expand_parent_indexs = d_goal_tree_parent_indexs;
                localTargetTreeCounter = localGoalTreeCounter;
            }
        }

        __syncthreads();
        if (localSampledCounter >= 1)
            return; // meet the max_iteration, then stop the block.
        if(tid == 0) {
            printf("localStartTreeCounter: %d\n", localStartTreeCounter);
            printf("localGoalTreeCounter: %d\n", localGoalTreeCounter);
            printf("localSampledCounter: %d\n", localSampledCounter);
            printf("Sampled configuration: ");
            printf("%f ", d_sampled_configurations[localSampledCounter * 7 + 0]);
            printf("%f ", d_sampled_configurations[localSampledCounter * 7 + 1]);
            printf("%f ", d_sampled_configurations[localSampledCounter * 7 + 2]);
            printf("%f ", d_sampled_configurations[localSampledCounter * 7 + 3]);
            printf("%f ", d_sampled_configurations[localSampledCounter * 7 + 4]);
            printf("%f ", d_sampled_configurations[localSampledCounter * 7 + 5]);
            printf("%f ", d_sampled_configurations[localSampledCounter * 7 + 6]);
            printf("\n");
        }
        // Load the sampled configuration into shared memory
        if (tid < 7) {
            local_sampled_configuration[tid] = d_sampled_configurations[localSampledCounter * 7 + tid];
        }
        __syncthreads();

        // Find the nearest configuration in the tree_to_expand to the sampled configuration with reduction operation

        float best_dist = FLT_MAX;
        int best_index = -1;
        for (int j = tid; j < localTargetTreeCounter; j += blockDim.x){
            float dist = 0.0f;
            float diff = 0.0f;
            diff = tree_to_expand[j * 7 + 0] - local_sampled_configuration[0];
            dist += diff * diff;
            diff = tree_to_expand[j * 7 + 1] - local_sampled_configuration[1];
            dist += diff * diff;
            diff = tree_to_expand[j * 7 + 2] - local_sampled_configuration[2];
            dist += diff * diff;
            diff = tree_to_expand[j * 7 + 3] - local_sampled_configuration[3];
            dist += diff * diff;
            diff = tree_to_expand[j * 7 + 4] - local_sampled_configuration[4];
            dist += diff * diff;
            diff = tree_to_expand[j * 7 + 5] - local_sampled_configuration[5];
            dist += diff * diff;
            diff = tree_to_expand[j * 7 + 6] - local_sampled_configuration[6];
            dist += diff * diff;

            if (dist < best_dist) {
                best_dist = dist;
                best_index = j;
            }
        }

        // Write the local best distance and index to the shared memory
        partial_distance_cost_from_nn[tid] = best_dist;
        partial_nn_index[tid] = best_index;
        __syncthreads();

        // Perform reduction to find the best distance and index
        for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
            if (tid < stride) {
                if (partial_distance_cost_from_nn[tid + stride] < partial_distance_cost_from_nn[tid]) {
                    partial_distance_cost_from_nn[tid] = partial_distance_cost_from_nn[tid + stride];
                    partial_nn_index[tid] = partial_nn_index[tid + stride];
                }
            }
            __syncthreads();
        }

        // After the reduction, thread 0 has the overall nearest neighbor's index and its squared distance.
        if (tid == 0) {
            local_nearest_neighbor_distance = sqrtf(partial_distance_cost_from_nn[0]);
            local_parent_index = partial_nn_index[0];
            motion_step = min((int)(local_nearest_neighbor_distance / 0.020000), 32);
            printf("Nearest neighbor index: %d, Euclidean distance: %f motion step: %d \n ", local_parent_index, local_nearest_neighbor_distance, motion_step);
        }
        __syncthreads();
        // Calculate the delta motion from the nearest configuration to the sampled configuration
        if (tid < 7) {
            local_parent_configuration[tid] = tree_to_expand[local_parent_index * 7 + tid];
            local_delta_motion[tid] = (local_sampled_configuration[tid] - local_parent_configuration[tid]) / local_nearest_neighbor_distance * 0.020000;
        }

        __syncthreads();
        // interpolate the new configuration from the nearest configuration and the sampled configuration
        for (int j = tid; j < 7 * motion_step; j += blockDim.x) {
            int state_ind_in_motion = j / 7;
            int joint_ind_in_state = j % 7;
            local_motion_configurations[j] = local_parent_configuration[joint_ind_in_state] + local_delta_motion[joint_ind_in_state] * state_ind_in_motion;
        }
        __syncthreads();

        // call the forward kinematics kernel
        kin_forward(&(local_motion_configurations[tid]), self_collision_spheres_pos_in_base);
        __syncthreads();

    }

}