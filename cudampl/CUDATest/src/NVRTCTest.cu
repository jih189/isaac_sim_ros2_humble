#include <nvrtc.h>
#include <cuda.h>
#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>

// Macro to check NVRTC errors.
#define NVRTC_SAFE_CALL(x) do { nvrtcResult result = x; \
    if (result != NVRTC_SUCCESS) { \
        std::cerr << "NVRTC error: " << nvrtcGetErrorString(result) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
} while(0)

// Macro to check CUDA Driver API errors.
#define CUDA_SAFE_CALL(call) do { \
    CUresult err = call; \
    if (err != CUDA_SUCCESS) { \
        const char *errStr; \
        cuGetErrorName(err, &errStr); \
        std::cerr << "CUDA error: " << errStr \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
} while(0)



// Structure to hold our compiled kernel function and module.
struct KernelFunction {
    CUmodule module;
    CUfunction function;
};

// This function uses NVRTC to compile a simple kernel that adds one to every element
// in an array and returns the compiled kernel function along with its module.
KernelFunction compileAddOneKernel() {
    // The kernel source code as a string.
    std::string kernelSource = R"(
        extern "C" __device__ __forceinline__ void multiple_two(float *n) {
            n[0] = n[0] * 2;
        }

        extern "C" __global__
        void addOneKernel(float* data, int n) {
            int idx = threadIdx.x + blockIdx.x * blockDim.x;
            if (idx < n) {
                multiple_two(&data[idx]);
                data[idx] += 1.0f;
            }
        }
    )";

    // Create an NVRTC program with the kernel source.
    nvrtcProgram prog;
    NVRTC_SAFE_CALL(nvrtcCreateProgram(&prog, kernelSource.c_str(),
                                         "addOneKernel.cu", 0, nullptr, nullptr));

    // Compile the program for a target GPU architecture.
    const char *opts[] = {"--gpu-architecture=compute_61"};
    NVRTC_SAFE_CALL(nvrtcCompileProgram(prog, 1, opts));

    // Optionally, get and print the compilation log.
    size_t logSize;
    NVRTC_SAFE_CALL(nvrtcGetProgramLogSize(prog, &logSize));
    if (logSize > 1) {
        std::string log(logSize, '\0');
        NVRTC_SAFE_CALL(nvrtcGetProgramLog(prog, &log[0]));
        std::cout << "NVRTC compilation log:\n" << log << std::endl;
    }

    // Get the PTX (compiled CUDA code).
    size_t ptxSize;
    NVRTC_SAFE_CALL(nvrtcGetPTXSize(prog, &ptxSize));
    std::string ptx(ptxSize, '\0');
    NVRTC_SAFE_CALL(nvrtcGetPTX(prog, &ptx[0]));

    // Destroy the NVRTC program.
    NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog));

    // Initialize the CUDA Driver API and create a context.
    CUDA_SAFE_CALL(cuInit(0));
    CUdevice cuDevice;
    CUDA_SAFE_CALL(cuDeviceGet(&cuDevice, 0));
    CUcontext context;
    CUDA_SAFE_CALL(cuCtxCreate(&context, 0, cuDevice));

    // Load the PTX module.
    CUmodule module;
    CUDA_SAFE_CALL(cuModuleLoadDataEx(&module, ptx.c_str(), 0, nullptr, nullptr));

    // Get the kernel function from the module.
    CUfunction function;
    CUDA_SAFE_CALL(cuModuleGetFunction(&function, module, "addOneKernel"));

    KernelFunction kf = {module, function};
    return kf;
}

int main() {
    // Compile the kernel and obtain the function.
    KernelFunction kf = compileAddOneKernel();

    // Prepare input data.
    int n = 10;
    std::vector<float> h_data(n);
    for (int i = 0; i < n; ++i) {
        h_data[i] = static_cast<float>(i);
    }
    std::cout << "Input data: ";
    for (const auto &val : h_data)
        std::cout << val << " ";
    std::cout << "\n";

    // Allocate device memory.
    CUdeviceptr d_data;
    CUDA_SAFE_CALL(cuMemAlloc(&d_data, n * sizeof(float)));

    // Copy data from host to device.
    CUDA_SAFE_CALL(cuMemcpyHtoD(d_data, h_data.data(), n * sizeof(float)));

    // Set up kernel parameters.
    void* args[] = { &d_data, &n };

    // Launch the kernel.
    int threadsPerBlock = 16;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    CUDA_SAFE_CALL(cuLaunchKernel(kf.function,
                                  blocks, 1, 1,          // Grid dimensions.
                                  threadsPerBlock, 1, 1,   // Block dimensions.
                                  0, 0,                   // Shared memory and stream.
                                  args, 0));

    // Wait for the kernel to finish.
    CUDA_SAFE_CALL(cuCtxSynchronize());

    // Copy the results back to the host.
    CUDA_SAFE_CALL(cuMemcpyDtoH(h_data.data(), d_data, n * sizeof(float)));

    // Print the output data.
    std::cout << "Output data: ";
    for (const auto &val : h_data)
        std::cout << val << " ";
    std::cout << "\n";

    // Clean up device memory and unload the module.
    CUDA_SAFE_CALL(cuMemFree(d_data));
    CUDA_SAFE_CALL(cuModuleUnload(kf.module));

    // Destroy the context.
    CUcontext context;
    cuCtxGetCurrent(&context);
    cuCtxDestroy(context);

    return 0;
}
