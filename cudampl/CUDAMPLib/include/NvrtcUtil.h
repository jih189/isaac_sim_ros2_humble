#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <nvrtc.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA driver API error checking macro.
#define DRIVER_SAFE_CALL(call)                                                   \
    do {                                                                         \
        CUresult res = call;                                                     \
        if (res != CUDA_SUCCESS) {                                               \
            const char* errStr;                                                  \
            cuGetErrorName(res, &errStr);                                        \
            std::cerr << "CUDA driver error in " << __FILE__ << "@" << __LINE__    \
                      << ": " << errStr << std::endl;                            \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    } while (0)

// NVRTC error checking macro.
#define NVRTC_SAFE_CALL(call)                                                    \
    do {                                                                         \
        nvrtcResult res = call;                                                  \
        if (res != NVRTC_SUCCESS) {                                              \
            std::cerr << "NVRTC error in " << __FILE__ << "@" << __LINE__ << ": " \
                      << nvrtcGetErrorString(res) << std::endl;                  \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    } while (0)

namespace CUDAMPLib
{
    // Class holding CUDA context, module, and kernel function.
    class KernelFunction {
    public:
        CUcontext context = nullptr;
        CUmodule module = nullptr;
        CUfunction function = nullptr;

        ~KernelFunction() {
            if (module) {
                // std::cout << "Destroying module" << std::endl;
                DRIVER_SAFE_CALL(cuModuleUnload(module));
            }
            if (context) {
                // std::cout << "Destroying context" << std::endl;
                DRIVER_SAFE_CALL(cuCtxDestroy(context));
            }
        }

        // Factory method to compile a kernel, load its module, and return a shared_ptr.
        static std::shared_ptr<KernelFunction> create(const char* kernel_code, const char* kernel_name) {
            std::shared_ptr<KernelFunction> kf(new KernelFunction());

            // 1. Initialize CUDA driver API.
            DRIVER_SAFE_CALL(cuInit(0));
            CUdevice cuDevice;
            DRIVER_SAFE_CALL(cuDeviceGet(&cuDevice, 0));

            // 2. Detect GPU architecture.
            int major = 0, minor = 0;
            DRIVER_SAFE_CALL(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice));
            DRIVER_SAFE_CALL(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice));
            std::string arch_option = "--gpu-architecture=compute_" + std::to_string(major) + std::to_string(minor);
            // std::cout << "Detected GPU architecture: " << arch_option << std::endl;

            // 3. Create and compile the NVRTC program.
            nvrtcProgram prog;
            NVRTC_SAFE_CALL(nvrtcCreateProgram(&prog, kernel_code, "kernel.cu", 0, nullptr, nullptr));
            const char* opts[] = { arch_option.c_str() };
            nvrtcResult compileResult = nvrtcCompileProgram(prog, 1, opts);

            // Print compilation log if available.
            size_t logSize;
            NVRTC_SAFE_CALL(nvrtcGetProgramLogSize(prog, &logSize));
            if (logSize > 1) {
                std::vector<char> log(logSize);
                NVRTC_SAFE_CALL(nvrtcGetProgramLog(prog, log.data()));
                std::cout << "Compilation log:\n" << log.data() << std::endl;
            }
            if (compileResult != NVRTC_SUCCESS) {
                std::cerr << "Failed to compile CUDA kernel." << std::endl;
                exit(EXIT_FAILURE);
            }

            // 4. Retrieve the PTX code.
            size_t ptxSize;
            NVRTC_SAFE_CALL(nvrtcGetPTXSize(prog, &ptxSize));
            std::vector<char> ptx(ptxSize);
            NVRTC_SAFE_CALL(nvrtcGetPTX(prog, ptx.data()));
            NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog));

            // 5. Create a CUDA context.
            DRIVER_SAFE_CALL(cuCtxCreate(&kf->context, 0, cuDevice));

            // 6. Load the PTX module.
            DRIVER_SAFE_CALL(cuModuleLoadDataEx(&kf->module, ptx.data(), 0, nullptr, nullptr));

            // 7. Get the kernel function handle.
            DRIVER_SAFE_CALL(cuModuleGetFunction(&kf->function, kf->module, kernel_name));

            return kf;
        }

        // Member function to launch the kernel.
        // You can pass grid dimensions, block dimensions, shared memory size, stream, and kernel parameters.
        void launchKernel(dim3 gridDim, dim3 blockDim, size_t sharedMem, CUstream stream, void** kernelParams) {
            DRIVER_SAFE_CALL(cuLaunchKernel(function,
                                            gridDim.x, gridDim.y, gridDim.z,
                                            blockDim.x, blockDim.y, blockDim.z,
                                            sharedMem, stream,
                                            kernelParams, nullptr));
        }

    private:
        KernelFunction() = default;
    };

    #define KernelFunctionPtr std::shared_ptr<KernelFunction>

} // namespace CUDAMPLib