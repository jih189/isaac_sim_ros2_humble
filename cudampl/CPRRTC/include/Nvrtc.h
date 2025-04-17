#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <fstream>  // Added for file I/O operations.
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

namespace CPRRTC
{
    // Class holding CUDA context, module, and kernel function.
    class KernelFunction {
    public:
        CUcontext context = nullptr;
        CUmodule module = nullptr;
        CUfunction function = nullptr;
        bool owns_context = false; // true if this object created the context

        ~KernelFunction() {
            if (module) {
                // Unload module.
                DRIVER_SAFE_CALL(cuModuleUnload(module));
            }
            // Only destroy context if we created it.
            if (owns_context && context) {
                DRIVER_SAFE_CALL(cuCtxDestroy(context));
            }
        }

        // Factory method to compile a kernel, load its module (or cached PTX), and return a shared_ptr.
        static std::shared_ptr<KernelFunction> create(const char* kernel_code, const char* kernel_name) {
            std::shared_ptr<KernelFunction> kf(new KernelFunction());

            // Determine the PTX file name based on the kernel name.
            std::string ptx_filename = std::string(kernel_name) + ".ptx";
            std::vector<char> ptx;

            // Check if cached PTX file exists.
            std::ifstream ptxFile(ptx_filename, std::ios::binary);
            if (ptxFile.good()) {
                DRIVER_SAFE_CALL(cuInit(0));
                CUdevice cuDevice;
                DRIVER_SAFE_CALL(cuDeviceGet(&cuDevice, 0));

                // Load PTX code from the file.
                ptxFile.seekg(0, std::ios::end);
                std::streamsize size = ptxFile.tellg();
                ptxFile.seekg(0, std::ios::beg);
                ptx.resize(size);
                if (!ptxFile.read(ptx.data(), size)) {
                    std::cerr << "Error reading PTX file: " << ptx_filename << std::endl;
                    exit(EXIT_FAILURE);
                }
                std::cout << "Loaded cached PTX from " << ptx_filename << std::endl;
            } else {
                // 1. Initialize CUDA driver API.
                DRIVER_SAFE_CALL(cuInit(0));
                CUdevice cuDevice;
                DRIVER_SAFE_CALL(cuDeviceGet(&cuDevice, 0));

                // 2. Detect GPU architecture.
                int major = 0, minor = 0;
                DRIVER_SAFE_CALL(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice));
                DRIVER_SAFE_CALL(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice));
                std::string arch_option = "--gpu-architecture=compute_" + std::to_string(major) + std::to_string(minor);

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
                ptx.resize(ptxSize);
                NVRTC_SAFE_CALL(nvrtcGetPTX(prog, ptx.data()));
                NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog));

                // 5. Save the PTX code to disk.
                std::ofstream outFile(ptx_filename, std::ios::binary);
                if (!outFile) {
                    std::cerr << "Error opening PTX file for writing: " << ptx_filename << std::endl;
                    exit(EXIT_FAILURE);
                }
                outFile.write(ptx.data(), ptx.size());
                outFile.close();
                std::cout << "Saved PTX to " << ptx_filename << std::endl;
            }

            // 6. Use the current context if available.
            CUcontext currentContext = nullptr;
            DRIVER_SAFE_CALL(cuCtxGetCurrent(&currentContext));
            if (currentContext == nullptr) {
                // No current context exists; create one.
                CUdevice cuDevice;
                DRIVER_SAFE_CALL(cuDeviceGet(&cuDevice, 0));
                DRIVER_SAFE_CALL(cuCtxCreate(&kf->context, 0, cuDevice));
                kf->owns_context = true;
            } else {
                kf->context = currentContext;
                kf->owns_context = false;
            }

            // 7. Load the PTX module.
            DRIVER_SAFE_CALL(cuModuleLoadDataEx(&kf->module, ptx.data(), 0, nullptr, nullptr));

            // 8. Get the kernel function handle.
            DRIVER_SAFE_CALL(cuModuleGetFunction(&kf->function, kf->module, kernel_name));

            return kf;
        }

        // Member function to launch the kernel.
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

    // define the kernel function pointer type
    #define KernelFunctionPtr std::shared_ptr<KernelFunction>
} // namespace CPRRTC
