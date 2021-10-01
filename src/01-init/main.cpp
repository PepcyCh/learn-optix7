#include <fstream>
#include <vector>

#include "cuda_runtime.h"
#include "optix_stubs.h"

#include "../defines.h"
#include "check_macros.h"
#include "cuda_buffer.h"
#include "launch_params.h"
#include "renderer_base.h"

struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) RaygenRecord {
    alignas(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    void *data;
};

struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) MissRecord {
    alignas(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    void *data;
};

struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) HitgroupRecord {
    alignas(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    int object_id;
};

class OptixInit : public OptixApp {
public:
    OptixInit(int width, int height, std::string_view &&title) : OptixApp(width, height, std::move(title)) {}

    bool Initialize() override {
        if (!OptixApp::Initialize()) {
            return false;
        }

        CreateModule();
        CreatePrograms();
        CreatePipeline();
        BuildSbt();

        launch_params_.frame_id = 0;
        launch_params_.frame_width = color_buffer_width_;
        launch_params_.frame_height = color_buffer_height_;
        launch_params_.color_buffer = reinterpret_cast<float *>(color_buffer_.DevicePtr());
        launch_params_buffer_.Alloc(sizeof(LaunchParams));

        return true;
    }

    bool Resize(int width, int height) override {
        if (OptixApp::Resize(width, height)) {
            launch_params_.frame_width = width;
            launch_params_.frame_height = height;
            launch_params_.color_buffer = reinterpret_cast<float *>(color_buffer_.DevicePtr());
            return true;
        }
        return false;
    }

private:
    void CreateModule() {
        module_compile_options_.maxRegisterCount = 50;
        module_compile_options_.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
        module_compile_options_.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_DEFAULT;

        pipeline_compile_options_ = {};
        pipeline_compile_options_.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
        pipeline_compile_options_.usesMotionBlur = false;
        pipeline_compile_options_.numPayloadValues = 2;
        pipeline_compile_options_.numAttributeValues = 2;
        pipeline_compile_options_.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
        pipeline_compile_options_.pipelineLaunchParamsVariableName = "optix_launch_params";

        pipeline_link_options_.maxTraceDepth = 2;

        std::ifstream ptx_fin(fmt::format("{}01-init/device_programs.ptx", PROJECT_SRC_DIR));
        const std::string ptx_str((std::istreambuf_iterator<char>(ptx_fin)), std::istreambuf_iterator<char>());

        char log[2048];
        size_t log_size = sizeof(log);
        OPTIX_CHECK(optixModuleCreateFromPTX(
            optix_context_,
            &module_compile_options_,
            &pipeline_compile_options_,
            ptx_str.c_str(),
            ptx_str.size(),
            log,
            &log_size,
            &module_
        ));
        if (log_size > 1) {
            fmt::print("in CreateModule - {}\n", log);
        }
    }

    void CreatePrograms() {
        char log[2048];
        size_t log_size;

        OptixProgramGroupOptions pg_options = {};
        OptixProgramGroupDesc pg_desc = {};
        pg_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        pg_desc.raygen.module = module_;
        pg_desc.raygen.entryFunctionName = "__raygen__RenderFrame";

        log_size = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(
            optix_context_,
            &pg_desc,
            1,
            &pg_options,
            log,
            &log_size,
            &raygen_pg_
        ));
        if (log_size > 1) {
            fmt::print("in CreatePrograms - raygen - {}\n", log);
        }

        pg_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        pg_desc.miss.module = module_;
        pg_desc.miss.entryFunctionName = "__miss__Radiance";

        log_size = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(
            optix_context_,
            &pg_desc,
            1,
            &pg_options,
            log,
            &log_size,
            &miss_pg_
        ));
        if (log_size > 1) {
            fmt::print("in CreatePrograms - miss - {}\n", log);
        }

        pg_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        pg_desc.hitgroup.moduleCH = module_;
        pg_desc.hitgroup.entryFunctionNameCH = "__closesthit__Radiance";
        pg_desc.hitgroup.moduleAH = module_;
        pg_desc.hitgroup.entryFunctionNameAH = "__anyhit__Radiance";

        log_size = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(
            optix_context_,
            &pg_desc,
            1,
            &pg_options,
            log,
            &log_size,
            &hitgroup_pg_
        ));
        if (log_size > 1) {
            fmt::print("in CreatePrograms - hitgroup - {}\n", log);
        }
    }

    void CreatePipeline() {
        const std::vector groups{raygen_pg_, miss_pg_, hitgroup_pg_};

        char log[2048];
        size_t log_size = sizeof(log);
        OPTIX_CHECK(optixPipelineCreate(
            optix_context_,
            &pipeline_compile_options_,
            &pipeline_link_options_,
            groups.data(),
            static_cast<int>(groups.size()),
            log,
            &log_size,
            &pipeline_
        ));
        if (log_size > 1) {
            fmt::print("in CreatePipeline - {}\n", log);
        }

        OPTIX_CHECK(optixPipelineSetStackSize(
            pipeline_,
            2 * 1024,
            2 * 1024,
            2 * 1024,
            1
        ));
    }

    void BuildSbt() {
        RaygenRecord raygen_record;
        OPTIX_CHECK(optixSbtRecordPackHeader(raygen_pg_, &raygen_record));
        raygen_record.data = nullptr;
        raygen_records_buffer_.AllocAndUpload(&raygen_record, sizeof(raygen_record));
        sbt.raygenRecord = raygen_records_buffer_.DevicePtr();

        MissRecord miss_record;
        OPTIX_CHECK(optixSbtRecordPackHeader(miss_pg_, &miss_record));
        miss_record.data = nullptr;
        miss_records_buffer_.AllocAndUpload(&miss_record, sizeof(miss_record));
        sbt.missRecordBase = miss_records_buffer_.DevicePtr();
        sbt.missRecordStrideInBytes = sizeof(MissRecord);
        sbt.missRecordCount = 1;

        HitgroupRecord hitgroup_record;
        OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_pg_, &hitgroup_record));
        hitgroup_record.object_id = 0;
        hitgroups_records_buffer_.AllocAndUpload(&hitgroup_record, sizeof(hitgroup_record));
        sbt.hitgroupRecordBase = hitgroups_records_buffer_.DevicePtr();
        sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
        sbt.hitgroupRecordCount = 1;
    }

    void Render() override {
        if (launch_params_.frame_width == 0 || launch_params_.frame_height == 0) {
            return;
        }

        launch_params_buffer_.Upload(&launch_params_, sizeof(launch_params_));
        launch_params_.frame_id += 1;

        OPTIX_CHECK(optixLaunch(
            pipeline_,
            cuda_stream_,
            launch_params_buffer_.DevicePtr(),
            launch_params_buffer_.Size(),
            &sbt,
            launch_params_.frame_width,
            launch_params_.frame_height,
            1
        ));

        CUDA_CHECK(cudaDeviceSynchronize());
    }

    OptixPipeline pipeline_;
    OptixPipelineCompileOptions pipeline_compile_options_ = {};
    OptixPipelineLinkOptions pipeline_link_options_ = {};

    OptixModule module_;
    OptixModuleCompileOptions module_compile_options_ = {};

    OptixProgramGroup raygen_pg_;
    OptixProgramGroup miss_pg_;
    OptixProgramGroup hitgroup_pg_;

    CudaBuffer raygen_records_buffer_;
    CudaBuffer miss_records_buffer_;
    CudaBuffer hitgroups_records_buffer_;

    OptixShaderBindingTable sbt = {};

    LaunchParams launch_params_ = {};
    CudaBuffer launch_params_buffer_;
};

int main() {
    try {
        OptixInit app(1200, 900, "Learn OptiX 7");
        if (!app.Initialize()) {
            fmt::print(stderr, "Failed to initialize\n");
            return -1;
        }

        app.MainLoop();
    } catch (const std::runtime_error &err) {
        fmt::print(stderr, "Runtime error: {}\n", err.what());
    }

    return 0;
}