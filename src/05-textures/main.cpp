#include <fstream>
#include <numbers>
#include <random>
#include <vector>

#include "cuda_runtime.h"
#include "optix_stubs.h"

#include "../defines.h"
#include "check_macros.h"
#include "cuda_buffer.h"
#include "cuda_texture.h"
#include "launch_params.h"
#include "renderer_base.h"
#include "scene.h"

namespace numbers = std::numbers;

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
    TriMeshData data;
};

struct MeshBuffer {
    CudaBuffer position_buffer;
    CudaBuffer normal_buffer;
    CudaBuffer uv_buffer;
    CudaBuffer index_buffer;
};

class OptixTextures : public OptixApp {
public:
    OptixTextures(int width, int height, std::string_view &&title) : OptixApp(width, height, std::move(title)) {}

    bool Initialize() override {
        if (!OptixApp::Initialize()) {
            return false;
        }

        BuildScene();
        CreateModule();
        CreatePrograms();
        CreatePipeline();
        BuildSbt();
        
        UpdateCamera();
        launch_params_.frame.id = 0;
        launch_params_.frame.width = color_buffer_width_;
        launch_params_.frame.height = color_buffer_height_;
        launch_params_.frame.color_buffer = color_buffer_.TypedPtr<Vec4>();
        launch_params_.traversable = BuildAccel();
        launch_params_buffer_.Alloc(sizeof(LaunchParams));

        return true;
    }
    
private:
    void BuildScene() {
        scene_ = Scene::LoadObj(fmt::format("{}models/sponza.obj", PROJECT_ROOT_DIR));

        for (const Mesh &mesh : scene_.meshes) {
            MeshBuffer data{};
            data.position_buffer.AllocAndUpload(
                mesh.data.positions.data(),
                mesh.data.positions.size() * sizeof(Vec3)
            );
            data.normal_buffer.AllocAndUpload(
                mesh.data.normals.data(),
                mesh.data.normals.size() * sizeof(Vec3)
            );
            data.uv_buffer.AllocAndUpload(
                mesh.data.uvs.data(),
                mesh.data.uvs.size() * sizeof(Vec2)
            );
            data.index_buffer.AllocAndUpload(
                mesh.data.indices.data(),
                mesh.data.indices.size() * sizeof(uint32_t)
            );

            meshes_buffers_.emplace_back(data);
        }

        scene_textures_.reserve(scene_.textures.size());
        for (const Texture &tex : scene_.textures) {
            scene_textures_.emplace_back(tex);
        }
    }

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

        std::ifstream ptx_fin(fmt::format("{}05-textures/device_programs.ptx", PROJECT_SRC_DIR));
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
        const std::vector groups{ raygen_pg_, miss_pg_, hitgroup_pg_ };

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
        sbt_.raygenRecord = raygen_records_buffer_.DevicePtr();

        MissRecord miss_record;
        OPTIX_CHECK(optixSbtRecordPackHeader(miss_pg_, &miss_record));
        miss_record.data = nullptr;
        miss_records_buffer_.AllocAndUpload(&miss_record, sizeof(miss_record));
        sbt_.missRecordBase = miss_records_buffer_.DevicePtr();
        sbt_.missRecordStrideInBytes = sizeof(MissRecord);
        sbt_.missRecordCount = 1;

        std::vector<HitgroupRecord> hitgroup_records(meshes_buffers_.size());
        for (size_t i = 0; i < meshes_buffers_.size(); i++) {
            OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_pg_, &hitgroup_records[i]));
            hitgroup_records[i].data.vertex = meshes_buffers_[i].position_buffer.TypedPtr<Vec3>();
            hitgroup_records[i].data.normal = meshes_buffers_[i].normal_buffer.TypedPtr<Vec3>();
            hitgroup_records[i].data.uv = meshes_buffers_[i].uv_buffer.TypedPtr<Vec2>();
            hitgroup_records[i].data.index = meshes_buffers_[i].index_buffer.TypedPtr<uint32_t>();

            hitgroup_records[i].data.base_color = scene_.meshes[i].base_color;
            if (scene_.meshes[i].base_color_map_index >= 0) {
                hitgroup_records[i].data.base_color_mapped = true;
                hitgroup_records[i].data.base_color_map =
                    scene_textures_[scene_.meshes[i].base_color_map_index].CudaObject();
            } else {
                hitgroup_records[i].data.base_color_mapped = false;
            }
        }
        hitgroups_records_buffer_.AllocAndUpload(
            hitgroup_records.data(),
            hitgroup_records.size() * sizeof(HitgroupRecord)
        );
        sbt_.hitgroupRecordBase = hitgroups_records_buffer_.DevicePtr();
        sbt_.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
        sbt_.hitgroupRecordCount = hitgroup_records.size();
    }

    OptixTraversableHandle BuildAccel() {
        OptixTraversableHandle accel_handle = 0;

        const uint32_t input_flags[] = { 0 };
        std::vector<CUdeviceptr> vertex_buffer_ptrs(meshes_buffers_.size());
        std::vector<OptixBuildInput> inputs(meshes_buffers_.size());
        for (size_t i = 0; i < meshes_buffers_.size(); i++) {
            vertex_buffer_ptrs[i] = meshes_buffers_[i].position_buffer.DevicePtr();

            inputs[i].type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
            inputs[i].triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
            inputs[i].triangleArray.vertexStrideInBytes = sizeof(Vec3);
            inputs[i].triangleArray.numVertices = scene_.meshes[i].data.positions.size();
            inputs[i].triangleArray.vertexBuffers = &vertex_buffer_ptrs[i];
            inputs[i].triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
            inputs[i].triangleArray.indexStrideInBytes = sizeof(uint32_t) * 3;
            inputs[i].triangleArray.numIndexTriplets = scene_.meshes[i].data.indices.size() / 3;
            inputs[i].triangleArray.indexBuffer = meshes_buffers_[i].index_buffer.DevicePtr();
            inputs[i].triangleArray.flags = input_flags;
            inputs[i].triangleArray.numSbtRecords = 1;
            inputs[i].triangleArray.sbtIndexOffsetBuffer = 0;
            inputs[i].triangleArray.sbtIndexOffsetSizeInBytes = 0;
            inputs[i].triangleArray.sbtIndexOffsetStrideInBytes = 0;
        }

        OptixAccelBuildOptions options = {};
        options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
        options.motionOptions.numKeys = 1;
        options.operation = OPTIX_BUILD_OPERATION_BUILD;

        OptixAccelBufferSizes buffer_sizes;
        OPTIX_CHECK(optixAccelComputeMemoryUsage(
            optix_context_,
            &options,
            inputs.data(),
            inputs.size(),
            &buffer_sizes
        ));

        CudaBuffer compacted_size_buffer;
        compacted_size_buffer.Alloc(sizeof(uint64_t));

        OptixAccelEmitDesc emit_desc = {};
        emit_desc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
        emit_desc.result = compacted_size_buffer.DevicePtr();

        CudaBuffer temp_buffer;
        temp_buffer.Alloc(buffer_sizes.tempSizeInBytes);
        CudaBuffer output_buffer;
        output_buffer.Alloc(buffer_sizes.outputSizeInBytes);
        OPTIX_CHECK(optixAccelBuild(
            optix_context_,
            nullptr,
            &options,
            inputs.data(),
            inputs.size(),
            temp_buffer.DevicePtr(),
            temp_buffer.Size(),
            output_buffer.DevicePtr(),
            output_buffer.Size(),
            &accel_handle,
            &emit_desc,
            1
        ));
        CUDA_CHECK(cudaDeviceSynchronize());

        uint64_t compacted_size;
        compacted_size_buffer.Download(&compacted_size, sizeof(uint64_t));

        accel_buffer_.Alloc(compacted_size);
        OPTIX_CHECK(optixAccelCompact(
            optix_context_,
            nullptr,
            accel_handle,
            accel_buffer_.DevicePtr(),
            accel_buffer_.Size(),
            &accel_handle
        ));
        CUDA_CHECK(cudaDeviceSynchronize());

        output_buffer.Free();
        temp_buffer.Free();
        compacted_size_buffer.Free();

        return accel_handle;
    }

    void Render() override {
        if (launch_params_.frame.width == 0 || launch_params_.frame.height == 0) {
            return;
        }

        launch_params_buffer_.Upload(&launch_params_, sizeof(launch_params_));
        launch_params_.frame.id += 1;

        OPTIX_CHECK(optixLaunch(
            pipeline_,
            cuda_stream_,
            launch_params_buffer_.DevicePtr(),
            launch_params_buffer_.Size(),
            &sbt_,
            launch_params_.frame.width,
            launch_params_.frame.height,
            1
        ));

        CUDA_CHECK(cudaDeviceSynchronize());
    }

    bool Resize(int width, int height) override {
        if (OptixApp::Resize(width, height)) {
            launch_params_.frame.width = width;
            launch_params_.frame.height = height;
            launch_params_.frame.color_buffer = color_buffer_.TypedPtr<Vec4>();
            UpdateCamera();
            return true;
        }
        return false;
    }

    void OnMouse(double x, double y, uint8_t state) override {
        if (state & 1) {
            const float dx = 0.25 * (x - last_mouse_x_) * numbers::pi / 180.0;
            const float dy = 0.25 * (y - last_mouse_y_) * numbers::pi / 180.0;
            camera_theta_ -= dx;
            camera_phi_ += dy;
            camera_phi_ = std::clamp(camera_phi_, 0.1f, numbers::pi_v<float> - 0.1f);
        } else if (state & 2) {
            const float dx = 0.005 * (x - last_mouse_x_);
            const float dy = 0.005 * (y - last_mouse_y_);
            camera_radius_ += (dx - dy) * 50.0f;
            camera_radius_ = std::clamp(camera_radius_, 5.0f, 15000.0f);
        }
        last_mouse_x_ = x;
        last_mouse_y_ = y;

        if (state & 3) {
            UpdateCamera();
        }
    }

    void UpdateCamera() {
        const float x = camera_radius_ * std::sin(camera_phi_) * std::cos(camera_theta_);
        const float y = camera_radius_ * std::cos(camera_phi_);
        const float z = camera_radius_ * std::sin(camera_phi_) * std::sin(camera_theta_);
        launch_params_.camera.position = Vec3(x, y, z) + camera_look_at_;
        launch_params_.camera.direction = (-Vec3(x, y, z)).Normalize();

        const float twice_tan_half_fov = 2.0f * std::tan(camera_fov_ * 0.5f);
        const float aspect = static_cast<float>(color_buffer_width_) / color_buffer_height_;
        launch_params_.camera.right =
            twice_tan_half_fov * aspect * launch_params_.camera.direction.Cross(Vec3::kY).Normalize();
        launch_params_.camera.up =
            twice_tan_half_fov * launch_params_.camera.right.Cross(launch_params_.camera.direction).Normalize();
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

    OptixShaderBindingTable sbt_ = {};

    CudaBuffer accel_buffer_;

    Scene scene_;
    std::vector<MeshBuffer> meshes_buffers_;
    std::vector<CudaTexture> scene_textures_;

    Vec3 camera_look_at_ = Vec3(0.0f, -100.0f, 0.0f);
    float camera_radius_ = 2000.0f;
    float camera_theta_ = numbers::pi_v<float> * 0.25f;
    float camera_phi_ = numbers::pi_v<float> * 0.25f;
    float camera_fov_ = numbers::pi_v<float> * 0.25f;
    double last_mouse_x_ = 0.0;
    double last_mouse_y_ = 0.0;

    LaunchParams launch_params_ = {};
    CudaBuffer launch_params_buffer_;
};

int main() {
    try {
        OptixTextures app(1200, 900, "Learn OptiX 7");
        if (!app.Initialize()) {
            fmt::print(stderr, "Failed to initialize\n");
            return -1;
        }

        app.MainLoop();
    }
    catch (const std::runtime_error &err) {
        fmt::print(stderr, "Runtime error: {}\n", err.what());
    }

    return 0;
}