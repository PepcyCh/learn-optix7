#include <fstream>
#include <numbers>
#include <random>
#include <unordered_map>
#include <vector>

#include "glad/glad.h"
#include "cuda_runtime.h"
#include "optix_stubs.h"

#define PCM_GL_MATRIX
#include "../defines.h"
#include "check_macros.h"
#include "cuda_buffer.h"
#include "gbuffer.h"
#include "gl_utils.h"
#include "launch_params.h"
#include "lights.h"
#include "renderer_base.h"
#include "scene.h"

namespace numbers = std::numbers;
namespace ranges = std::ranges;

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
    void *data;
};

struct MeshBuffer {
    CudaBuffer position_buffer;
    CudaBuffer index_buffer;
};

struct MeshDataGl {
    GLuint vao;
    GLuint base_color_tex;
    int indices_count;
    pcm::Vec4 base_color;
};

class OptixGBuffer : public OptixApp {
public:
    OptixGBuffer(int width, int height, std::string_view &&title) : OptixApp(width, height, std::move(title)) {}

    bool Initialize() override {
        if (!OptixApp::Initialize()) {
            return false;
        }

        if (GlUtils::CheckExtension("GL_ARB_clip_control")) {
            use_reversed_z_ = true;
            glClipControl(GL_LOWER_LEFT, GL_ZERO_TO_ONE);
            glDepthFunc(GL_GREATER);
            glClearDepth(0.0);
        } else {
            use_reversed_z_ = false;
            glDepthFunc(GL_LESS);
            glClearDepth(1.0);
        }

        BuildScene();
        BuildSceneGl();
        BuildLights();
        CreateModule();
        CreatePrograms();
        CreatePipeline();
        BuildSbt();
        InitGBuffer();

        prev_color_buffer_.Init(color_buffer_width_, color_buffer_height_, 4 * sizeof(float), GL_RGBA, GL_FLOAT);
        
        UpdateCamera();
        launch_params_.frame.width = color_buffer_width_;
        launch_params_.frame.height = color_buffer_height_;
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
                mesh.data.positions.size() * sizeof(pcm::Vec3)
            );
            data.index_buffer.AllocAndUpload(
                mesh.data.indices.data(),
                mesh.data.indices.size() * sizeof(uint32_t)
            );

            meshes_buffers_.emplace_back(data);
        }
    }

    void BuildSceneGl() {
        std::vector<GLuint> textures;
        textures.reserve(scene_.textures.size());
        for (const Texture &tex : scene_.textures) {
            GLuint tex_id;
            glGenTextures(1, &tex_id);

            glBindTexture(GL_TEXTURE_2D, tex_id);
            GLenum internal_format;
            GLenum channel_format;
            GLenum channel_type;
            if (tex.component == 3) {
                channel_format = GL_RGB;
                if (tex.pixel_stride == 3) {
                    internal_format = GL_RGB;
                    channel_type = GL_UNSIGNED_BYTE;
                } else if (tex.pixel_stride == 12) {
                    internal_format = GL_RGB32F;
                    channel_type = GL_FLOAT;
                }
            } else if (tex.component == 4) {
                channel_format = GL_RGBA;
                if (tex.pixel_stride == 4) {
                    internal_format = GL_RGBA;
                    channel_type = GL_UNSIGNED_BYTE;
                } else if (tex.pixel_stride == 16) {
                    internal_format = GL_RGBA32F;
                    channel_type = GL_FLOAT;
                }
            }
            glTexImage2D(GL_TEXTURE_2D, 0, internal_format, tex.width, tex.height, 0,
                channel_format, channel_type, tex.data.data());
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

            textures.push_back(tex_id);
        }

        meshes_data_gl_.reserve(scene_.meshes.size());
        for (const Mesh &mesh : scene_.meshes) {
            MeshDataGl data{};
            glGenVertexArrays(1, &data.vao);
            glBindVertexArray(data.vao);

            GLuint pos_vb;
            glGenBuffers(1, &pos_vb);
            glBindBuffer(GL_ARRAY_BUFFER, pos_vb);
            glBufferData(GL_ARRAY_BUFFER, mesh.data.positions.size() * sizeof(pcm::Vec3),
                mesh.data.positions.data(), GL_STATIC_DRAW);
            glEnableVertexAttribArray(0);
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(pcm::Vec3), nullptr);

            GLuint norm_vb;
            glGenBuffers(1, &norm_vb);
            glBindBuffer(GL_ARRAY_BUFFER, norm_vb);
            glBufferData(GL_ARRAY_BUFFER, mesh.data.normals.size() * sizeof(pcm::Vec3),
                mesh.data.normals.data(), GL_STATIC_DRAW);
            glEnableVertexAttribArray(1);
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(pcm::Vec3), nullptr);

            GLuint uv_vb;
            glGenBuffers(1, &uv_vb);
            glBindBuffer(GL_ARRAY_BUFFER, uv_vb);
            glBufferData(GL_ARRAY_BUFFER, mesh.data.uvs.size() * sizeof(pcm::Vec2),
                mesh.data.uvs.data(), GL_STATIC_DRAW);
            glEnableVertexAttribArray(2);
            glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(pcm::Vec2), nullptr);

            GLuint ib;
            glGenBuffers(1, &ib);
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ib);
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, mesh.data.indices.size() * sizeof(uint32_t),
                mesh.data.indices.data(), GL_STATIC_DRAW);
            data.indices_count = mesh.data.indices.size();

            glBindVertexArray(0);
            glDeleteBuffers(1, &pos_vb);
            glDeleteBuffers(1, &norm_vb);
            glDeleteBuffers(1, &uv_vb);
            glDeleteBuffers(1, &ib);

            if (mesh.base_color_map_index >= 0) {
                data.base_color = pcm::Vec4(mesh.base_color, 1.0f);
                data.base_color_tex = textures[mesh.base_color_map_index];
            } else {
                data.base_color = pcm::Vec4(mesh.base_color, 0.0f);
            }

            meshes_data_gl_.emplace_back(data);
        }
    }

    void BuildLights() {
        lights_.AddMesh(
            GeometryUtils::Cube(25.0f, 2.0f, 25.0f),
            pcm::Vec3(10000.0f, 10000.0f, 10000.0f),
            pcm::Vec3(-907.108f, 2205.875f, -400.0267f)
        );

        lights_.BuildAliasTable();

        lights_vertex_buffer_.AllocAndUpload(lights_.vertices.data(), lights_.vertices.size() * sizeof(pcm::Vec3));
        lights_data_buffer_.AllocAndUpload(lights_.lights.data(), lights_.lights.size() * sizeof(LightData));

        launch_params_.light.light_count = lights_.lights.size();
        launch_params_.light.vertex = lights_vertex_buffer_.TypedPtr<pcm::Vec3>();
        launch_params_.light.data = lights_data_buffer_.TypedPtr<LightData>();
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

        std::ifstream ptx_fin(fmt::format("{}08-g-buffer/device_programs.ptx", PROJECT_SRC_DIR));
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

        // raygen
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

        // miss
        miss_pg_.resize(static_cast<size_t>(RayType::eCount));

        pg_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        pg_desc.miss.module = module_;
        pg_desc.miss.entryFunctionName = "__miss__Shadow";

        log_size = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(
            optix_context_,
            &pg_desc,
            1,
            &pg_options,
            log,
            &log_size,
            &miss_pg_[static_cast<size_t>(RayType::eShadow)]
        ));
        if (log_size > 1) {
            fmt::print("in CreatePrograms - miss - {}\n", log);
        }

        // hit group
        hitgroup_pg_.resize(static_cast<size_t>(RayType::eCount));

        pg_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        pg_desc.hitgroup.moduleCH = module_;
        pg_desc.hitgroup.entryFunctionNameCH = "__closesthit__Empty";
        pg_desc.hitgroup.moduleAH = module_;
        pg_desc.hitgroup.entryFunctionNameAH = "__anyhit__Empty";

        log_size = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(
            optix_context_,
            &pg_desc,
            1,
            &pg_options,
            log,
            &log_size,
            &hitgroup_pg_[static_cast<size_t>(RayType::eShadow)]
        ));
        if (log_size > 1) {
            fmt::print("in CreatePrograms - hitgroup - {}\n", log);
        }
    }

    void CreatePipeline() {
        std::vector<OptixProgramGroup> groups;
        groups.reserve(1 + miss_pg_.size() + hitgroup_pg_.size());
        groups.push_back(raygen_pg_);
        ranges::copy(miss_pg_, std::back_inserter(groups));
        ranges::copy(hitgroup_pg_, std::back_inserter(groups));

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

        std::vector<MissRecord> miss_records(miss_pg_.size());
        for (size_t i = 0; i < miss_pg_.size(); i++) {
            OPTIX_CHECK(optixSbtRecordPackHeader(miss_pg_[i], &miss_records[i]));
            miss_records[i].data = nullptr;
        }
        miss_records_buffer_.AllocAndUpload(miss_records.data(), miss_records.size() * sizeof(MissRecord));
        sbt_.missRecordBase = miss_records_buffer_.DevicePtr();
        sbt_.missRecordStrideInBytes = sizeof(MissRecord);
        sbt_.missRecordCount = miss_records.size();

        std::vector<HitgroupRecord> hitgroup_records;
        hitgroup_records.reserve(meshes_buffers_.size() * hitgroup_pg_.size());
        for (size_t i = 0; i < meshes_buffers_.size(); i++) {
            for (const OptixProgramGroup pg : hitgroup_pg_) {
                HitgroupRecord rec;
                OPTIX_CHECK(optixSbtRecordPackHeader(pg, &rec));
                rec.data = nullptr;
                hitgroup_records.emplace_back(rec);
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
            inputs[i].triangleArray.vertexStrideInBytes = sizeof(pcm::Vec3);
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

    void InitGBuffer() {
        const GLuint g_buffer_vs = GlUtils::LoadShader(
            fmt::format("{}/08-g-buffer/shaders/gbuffer.vert", PROJECT_SRC_DIR), GL_VERTEX_SHADER);
        const GLuint g_buffer_fs = GlUtils::LoadShader(
            fmt::format("{}/08-g-buffer/shaders/gbuffer.frag", PROJECT_SRC_DIR), GL_FRAGMENT_SHADER);
        g_buffer_shader_ = glCreateProgram();
        glAttachShader(g_buffer_shader_, g_buffer_vs);
        glAttachShader(g_buffer_shader_, g_buffer_fs);
        glLinkProgram(g_buffer_shader_);
        glDeleteShader(g_buffer_vs);
        glDeleteShader(g_buffer_fs);
        
        g_buffer_shader_locs_["proj"] = glGetUniformLocation(g_buffer_shader_, "u_proj");
        g_buffer_shader_locs_["view"] = glGetUniformLocation(g_buffer_shader_, "u_view");
        g_buffer_shader_locs_["base_color"] = glGetUniformLocation(g_buffer_shader_, "u_base_color");
        g_buffer_shader_locs_["base_color_tex"] = glGetUniformLocation(g_buffer_shader_, "u_base_color_tex");

        g_buffer_.Init(color_buffer_width_, color_buffer_height_);
    }

    void Render() override {
        if (launch_params_.frame.width == 0 || launch_params_.frame.height == 0) {
            return;
        }

        RenderGBuffer();

        screen_tex_ = g_buffer_.base_color_tex;

        launch_params_.frame.pos_buffer = g_buffer_.pos_buffer.TypedMap<pcm::Vec4>();
        launch_params_.frame.norm_buffer = g_buffer_.norm_buffer.TypedMap<pcm::Vec4>();
        launch_params_.frame.base_color_buffer = g_buffer_.base_color_buffer.TypedMap<pcm::Vec4>();
        
        launch_params_buffer_.Upload(&launch_params_, sizeof(launch_params_));
        
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
        
        color_buffer_.Unmap();
        prev_color_buffer_.Unmap();
        
        g_buffer_.pos_buffer.Unmap();
        g_buffer_.norm_buffer.Unmap();
        g_buffer_.base_color_buffer.Unmap();
    }

    void RenderGBuffer() {
        glBindFramebuffer(GL_FRAMEBUFFER, g_buffer_.fbo);

        glEnable(GL_DEPTH_TEST);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glUseProgram(g_buffer_shader_);

        glUniformMatrix4fv(g_buffer_shader_locs_["proj"], 1, GL_FALSE, reinterpret_cast<float *>(&camera_proj_));
        glUniformMatrix4fv(g_buffer_shader_locs_["view"], 1, GL_FALSE, reinterpret_cast<float *>(&camera_view_));

        for (MeshDataGl &data : meshes_data_gl_) {
            glBindVertexArray(data.vao);
            glUniform4fv(g_buffer_shader_locs_["base_color"], 1, reinterpret_cast<float *>(&data.base_color));
            glBindTexture(GL_TEXTURE_2D, data.base_color_tex);
            glUniform1i(g_buffer_shader_locs_["base_color_tex"], 0);
            glDrawElements(GL_TRIANGLES, data.indices_count, GL_UNSIGNED_INT, nullptr);
        }
        glBindVertexArray(0);

        glBindTexture(GL_TEXTURE_2D, 0);

        g_buffer_.pos_buffer.PackFrom(GL_COLOR_ATTACHMENT0);
        g_buffer_.norm_buffer.PackFrom(GL_COLOR_ATTACHMENT1);
        g_buffer_.base_color_buffer.PackFrom(GL_COLOR_ATTACHMENT2);

        glDisable(GL_DEPTH_TEST);

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    bool Resize(int width, int height) override {
        if (OptixApp::Resize(width, height)) {
            prev_color_buffer_.Resize(width, height, 4 * sizeof(float), GL_RGBA, GL_FLOAT);

            g_buffer_.Resize(width, height);

            launch_params_.frame.width = width;
            launch_params_.frame.height = height;
            UpdateCamera();
            return true;
        }
        return false;
    }

    void Update() override {
        launch_params_.frame.curr_time = frame_counter_.TotalTime() * 1000.0;

        accum_frame_count_ += 1;
        launch_params_.frame.accum_weight = 1.0f / accum_frame_count_;

        if (accum_frame_is_prev_) {
            launch_params_.frame.color_buffer = prev_color_buffer_.TypedMap<pcm::Vec4>();
            launch_params_.frame.prev_color_buffer = color_buffer_.TypedMap<pcm::Vec4>();
        } else {
            launch_params_.frame.color_buffer = color_buffer_.TypedMap<pcm::Vec4>();
            launch_params_.frame.prev_color_buffer = prev_color_buffer_.TypedMap<pcm::Vec4>();
        }
        accum_frame_is_prev_ = !accum_frame_is_prev_;
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
        const pcm::Vec3 position = pcm::Vec3(x, y, z) + camera_look_at_;

        const float aspect = static_cast<float>(color_buffer_width_) / color_buffer_height_;

        if (use_reversed_z_) {
            camera_proj_ = pcm::PerspectiveReverseZ(camera_fov_, aspect, 0.01f, 10000.0f, true);
        } else {
            camera_proj_ = pcm::Perspective(camera_fov_, aspect, 0.01f, 10000.0f, true);
        }
        camera_view_ = pcm::LookAt(position, camera_look_at_, pcm::Vec3::UnitY());

        accum_frame_count_ = 0;
    }

    OptixPipeline pipeline_;
    OptixPipelineCompileOptions pipeline_compile_options_ = {};
    OptixPipelineLinkOptions pipeline_link_options_ = {};

    OptixModule module_;
    OptixModuleCompileOptions module_compile_options_ = {};

    OptixProgramGroup raygen_pg_;
    std::vector<OptixProgramGroup> miss_pg_;
    std::vector<OptixProgramGroup> hitgroup_pg_;

    CudaBuffer raygen_records_buffer_;
    CudaBuffer miss_records_buffer_;
    CudaBuffer hitgroups_records_buffer_;

    OptixShaderBindingTable sbt_ = {};

    CudaBuffer accel_buffer_;

    Scene scene_;
    std::vector<MeshBuffer> meshes_buffers_;
    Lights lights_;
    CudaBuffer lights_vertex_buffer_;
    CudaBuffer lights_data_buffer_;

    std::vector<MeshDataGl> meshes_data_gl_;
    GBuffer g_buffer_;
    GLuint g_buffer_shader_;
    std::unordered_map<std::string, GLint> g_buffer_shader_locs_;

    PixelUnpackBuffer prev_color_buffer_;
    uint32_t accum_frame_count_ = 0;
    bool accum_frame_is_prev_ = false;

    pcm::Vec3 camera_look_at_ = pcm::Vec3(0.0f, -100.0f, 0.0f);
    float camera_radius_ = 2000.0f;
    float camera_theta_ = numbers::pi_v<float> * 0.25f;
    float camera_phi_ = numbers::pi_v<float> * 0.25f;
    float camera_fov_ = numbers::pi_v<float> * 0.25f;
    double last_mouse_x_ = 0.0;
    double last_mouse_y_ = 0.0;
    pcm::Mat4 camera_proj_;
    pcm::Mat4 camera_view_;
    bool use_reversed_z_;

    LaunchParams launch_params_ = {};
    CudaBuffer launch_params_buffer_;
};

int main() {
    try {
        OptixGBuffer app(1200, 900, "Learn OptiX 7");
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