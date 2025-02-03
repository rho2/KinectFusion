/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */


#define VMA_IMPLEMENTATION
#define IMGUI_DEFINE_MATH_OPERATORS

#include "VirtualSensor.h"

#include "backends/imgui_impl_vulkan.h"
#include "common/vk_context.hpp"
#include "glm/gtc/noise.hpp"
#include "nvh/primitives.hpp"
#include "nvvk/debug_util_vk.hpp"
#include "nvvk/descriptorsets_vk.hpp"
#include "nvvk/dynamicrendering_vk.hpp"
#include "nvvk/extensions_vk.hpp"
#include "nvvk/images_vk.hpp"
#include "nvvk/pipeline_vk.hpp"
#include "nvvk/shaders_vk.hpp"
#include "nvvkhl/alloc_vma.hpp"
#include "nvvkhl/element_benchmark_parameters.hpp"
#include "nvvkhl/element_camera.hpp"
#include "nvvkhl/element_gui.hpp"
#include "nvvkhl/gbuffer.hpp"
#include "nvvkhl/shaders/dh_comp.h"
#include "nvvk/renderpasses_vk.hpp"

namespace DH {
using namespace glm;
#include "shaders/device_host.h"  // Shared between host and device
}  // namespace DH

#include "_autogen/raster_slang.h"
#include "_autogen/perlin_slang.h"
const auto& comp_shd = std::vector<uint32_t>{std::begin(perlinSlang), std::end(perlinSlang)};


#include "imgui/imgui_helper.h"
#include "imgui/imgui_camera_widget.h"

class Texture3dSample : public nvvkhl::IAppElement
{
  struct Settings
  {
    uint32_t             powerOfTwoSize = 8;
    bool                 useGpu         = false;
    bool                 renderNormals  = false;
    VkFilter             magFilter      = VK_FILTER_NEAREST;
    VkSamplerAddressMode addressMode    = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    DH::PerlinSettings   perlin         = DH::PerlinSettings();
    int                  headlight      = 1;
    glm::vec3            toLight        = {1.F, 1.F, 1.F};
    int                  steps          = 500;
    glm::vec4            surfaceColor   = {0.8F, 0.8F, 0.8F, 1.0F};
    uint32_t             getSize() { return 1 << powerOfTwoSize; }
    uint32_t             getTotalSize() { return getSize() * getSize() * getSize(); }
  };

public:
  Texture3dSample()           = default;
  ~Texture3dSample() override = default;

  // Implementation of nvvk::IApplication interface
  void onAttach(nvvkhl::Application* app) override
  {
    if (!sensor.Init("../Data/rgbd_dataset_freiburg1_xyz/")) {
      std::cout << "Failed to initialize the sensor!\nCheck file path!" << std::endl;
      exit(1);
    }

    m_app    = app;
    m_device = m_app->getDevice();

    // Create the Vulkan allocator (VMA)
    m_alloc       = std::make_unique<nvvkhl::AllocVma>(VmaAllocatorCreateInfo{
              .flags          = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
              .physicalDevice = app->getPhysicalDevice(),
              .device         = app->getDevice(),
              .instance       = app->getInstance(),
    });  // Allocator
    m_dutil       = std::make_unique<nvvk::DebugUtil>(m_device);
    m_dsetCompute = std::make_unique<nvvk::DescriptorSetContainer>(m_device);
    m_dsetRaster  = std::make_unique<nvvk::DescriptorSetContainer>(m_device);
    m_depthFormat = nvvk::findDepthFormat(app->getPhysicalDevice());

    voxel_grid.resize(m_settings.getTotalSize());
    std::fill(voxel_grid.begin(), voxel_grid.end(), 1.0f);

    voxel_grid_weights.resize(m_settings.getTotalSize());
    std::fill(voxel_grid_weights.begin(), voxel_grid_weights.end(), 0.0f);

    createVkBuffers();
    createComputePipeline();
    createTextureBuffers();
    createTexture();
    createGraphicPipeline();

    // Setting the default camera
    CameraManip.setClipPlanes({0.01F, 100.0F});
    CameraManip.setLookat({-0.5F, 0.5F, 2.0F}, {0.0F, 0.0F, 0.0F}, {0.0F, 1.0F, 0.0F});
  };

  void onDetach() override
  {
    vkDeviceWaitIdle(m_device);
    destroyResources();
  };


  void onUIRender() override {
    namespace PE      = ImGuiH::PropertyEditor;
    auto& s           = m_settings;
    bool  redoTexture = false;

    // Settings
    ImGui::Begin("Settings");

    ImGuiH::CameraWidget();

    ImGui::Text("Ray Marching");
    PE::begin();
    PE::entry(
        "Steps", [&] { return ImGui::SliderInt("##2", (int*)&m_settings.steps, 1, 500); }, "Number of maximum steps.");
    PE::end();

    PE::begin();
    PE::entry(
        "Render normals", [&] { return ImGui::Checkbox("##4", &s.renderNormals); },
        "Render normals instead of pure image");
    PE::end();

    PE::begin();
    if (PE::Button("NextFrame", {-1, 20})) {
      processFrame();
      redoTexture = TRUE;
    }
    PE::end();

    if(redoTexture)
    {
      vkDeviceWaitIdle(m_device);
      createTexture();
    }

    ImGui::TextDisabled("%d FPS / %.3fms", static_cast<int>(ImGui::GetIO().Framerate), 1000.F / ImGui::GetIO().Framerate);
    ImGui::End();

    // Using viewport Window
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0F, 0.0F));
    ImGui::Begin("Viewport");
    if(m_texture.image != nullptr)
    {
      ImGui::Image(m_gBuffers->getDescriptorSet(), ImGui::GetContentRegionAvail());
    }

    ImGui::End();
    ImGui::PopStyleVar();
  }

  void onRender(VkCommandBuffer cmd) override
  {
    const nvvk::DebugUtil::ScopedCmdLabel sdbg = m_dutil->DBG_SCOPE(cmd);

    if(m_dirty)
    {
      setData(cmd);
    }

    const float aspect_ratio = m_gBuffers->getAspectRatio();
    glm::vec3   eye;
    glm::vec3   center;
    glm::vec3   up;
    CameraManip.getLookat(eye, center, up);

    // Update Frame buffer uniform buffer
    DH::FrameInfo    finfo{};
    const glm::vec2& clip = CameraManip.getClipPlanes();
    finfo.view            = CameraManip.getMatrix();
    finfo.proj            = glm::perspectiveRH_ZO(glm::radians(CameraManip.getFov()), aspect_ratio, clip.x, clip.y);
    finfo.proj[1][1] *= -1;
    finfo.camPos    = eye;
    finfo.headlight = m_settings.headlight;
    finfo.toLight   = m_settings.toLight;
    vkCmdUpdateBuffer(cmd, m_frameInfo.buffer, 0, sizeof(DH::FrameInfo), &finfo);

    // Drawing the primitives in a G-Buffer
    nvvk::createRenderingInfo r_info({{0, 0}, m_gBuffers->getSize()}, {m_gBuffers->getColorImageView()},
                                     m_gBuffers->getDepthImageView(), VK_ATTACHMENT_LOAD_OP_CLEAR,
                                     VK_ATTACHMENT_LOAD_OP_CLEAR, m_clearColor);
    r_info.pStencilAttachment = nullptr;

    nvvk::cmdBarrierImageLayout(cmd, m_gBuffers->getColorImage(), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    vkCmdBeginRendering(cmd, &r_info);
    {
      const VkDeviceSize offsets{0};
      m_app->setViewport(cmd);
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphicsPipeline);
      vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_dsetRaster->getPipeLayout(), 0,
                                static_cast<uint32_t>(m_dsetRastWrites.size()), m_dsetRastWrites.data());

      // Push constant information
      DH::PushConstant pushConstant{};
      pushConstant.steps     = m_settings.steps;
      pushConstant.color     = m_settings.surfaceColor;
      pushConstant.transfo   = glm::mat4(1);  // Identity
      pushConstant.size      = m_settings.getSize();
      pushConstant.render_normals = m_settings.renderNormals;
      vkCmdPushConstants(cmd, m_dsetRaster->getPipeLayout(), VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                         0, sizeof(DH::PushConstant), &pushConstant);

      vkCmdBindVertexBuffers(cmd, 0, 1, &m_vertices.buffer, &offsets);
      vkCmdBindIndexBuffer(cmd, m_indices.buffer, 0, VK_INDEX_TYPE_UINT32);
      int32_t num_indices = 36;
      vkCmdDrawIndexed(cmd, num_indices, 1, 0, 0, 0);
    }
    vkCmdEndRendering(cmd);
    nvvk::cmdBarrierImageLayout(cmd, m_gBuffers->getColorImage(), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);
  }

  void onResize(uint32_t width, uint32_t height) override
  {
    m_gBuffers = std::make_unique<nvvkhl::GBuffer>(m_device, m_alloc.get(), VkExtent2D{width, height}, m_colorFormat, m_depthFormat);
  }

private:
  void processFrame()
  {
    nvh::ScopedTimer st(__FUNCTION__);
    sensor.ProcessNextFrame();
  }

  void createTextureBuffers() {
    assert(!m_texture.image);

    uint32_t realSize = m_settings.getSize();

    VkFormat imgFormat = VK_FORMAT_R32_SFLOAT;

    VkImageCreateInfo create_info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    create_info.imageType     = VK_IMAGE_TYPE_3D;
    create_info.format        = imgFormat;
    create_info.mipLevels     = 1;
    create_info.arrayLayers   = 1;
    create_info.samples       = VK_SAMPLE_COUNT_1_BIT;
    create_info.extent.width  = realSize;
    create_info.extent.height = realSize;
    create_info.extent.depth  = realSize;
    create_info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                        | VK_BUFFER_USAGE_TRANSFER_DST_BIT;


    VkSamplerCreateInfo sampler_info{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sampler_info.addressModeU = m_settings.addressMode;
    sampler_info.addressModeV = m_settings.addressMode;
    sampler_info.addressModeW = m_settings.addressMode;
    sampler_info.magFilter    = m_settings.magFilter;
    nvvk::Image texImage      = m_alloc->createImage(create_info);

    VkImageViewCreateInfo view_info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    view_info.pNext                           = nullptr;
    view_info.image                           = texImage.image;
    view_info.format                          = imgFormat;
    view_info.viewType                        = VK_IMAGE_VIEW_TYPE_3D;
    view_info.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    view_info.subresourceRange.baseMipLevel   = 0;
    view_info.subresourceRange.levelCount     = VK_REMAINING_MIP_LEVELS;
    view_info.subresourceRange.baseArrayLayer = 0;
    view_info.subresourceRange.layerCount     = VK_REMAINING_ARRAY_LAYERS;

    m_texture                        = m_alloc->createTexture(texImage, view_info, sampler_info);
    m_texture.descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;


    nvvk::Image texImageWeights      = m_alloc->createImage(create_info);
    VkImageViewCreateInfo view_info_weights{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    view_info_weights.pNext                           = nullptr;
    view_info_weights.image                           = texImageWeights.image;
    view_info_weights.format                          = imgFormat;
    view_info_weights.viewType                        = VK_IMAGE_VIEW_TYPE_3D;
    view_info_weights.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    view_info_weights.subresourceRange.baseMipLevel   = 0;
    view_info_weights.subresourceRange.levelCount     = VK_REMAINING_MIP_LEVELS;
    view_info_weights.subresourceRange.baseArrayLayer = 0;
    view_info_weights.subresourceRange.layerCount     = VK_REMAINING_ARRAY_LAYERS;

    m_texture_weights                        = m_alloc->createTexture(texImageWeights, view_info_weights, sampler_info);
    m_texture_weights.descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkImageCreateInfo create_info_depth_map{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    create_info_depth_map.imageType     = VK_IMAGE_TYPE_2D;
    create_info_depth_map.format        = VK_FORMAT_R32_SFLOAT;
    create_info_depth_map.mipLevels     = 1;
    create_info_depth_map.arrayLayers   = 1;
    create_info_depth_map.samples       = VK_SAMPLE_COUNT_1_BIT;
    create_info_depth_map.extent.width  = 640;
    create_info_depth_map.extent.height = 480;
    create_info_depth_map.extent.depth  = 1;
    create_info_depth_map.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                        | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    VkSamplerCreateInfo depth_sampler_info{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    depth_sampler_info.addressModeU = m_settings.addressMode;
    depth_sampler_info.addressModeV = m_settings.addressMode;
    depth_sampler_info.addressModeW = m_settings.addressMode;
    depth_sampler_info.magFilter    = VK_FILTER_LINEAR;
    depth_sampler_info.minFilter    = VK_FILTER_LINEAR;
    nvvk::Image texImageDepth      = m_alloc->createImage(create_info_depth_map);

    VkImageViewCreateInfo depth_view_info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    depth_view_info.pNext                           = nullptr;
    depth_view_info.image                           = texImageDepth.image;
    depth_view_info.format                          = VK_FORMAT_R32_SFLOAT;
    depth_view_info.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
    depth_view_info.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    depth_view_info.subresourceRange.baseMipLevel   = 0;
    depth_view_info.subresourceRange.levelCount     = VK_REMAINING_MIP_LEVELS;
    depth_view_info.subresourceRange.baseArrayLayer = 0;
    depth_view_info.subresourceRange.layerCount     = VK_REMAINING_ARRAY_LAYERS;

    m_depth_texture                        = m_alloc->createTexture(texImageDepth, depth_view_info, depth_sampler_info);
    m_depth_texture.descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
  }

  void createTexture()
  {
    nvh::ScopedTimer st(__FUNCTION__);

    VkCommandBuffer cmd = m_app->createTempCmdBuffer();


    // The descriptors
    m_dsetCompWrites.clear();
    m_dsetCompWrites.emplace_back(m_dsetCompute->makeWrite(0, 0, &m_texture.descriptor));
    m_dsetCompWrites.emplace_back(m_dsetCompute->makeWrite(0, 1, &m_texture_weights.descriptor));
    m_dsetCompWrites.emplace_back(m_dsetCompute->makeWrite(0, 2, &m_depth_texture.descriptor));

    nvvk::cmdBarrierImageLayout(cmd, m_texture.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    nvvk::cmdBarrierImageLayout(cmd, m_texture_weights.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    setData(cmd);

    m_app->submitAndWaitTempCmdBuffer(cmd);

    // Debugging information
    m_dutil->setObjectName(m_texture.image, "Image");
    m_dutil->setObjectName(m_texture.descriptor.sampler, "Sampler");
    m_descriptorSet = ImGui_ImplVulkan_AddTexture(m_texture.descriptor.sampler, m_texture.descriptor.imageView,
                                                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  }

  void fillPerlinImage(std::vector<float>& imageData)
  {
    nvh::ScopedTimer st(__FUNCTION__);
    uint32_t realSize = m_settings.getSize();

    if (sensor.m_currentIdx == -1)
      return;

    float* depthMap = sensor.GetDepth();
    Matrix3f depthIntrinsics = sensor.GetDepthIntrinsics();

    Matrix4f foo = sensor.GetTrajectory() * sensor.GetDepthExtrinsics();

    for (unsigned int k = 0; k < 256; k++) {
      for (unsigned int j = 0; j < 256; j++) {
        for (unsigned int i = 0; i < 256; i++) {
          Vector4f coord = Vector4f(
            (i-128)/100.,
            (k-128)/100.,
            (j-128)/100.,
            1.0f
            );

          Vector3f camera_coord = (foo * coord).head(3);

          Vector3f pix_coord = depthIntrinsics * camera_coord;

          if (pix_coord.z() <= 0.0) {
            continue;
          }

          Vector2i pix = Vector2i(pix_coord.x() / pix_coord.z(), pix_coord.y() / pix_coord.z());

          if (pix.x() < 0 || pix.x() >= 640 || pix.y() < 0 || pix.y() >= 480) {
            continue;
          }

          float d = depthMap[pix.y() * 640 + pix.x()];
          if (d == MINF) {
            continue;
          }

          float sdf = d - camera_coord.z();

          float trunc_distance = 0.1f;

          if (sdf < -trunc_distance) {
            continue;
          }

          sdf = std::min(1.0f, fabsf(sdf) / trunc_distance) * copysignf(1.0f, sdf);

          float old_weight = voxel_grid_weights[k * realSize * realSize + j * realSize + i];
          float new_weight = old_weight + 1;

          float old_value = imageData[k * realSize * realSize + j * realSize + i];
          float new_value = (old_value * old_weight + sdf) / new_weight;

          imageData[k * realSize * realSize + j * realSize + i] = new_value;
          voxel_grid_weights[k * realSize * realSize + j * realSize + i] = new_weight;
        }
      }
    }
  }

  void setData(VkCommandBuffer cmd)
  {
    const nvvk::DebugUtil::ScopedCmdLabel sdbg = m_dutil->DBG_SCOPE(cmd);
    assert(m_texture.image);

    uint32_t realSize = m_settings.getSize();
    if(m_settings.useGpu)
    {
      runCompute(cmd, {realSize, realSize, realSize});
    }
    else
    {
      fillPerlinImage(voxel_grid);

      const VkOffset3D               offset{0};
      const VkImageSubresourceLayers subresource{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
      const VkExtent3D               extent{realSize, realSize, realSize};
      nvvk::cmdBarrierImageLayout(cmd, m_texture.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

      nvvk::StagingMemoryManager* staging = m_alloc->getStaging();
      staging->cmdToImage(cmd, m_texture.image, offset, extent, subresource, voxel_grid.size() * sizeof(float), voxel_grid.data());

      nvvk::cmdBarrierImageLayout(cmd, m_texture.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);
    }
    m_dirty = false;
  }

  void createComputePipeline()
  {
    nvvk::DebugUtil dbg(m_device);

    auto& d = m_dsetCompute;
    d->addBinding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    d->addBinding(1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    d->addBinding(2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    d->initLayout(VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR);

    m_dutil->DBG_NAME(d->getLayout());

    VkPushConstantRange push_constant_ranges = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(DH::PerlinSettings)};

    d->initPipeLayout(1, &push_constant_ranges);
    m_dutil->DBG_NAME(d->getPipeLayout());

    VkPipelineShaderStageCreateInfo stage_info{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage_info.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    stage_info.module = nvvk::createShaderModule(m_device, comp_shd);
    stage_info.pName  = "computeMain";

    VkComputePipelineCreateInfo comp_info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    comp_info.layout = d->getPipeLayout();
    comp_info.stage  = stage_info;

    vkCreateComputePipelines(m_device, {}, 1, &comp_info, nullptr, &m_computePipeline);
    m_dutil->DBG_NAME(m_computePipeline);

    // Clean up
    vkDestroyShaderModule(m_device, comp_info.stage.module, nullptr);
  }

  void runCompute(VkCommandBuffer cmd, const VkExtent3D& size)
  {
    {
      {
        const VkOffset3D               offset{0};
        const VkImageSubresourceLayers subresource{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        const VkExtent3D               extent{640, 480, 1};
        nvvk::cmdBarrierImageLayout(cmd, m_depth_texture.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

        nvvk::StagingMemoryManager* staging = m_alloc->getStaging();
        staging->cmdToImage(cmd, m_depth_texture.image, offset, extent, subresource, 640 * 480 * sizeof(float), sensor.m_depthFrame);

        nvvk::cmdBarrierImageLayout(cmd, m_depth_texture.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);
      }
    }

    if (sensor.m_currentIdx < 0) return;

    DH::PerlinSettings perlin   = m_settings.perlin;

    Matrix4f foo = sensor.GetTrajectory() * sensor.GetDepthExtrinsics();
    for (int i = 0; i < 4; ++i) {
      perlin.transform[i].x = foo.row(i).x();
      perlin.transform[i].y = foo.row(i).y();
      perlin.transform[i].z = foo.row(i).z();
      perlin.transform[i].w = foo.row(i).w();
    }

    vkCmdPushConstants(cmd, m_dsetCompute->getPipeLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(DH::PerlinSettings), &perlin);
    vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_dsetCompute->getPipeLayout(), 0,
                              static_cast<uint32_t>(m_dsetCompWrites.size()), m_dsetCompWrites.data());
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_computePipeline);
    vkCmdDispatch(cmd, 256, 256, 256);
    gpu_ini = true;
  }


  void destroyResources()
  {
    m_dsetCompute->deinit();
    m_dsetRaster->deinit();
    vkDestroyPipeline(m_device, m_computePipeline, nullptr);
    vkDestroyPipeline(m_device, m_graphicsPipeline, nullptr);

    m_alloc->destroy(m_vertices);
    m_alloc->destroy(m_indices);
    m_alloc->destroy(m_frameInfo);
    m_alloc->destroy(m_texture);
    m_alloc->destroy(m_texture_weights);
    m_alloc->destroy(m_depth_texture);
  }

  void createVkBuffers()
  {
    VkCommandBuffer cmd = m_app->createTempCmdBuffer();

    // Creating the Cube on the GPU
    nvh::PrimitiveMesh mesh = nvh::createCube();
    m_vertices              = m_alloc->createBuffer(cmd, mesh.vertices, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    m_indices               = m_alloc->createBuffer(cmd, mesh.triangles, VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
    m_dutil->DBG_NAME(m_vertices.buffer);
    m_dutil->DBG_NAME(m_indices.buffer);

    // Frame information: camera matrix
    m_frameInfo = m_alloc->createBuffer(sizeof(DH::FrameInfo), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    m_dutil->DBG_NAME(m_frameInfo.buffer);

    m_app->submitAndWaitTempCmdBuffer(cmd);
  }

  void createGraphicPipeline()
  {
    m_dsetRaster->addBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL);
    m_dsetRaster->addBinding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_ALL);
    m_dsetRaster->initLayout(VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR);

    const VkPushConstantRange push_constant_ranges = {VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                                                      sizeof(DH::PushConstant)};
    m_dsetRaster->initPipeLayout(1, &push_constant_ranges);

    // Descriptors writes
    m_descBufInfo = std::make_unique<VkDescriptorBufferInfo>(VkDescriptorBufferInfo{m_frameInfo.buffer, 0, VK_WHOLE_SIZE});
    m_dsetRastWrites.emplace_back(m_dsetRaster->makeWrite(0, 0, m_descBufInfo.get()));
    m_dsetRastWrites.emplace_back(m_dsetRaster->makeWrite(0, 1, &m_texture.descriptor));

    VkPipelineRenderingCreateInfo prend_info{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR};
    prend_info.colorAttachmentCount    = 1;
    prend_info.pColorAttachmentFormats = &m_colorFormat;
    prend_info.depthAttachmentFormat   = m_depthFormat;

    // Creating the Pipeline
    nvvk::GraphicsPipelineState pstate;
    pstate.rasterizationState.cullMode = VK_CULL_MODE_NONE;
    pstate.addBindingDescriptions({{0, sizeof(nvh::PrimitiveVertex)}});
    pstate.addAttributeDescriptions({
        {0, 0, VK_FORMAT_R32G32B32_SFLOAT, static_cast<uint32_t>(offsetof(nvh::PrimitiveVertex, p))},  // Position
    });

    nvvk::GraphicsPipelineGenerator pgen(m_device, m_dsetRaster->getPipeLayout(), prend_info, pstate);

    VkShaderModule shaderModule = nvvk::createShaderModule(m_device, &rasterSlang[0], sizeof(rasterSlang));
    pgen.addShader(shaderModule, VK_SHADER_STAGE_VERTEX_BIT, "vertexMain");
    pgen.addShader(shaderModule, VK_SHADER_STAGE_FRAGMENT_BIT, "fragmentMain");


    m_graphicsPipeline = pgen.createPipeline();
    m_dutil->setObjectName(m_graphicsPipeline, "Graphics");
    pgen.clearShaders();
    vkDestroyShaderModule(m_device, shaderModule, nullptr);
  }


private:
  // Local data
  nvvk::Texture   m_texture;
  nvvk::Texture   m_texture_weights;
  nvvk::Texture   m_depth_texture;
  VkDevice        m_device          = VK_NULL_HANDLE;
  VkDescriptorSet m_descriptorSet   = VK_NULL_HANDLE;
  VkPipeline      m_computePipeline = VK_NULL_HANDLE;  // The graphic pipeline to render
  bool            m_dirty           = false;

  std::unique_ptr<nvvkhl::AllocVma>             m_alloc;
  std::unique_ptr<nvvk::DebugUtil>              m_dutil;
  std::unique_ptr<nvvk::DescriptorSetContainer> m_dsetRaster;   // Holding the descriptor set information
  std::unique_ptr<nvvk::DescriptorSetContainer> m_dsetCompute;  // Holding the descriptor set information
  std::unique_ptr<nvvkhl::GBuffer>              m_gBuffers;     // G-Buffers: color + depth

  nvvk::Buffer m_vertices;  // Buffer of the vertices
  nvvk::Buffer m_indices;   // Buffer of the indices
  nvvk::Buffer m_frameInfo;

  nvvkhl::Application*                    m_app = nullptr;
  std::vector<VkWriteDescriptorSet>       m_dsetCompWrites;
  std::vector<VkWriteDescriptorSet>       m_dsetRastWrites;
  std::unique_ptr<VkDescriptorBufferInfo> m_descBufInfo;

  Settings m_settings = {};

  VkFormat          m_colorFormat      = VK_FORMAT_R8G8B8A8_UNORM;       // Color format of the image
  VkFormat          m_depthFormat      = VK_FORMAT_X8_D24_UNORM_PACK32;  // Depth format of the depth buffer
  VkPipeline        m_graphicsPipeline = VK_NULL_HANDLE;                 // The graphic pipeline to render
  VkClearColorValue m_clearColor       = {{0.3F, 0.3F, 0.3F, 1.0F}};     // Clear color

  VirtualSensor sensor;
  std::vector<float> voxel_grid;
  std::vector<float> voxel_grid_weights;
  bool gpu_ini = false;
};

int main(int argc, char** argv)
{
  VkContextSettings vkSetup{
      .instanceExtensions = {VK_EXT_DEBUG_UTILS_EXTENSION_NAME},
      .deviceExtensions   = {{VK_KHR_SWAPCHAIN_EXTENSION_NAME}, {VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME}},
      .queues             = {VK_QUEUE_GRAPHICS_BIT},
  };
  nvvkhl::addSurfaceExtensions(vkSetup.instanceExtensions);

  // Vulkan context creation
  auto vkContext = std::make_unique<VulkanContext>(vkSetup);
  if(!vkContext->isValid())
    std::exit(0);

  // Loading function pointers
  load_VK_EXTENSIONS(vkContext->getInstance(), vkGetInstanceProcAddr, vkContext->getDevice(), vkGetDeviceProcAddr);

  nvvkhl::ApplicationCreateInfo appSetup;
  appSetup.name           = fmt::format("{} ({})", PROJECT_NAME, SHADER_LANGUAGE_STR);
  appSetup.vSync          = true;
  appSetup.instance       = vkContext->getInstance();
  appSetup.device         = vkContext->getDevice();
  appSetup.physicalDevice = vkContext->getPhysicalDevice();
  appSetup.queues         = vkContext->getQueueInfos();

  // Create the application
  auto app = std::make_unique<nvvkhl::Application>(appSetup);

  // Create this example
  auto test = std::make_shared<nvvkhl::ElementBenchmarkParameters>(argc, argv);
  app->addElement(test);
  app->addElement(std::make_shared<nvvkhl::ElementCamera>());
  app->addElement(std::make_shared<nvvkhl::ElementDefaultMenu>());
  app->addElement(std::make_shared<nvvkhl::ElementDefaultWindowTitle>("", fmt::format("({})", SHADER_LANGUAGE_STR)));  // Window title info
  app->addElement(std::make_shared<Texture3dSample>());

  app->run();

  app.reset();
  vkContext.reset();

  return test->errorCode();
}
