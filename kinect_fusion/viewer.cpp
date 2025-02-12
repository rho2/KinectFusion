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
    uint32_t             voxelSize;
    bool                 renderNormals  = false;
    VkFilter             magFilter      = VK_FILTER_NEAREST;
    VkSamplerAddressMode addressMode    = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    DH::PerlinSettings   perlin         = DH::PerlinSettings();
    int                  headlight      = 1;
    glm::vec3            toLight        = {0.F, 1.F, 0.F};
    int                  steps          = 1000;
    glm::vec4            surfaceColor   = {0.8F, 0.8F, 0.8F, 1.0F};
    uint32_t             getSize() { return voxelSize; }
    uint32_t             getTotalSize() { return getSize() * getSize() * getSize(); }
  };

public:
  Texture3dSample()           = default;
  ~Texture3dSample() override = default;

  // Implementation of nvvk::IApplication interface
  void onAttach(nvvkhl::Application* app) override
  {
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
    // m_dsetCompute = std::make_unique<nvvk::DescriptorSetContainer>(m_device);
    m_dsetRaster  = std::make_unique<nvvk::DescriptorSetContainer>(m_device);
    m_depthFormat = nvvk::findDepthFormat(app->getPhysicalDevice());

    fillPerlinImage();

    createVkBuffers();
    createTextureBuffers();
    createTexture();
    createGraphicPipeline();

    // Setting the default camera
    CameraManip.setClipPlanes({0.01F, 100.0F});
    CameraManip.setLookat({-0.58981F, 0.25795F, -0.26918F}, {-0.03199F, -0.12156F, -0.08628F}, {0.0F, 1.0F, 0.0F});
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
        "Steps", [&] { return ImGui::SliderInt("##2", (int*)&m_settings.steps, 1, 1000); }, "Number of maximum steps.");
    PE::end();

    PE::begin();
    PE::entry(
        "Render normals", [&] { return ImGui::Checkbox("##4", &s.renderNormals); },
        "Render normals instead of pure image");
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
  void createTextureBuffers() {
    assert(!m_texture.image);

    VkImageCreateInfo create_info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    create_info.imageType     = VK_IMAGE_TYPE_3D;
    create_info.format        = VK_FORMAT_R32_SFLOAT;
    create_info.mipLevels     = 1;
    create_info.arrayLayers   = 1;
    create_info.samples       = VK_SAMPLE_COUNT_1_BIT;
    create_info.extent.width  = 1;
    create_info.extent.height = 1;
    create_info.extent.depth  = 1;
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
    view_info.format                          = VK_FORMAT_R32_SFLOAT;
    view_info.viewType                        = VK_IMAGE_VIEW_TYPE_3D;
    view_info.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    view_info.subresourceRange.baseMipLevel   = 0;
    view_info.subresourceRange.levelCount     = VK_REMAINING_MIP_LEVELS;
    view_info.subresourceRange.baseArrayLayer = 0;
    view_info.subresourceRange.layerCount     = VK_REMAINING_ARRAY_LAYERS;

    m_texture                        = m_alloc->createTexture(texImage, view_info, sampler_info);
    m_texture.descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
  }

  void createTexture()
  {
    nvh::ScopedTimer st(__FUNCTION__);

    VkCommandBuffer cmd = m_app->createTempCmdBuffer();

    nvvk::cmdBarrierImageLayout(cmd, m_texture.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    setData(cmd);

    m_app->submitAndWaitTempCmdBuffer(cmd);

    // Debugging information
    m_dutil->setObjectName(m_texture.image, "Image");
    m_dutil->setObjectName(m_texture.descriptor.sampler, "Sampler");
    m_descriptorSet = ImGui_ImplVulkan_AddTexture(m_texture.descriptor.sampler, m_texture.descriptor.imageView,
                                                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  }

  void fillPerlinImage()
  {
    nvh::ScopedTimer st(__FUNCTION__);

    std::ostringstream filename;
    filename << "sdf_values" << "_" << std::setw(4) << std::setfill('0') << 0 << ".bin";

    std::ifstream file(filename.str(), std::ios::binary);

    if (!file) {
      std::cerr << "Error opening file: " << filename.str() << std::endl;
      return;
    }

    file.read(reinterpret_cast<char*>(&m_settings.voxelSize), sizeof(uint32_t));


    uint32_t realSize = m_settings.getSize();
    uint32_t totalSize = realSize * realSize * realSize;

    voxel_grid.resize(totalSize);
    file.read(reinterpret_cast<char*>(voxel_grid.data()), totalSize * sizeof(float));
    std::cout << "Read file: " << filename.str() << " Size: " << m_settings.voxelSize << "\n";
  }

  void setData(VkCommandBuffer cmd)
  {
    const nvvk::DebugUtil::ScopedCmdLabel sdbg = m_dutil->DBG_SCOPE(cmd);
    assert(m_texture.image);

    nvvk::StagingMemoryManager* staging = m_alloc->getStaging();
    staging->cmdToBuffer(cmd, m_bufferGrid.buffer, 0, voxel_grid.size() * sizeof(float), voxel_grid.data());

    m_dirty = false;
  }

  void destroyResources()
  {
    m_dsetRaster->deinit();
    vkDestroyPipeline(m_device, m_graphicsPipeline, nullptr);

    m_alloc->destroy(m_vertices);
    m_alloc->destroy(m_indices);
    m_alloc->destroy(m_frameInfo);
    m_alloc->destroy(m_bufferGrid);
    m_alloc->destroy(m_texture);
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

    m_bufferGrid = m_alloc->createBuffer(sizeof(float) * m_settings.getTotalSize(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    m_dutil->DBG_NAME(m_bufferGrid.buffer);

    m_app->submitAndWaitTempCmdBuffer(cmd);
  }

  void createGraphicPipeline()
  {
    m_dsetRaster->addBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL);
    m_dsetRaster->addBinding(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
    m_dsetRaster->initLayout(VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR);

    const VkPushConstantRange push_constant_ranges = {VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                                                      sizeof(DH::PushConstant)};
    m_dsetRaster->initPipeLayout(1, &push_constant_ranges);

    // Descriptors writes
    m_descBufInfo = std::make_unique<VkDescriptorBufferInfo>(VkDescriptorBufferInfo{m_frameInfo.buffer, 0, VK_WHOLE_SIZE});
    m_descBufInfoGrid = std::make_unique<VkDescriptorBufferInfo>(VkDescriptorBufferInfo{m_bufferGrid.buffer, 0, VK_WHOLE_SIZE});

    m_dsetRastWrites.emplace_back(m_dsetRaster->makeWrite(0, 0, m_descBufInfo.get()));
    // m_dsetRastWrites.emplace_back(m_dsetRaster->makeWrite(0, 1, &m_texture.descriptor));
    m_dsetRastWrites.emplace_back(m_dsetRaster->makeWrite(0, 1, m_descBufInfoGrid.get()));

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
  VkDevice        m_device          = VK_NULL_HANDLE;
  VkDescriptorSet m_descriptorSet   = VK_NULL_HANDLE;
  bool            m_dirty           = false;

  std::unique_ptr<nvvkhl::AllocVma>             m_alloc;
  std::unique_ptr<nvvk::DebugUtil>              m_dutil;
  std::unique_ptr<nvvk::DescriptorSetContainer> m_dsetRaster;   // Holding the descriptor set information
  std::unique_ptr<nvvkhl::GBuffer>              m_gBuffers;     // G-Buffers: color + depth

  nvvk::Buffer m_vertices;  // Buffer of the vertices
  nvvk::Buffer m_indices;   // Buffer of the indices
  nvvk::Buffer m_frameInfo;
  nvvk::Buffer m_bufferGrid;

  nvvkhl::Application*                    m_app = nullptr;
  std::vector<VkWriteDescriptorSet>       m_dsetRastWrites;
  std::unique_ptr<VkDescriptorBufferInfo> m_descBufInfo;
  std::unique_ptr<VkDescriptorBufferInfo> m_descBufInfoGrid;

  Settings m_settings = {};

  VkFormat          m_colorFormat      = VK_FORMAT_R8G8B8A8_UNORM;       // Color format of the image
  VkFormat          m_depthFormat      = VK_FORMAT_X8_D24_UNORM_PACK32;  // Depth format of the depth buffer
  VkPipeline        m_graphicsPipeline = VK_NULL_HANDLE;                 // The graphic pipeline to render
  VkClearColorValue m_clearColor       = {{0.3F, 0.3F, 0.3F, 1.0F}};     // Clear color

  std::vector<float> voxel_grid;
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
