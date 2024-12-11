/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <vulkan/vulkan_core.h>

#include "imgui/imgui_helper.h"
#include "nvvk/context_vk.hpp"
#include "nvvk/descriptorsets_vk.hpp"
#include "nvvk/pipeline_vk.hpp"
#include "nvvk/shaders_vk.hpp"
#include "nvvk/debug_util_vk.hpp"

#include "shaders/dh_comp.h"
#include "_autogen/passthrough.vert.glsl.h"
#include "_autogen/tonemapper.frag.glsl.h"
#include "_autogen/tonemapper.comp.glsl.h"

#include "tonemap_postprocess.hpp"

namespace nvvkhl {
TonemapperPostProcess::TonemapperPostProcess(VkDevice device, nvvk::ResourceAllocator* alloc)
{
  init(device, alloc);
}

TonemapperPostProcess::~TonemapperPostProcess()
{
  deinit();
}

void TonemapperPostProcess::init(VkDevice device, nvvk::ResourceAllocator* alloc)
{
  m_device = (device);
  m_dutil  = (std::make_unique<nvvk::DebugUtil>(m_device));
  m_dsetGraphics.init(m_device);
  m_dsetCompute.init(m_device);
  m_settings = nvvkhl_shaders::defaultTonemapper();
}

void TonemapperPostProcess::deinit()
{
  vkDestroyPipeline(m_device, m_graphicsPipeline, nullptr);
  vkDestroyPipeline(m_device, m_computePipeline, nullptr);
  m_dsetGraphics.deinit();
  m_dsetCompute.deinit();
}

void TonemapperPostProcess::createGraphicPipeline(VkFormat colorFormat, VkFormat depthFormat)
{
  m_mode = TmMode::eGraphic;

  auto& d = m_dsetGraphics;
  m_dsetGraphics.addBinding(nvvkhl_shaders::eTonemapperInput, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT);
  m_dsetGraphics.initLayout(VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR);
  m_dutil->DBG_NAME(m_dsetGraphics.getLayout());

  VkPushConstantRange push_constant_ranges = {VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(nvvkhl_shaders::Tonemapper)};

  m_dsetGraphics.initPipeLayout(1, &push_constant_ranges);
  m_dutil->DBG_NAME(m_dsetGraphics.getPipeLayout());

  VkPipelineRenderingCreateInfo prend_info{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR};
  prend_info.colorAttachmentCount    = 1;
  prend_info.pColorAttachmentFormats = &colorFormat;
  prend_info.depthAttachmentFormat   = depthFormat;

  // Creating the Pipeline
  nvvk::GraphicsPipelineState pstate;
  pstate.rasterizationState.cullMode = VK_CULL_MODE_NONE;

  nvvk::GraphicsPipelineGenerator pgen(m_device, m_dsetGraphics.getPipeLayout(), prend_info, pstate);
  pgen.addShader(std::vector<uint32_t>{std::begin(passthrough_vert_glsl), std::end(passthrough_vert_glsl)}, VK_SHADER_STAGE_VERTEX_BIT);
  pgen.addShader(std::vector<uint32_t>{std::begin(tonemapper_frag_glsl), std::end(tonemapper_frag_glsl)}, VK_SHADER_STAGE_FRAGMENT_BIT);

  m_graphicsPipeline = pgen.createPipeline();
  m_dutil->DBG_NAME(m_graphicsPipeline);
  pgen.clearShaders();
}


void TonemapperPostProcess::createComputePipeline()
{
  m_mode = TmMode::eCompute;

  nvvk::DebugUtil dbg(m_device);

  m_dsetCompute.addBinding(nvvkhl_shaders::eTonemapperInput, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
  m_dsetCompute.addBinding(nvvkhl_shaders::eTonemapperOutput, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
  m_dsetCompute.initLayout(VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR);
  m_dutil->DBG_NAME(m_dsetCompute.getLayout());

  VkPushConstantRange push_constant_ranges = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(nvvkhl_shaders::Tonemapper)};

  m_dsetCompute.initPipeLayout(1, &push_constant_ranges);
  m_dutil->DBG_NAME(m_dsetCompute.getPipeLayout());

  VkPipelineShaderStageCreateInfo stage_info{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  stage_info.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
  stage_info.module = nvvk::createShaderModule(m_device, tonemapper_comp_glsl, sizeof(tonemapper_comp_glsl));
  stage_info.pName  = "main";

  VkComputePipelineCreateInfo comp_info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  comp_info.layout = m_dsetCompute.getPipeLayout();
  comp_info.stage  = stage_info;

  vkCreateComputePipelines(m_device, {}, 1, &comp_info, nullptr, &m_computePipeline);
  m_dutil->DBG_NAME(m_computePipeline);

  // Clean up
  vkDestroyShaderModule(m_device, comp_info.stage.module, nullptr);
}

void TonemapperPostProcess::updateGraphicDescriptorSets(VkDescriptorImageInfo inImage)
{
  assert(m_mode == TmMode::eGraphic);
  m_iimage = inImage;
  m_writes.clear();
  m_writes.emplace_back(m_dsetGraphics.makeWrite(0, nvvkhl_shaders::eTonemapperInput, &m_iimage));
}

void TonemapperPostProcess::updateComputeDescriptorSets(VkDescriptorImageInfo inImage, VkDescriptorImageInfo outImage)
{
  assert(m_mode == TmMode::eCompute);
  m_iimage = inImage;
  m_oimage = outImage;
  m_writes.clear();
  m_writes.emplace_back(m_dsetCompute.makeWrite(0, nvvkhl_shaders::eTonemapperInput, &m_iimage));
  m_writes.emplace_back(m_dsetCompute.makeWrite(0, nvvkhl_shaders::eTonemapperOutput, &m_oimage));
}


void TonemapperPostProcess::runGraphic(VkCommandBuffer cmd)
{
  assert(m_mode == TmMode::eGraphic);
  auto sdbg = m_dutil->DBG_SCOPE(cmd);
  vkCmdPushConstants(cmd, m_dsetGraphics.getPipeLayout(), VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                     sizeof(nvvkhl_shaders::Tonemapper), &m_settings);

  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphicsPipeline);
  vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_dsetGraphics.getPipeLayout(), 0,
                            static_cast<uint32_t>(m_writes.size()), m_writes.data());
  vkCmdDraw(cmd, 3, 1, 0, 0);
}

void TonemapperPostProcess::runCompute(VkCommandBuffer cmd, const VkExtent2D& size)
{
  assert(m_mode == TmMode::eCompute);
  auto sdbg = m_dutil->DBG_SCOPE(cmd);
  vkCmdPushConstants(cmd, m_dsetCompute.getPipeLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0,
                     sizeof(nvvkhl_shaders::Tonemapper), &m_settings);
  vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_dsetCompute.getPipeLayout(), 0,
                            static_cast<uint32_t>(m_writes.size()), m_writes.data());
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_computePipeline);
  VkExtent2D group_counts = getGroupCounts(size);
  vkCmdDispatch(cmd, group_counts.width, group_counts.height, 1);
}


bool TonemapperPostProcess::onUI()
{
  bool changed{false};

  const char* items[] = {"Filmic", "Uncharted 2", "Clip", "ACES", "AgX", "Khronos PBR"};

  namespace PE = ImGuiH::PropertyEditor;
  PE::begin();
  changed |= PE::Combo("Method", &m_settings.method, items, IM_ARRAYSIZE(items));
  changed |=
      PE::entry("Active", [&]() { return ImGui::Checkbox("Active", reinterpret_cast<bool*>(&m_settings.isActive)); });
  changed |= PE::SliderFloat("Exposure", &m_settings.exposure, 0.1F, 15.0F, "%.3f", ImGuiSliderFlags_Logarithmic);
  changed |= PE::SliderFloat("Brightness", &m_settings.brightness, 0.0F, 2.0F);
  changed |= PE::SliderFloat("Contrast", &m_settings.contrast, 0.0F, 2.0F);
  changed |= PE::SliderFloat("Saturation", &m_settings.saturation, 0.0F, 2.0F);
  changed |= PE::SliderFloat("Vignette", &m_settings.vignette, 0.0F, 1.0F);

  if(ImGui::SmallButton("reset"))
  {
    m_settings = nvvkhl_shaders::defaultTonemapper();
    changed    = true;
  }
  PE::end();
  return changed;
}

}  // namespace nvvkhl
