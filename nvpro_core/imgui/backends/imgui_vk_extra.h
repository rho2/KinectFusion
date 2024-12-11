/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

/// @DOC_SKIP (keyword to exclude this file from automatic README.md generation)

#include <backends/imgui_impl_vulkan.h>

// the usage of these functions here implies that one doesn't leverage imgui's own vulkan
// window management.
//
// either use renderpass version, or dynamic rendering

namespace ImGui {
void InitVK(VkDevice device, VkPhysicalDevice physicalDevice, VkQueue queue, uint32_t queueFamilyIndex, VkRenderPass pass, uint32_t subPassIndex = 0);
void InitVK(VkInstance                           instance,
            VkDevice                             device,
            VkPhysicalDevice                     physicalDevice,
            VkQueue                              queue,
            uint32_t                             queueFamilyIndex,
            const VkPipelineRenderingCreateInfo& dynamicRendering);
void ShutdownVK();
}  // namespace ImGui
