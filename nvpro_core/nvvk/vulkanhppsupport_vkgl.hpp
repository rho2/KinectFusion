/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#if NVP_SUPPORTS_OPENGL

#pragma once

#include "memorymanagement_vkgl.hpp"  // This needs to be first to not break the build

#include "nvvk/vulkanhppsupport.hpp"

namespace nvvkpp {
/** @DOC_START
# class nvvkpp::ResourceAllocatorGLInterop

>  ResourceAllocatorGLInterop is a helper class to manage Vulkan and OpenGL memory allocation and interop.

This class is a wrapper around the `nvvk::DeviceMemoryAllocatorGL` and `nvvk::DeviceMemoryAllocator` classes, which are used to allocate memory for Vulkan and OpenGL resources.

@DOC_END */

class ResourceAllocatorGLInterop : public ExportResourceAllocator
{
public:
  ResourceAllocatorGLInterop() = default;
  ResourceAllocatorGLInterop(VkDevice device, VkPhysicalDevice physicalDevice, VkDeviceSize stagingBlockSize = NVVK_DEFAULT_STAGING_BLOCKSIZE);
  ~ResourceAllocatorGLInterop();

  void init(VkDevice device, VkPhysicalDevice physicalDevice, VkDeviceSize stagingBlockSize = NVVK_DEFAULT_STAGING_BLOCKSIZE);
  void deinit();

  nvvk::DeviceMemoryAllocatorGL& getDmaGL() const { return *m_dmaGL; }
  nvvk::AllocationGL             getAllocationGL(nvvk::MemHandle memHandle) const;

protected:
  std::unique_ptr<nvvk::DeviceMemoryAllocatorGL> m_dmaGL;
  std::unique_ptr<nvvk::DeviceMemoryAllocator>   m_dma;
};

}  // namespace nvvkpp

#endif
