/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */


//////////////////////////////////////////////////////////////////////////
/** @DOC_START
# Function nvvk::checkResult
>  Returns true on critical error result, logs errors.

Use `NVVK_CHECK(result)` to automatically log filename/linenumber.
@DOC_END */

#pragma once

#include <cassert>
#include <functional>
#include <vulkan/vulkan_core.h>

namespace nvvk {
bool checkResult(VkResult result, const char* message = nullptr);
bool checkResult(VkResult result, const char* file, int32_t line);

/** @DOC_START
# Function nvvk::setCheckResultHook
>  Allow replacing nvvk::checkResult() calls. E.g. to catch
`VK_ERROR_DEVICE_LOST` and wait for aftermath to write the crash dump.
@DOC_END */
using CheckResultCallback = std::function<bool(VkResult, const char*, int32_t, const char*)>;
void setCheckResultHook(const CheckResultCallback& callback);

#ifndef NVVK_CHECK
#define NVVK_CHECK(result) nvvk::checkResult(result, __FILE__, __LINE__)
#endif

}  // namespace nvvk
