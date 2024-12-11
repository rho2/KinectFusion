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


#version 450
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "dh_hdr.h"
#include "constants.h"
#include "func.h"

layout(local_size_x = WORKGROUP_SIZE, local_size_y = WORKGROUP_SIZE, local_size_z = 1) in;

layout(set = 0, binding = eHdrBrdf) writeonly uniform image2D gOutHdr;
layout(set = 1, binding = eHdr) uniform sampler2D gInHdr;


layout(push_constant) uniform SkyDomePushConstant_
{
  HdrDomePushConstant pc;
};

void main()
{
  const vec2 pixel_center = vec2(gl_GlobalInvocationID.xy) + vec2(0.5);
  const vec2 in_uv        = pixel_center / vec2(imageSize(gOutHdr));
  const vec2 d            = in_uv * 2.0F - 1.0F;
  vec3       direction    = vec3(pc.mvp * vec4(d.x, d.y, 1.0F, 1.0F));

  direction = rotate(direction, vec3(0.0F, 1.0F, 0.0F), -pc.rotation);

  const vec2 uv = getSphericalUv(normalize(direction.xyz));
  vec3       color;
  if(pc.blur > 0.0F)
  {
    color = smoothHDRBlur(gInHdr, uv, pc.blur).rgb;
  }
  else
  {
    color = texture(gInHdr, uv).rgb;
  }
  color *= pc.multColor.rgb;
  imageStore(gOutHdr, ivec2(gl_GlobalInvocationID.xy), vec4(color, 1.0));
}
