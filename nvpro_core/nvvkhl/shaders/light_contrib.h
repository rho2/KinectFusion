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

#ifndef LIGHT_CONTRIB_H
#define LIGHT_CONTRIB_H 1

#include "func.h"
#include "dh_lighting.h"

/* @DOC_START
# Function `singleLightContribution`
> Returns an estimate of the contribution of a light to a given point on a surface.

Inputs:
- `light`: A `Light` structure; see dh_lighting.h.
- `surfacePos`: The surface position.
- `surfaceNormal`: The surface normal.
- `randVal`: Used to randomly sample points on area (disk) lights. This means
that the light contribution from an area light will be noisy.
@DOC_END */
LightContrib singleLightContribution(in Light light, in vec3 surfacePos, in vec3 surfaceNormal, in vec2 randVal)
{
  LightContrib contrib;
  contrib.incidentVector  = vec3(0.0F);
  contrib.halfAngularSize = 0.0F;
  contrib.intensity       = vec3(0.0F);
  contrib.distance        = INFINITE;
  float irradiance        = 0.0F;

  if(light.type == eLightTypeDirectional)
  {
    if(dot(surfaceNormal, -light.direction) <= 0.0)
      return contrib;

    contrib.incidentVector  = light.direction;
    contrib.halfAngularSize = light.angularSizeOrInvRange * 0.5F;
    irradiance              = light.intensity;
  }
  else if(light.type == eLightTypeSpot || light.type == eLightTypePoint)
  {
    vec3  light_to_surface = surfacePos - light.position;
    float distance         = sqrt(dot(light_to_surface, light_to_surface));
    float r_distance       = 1.0F / distance;

    contrib.distance       = distance;
    contrib.incidentVector = light_to_surface * r_distance;

    float attenuation = 1.F;
    if(light.angularSizeOrInvRange > 0.0F)
    {
      attenuation = square(saturate(1.0F - square(square(distance * light.angularSizeOrInvRange))));

      if(attenuation == 0.0F)
        return contrib;
    }

    float spotlight = 1.0F;
    if(light.type == eLightTypeSpot)
    {
      float lDotD           = dot(contrib.incidentVector, light.direction);
      float direction_angle = acos(lDotD);
      spotlight             = 1.0F - smoothstep(light.innerAngle, light.outerAngle, direction_angle);

      if(spotlight == 0.0F)
        return contrib;
    }

    if(light.radius > 0.0F)
    {
      contrib.halfAngularSize = atan(min(light.radius * r_distance, 1.0F));

      // A good enough approximation for 2 * (1 - cos(halfAngularSize)), numerically more accurate for small angular sizes
      float solidAngleOverPi = square(contrib.halfAngularSize);

      float radianceTimesPi = light.intensity / square(light.radius);

      irradiance = radianceTimesPi * solidAngleOverPi;
    }
    else
    {
      irradiance = light.intensity * square(r_distance);
    }

    irradiance *= spotlight * attenuation;
  }

  contrib.intensity = irradiance * light.color;


  if(contrib.halfAngularSize > 0.0F)
  {  // <----- Sampling area lights
    float angular_size = contrib.halfAngularSize;

    // section 34  https://people.cs.kuleuven.be/~philip.dutre/GI/TotalCompendium.pdf
    vec3  dir;
    float tmp      = (1.0F - randVal.y * (1.0F - cos(angular_size)));
    float tmp2     = tmp * tmp;
    float theta    = sqrt(1.0F - tmp2);
    dir.x          = cos(M_TWO_PI * randVal.x) * theta;
    dir.y          = sin(M_TWO_PI * randVal.x) * theta;
    dir.z          = tmp;
    vec3 light_dir = -contrib.incidentVector;
    vec3 tangent, binormal;
    orthonormalBasis(light_dir, tangent, binormal);
    mat3 tbn  = mat3(tangent, binormal, light_dir);
    light_dir = normalize(tbn * dir);

    contrib.incidentVector = -light_dir;
  }

  return contrib;
}

/* @DOC_START
# Function `singleLightContribution` (three-argument overload)
> Like `singleLightContribution` above, but without using random values.

Because this is equivalent to `singleLightContribution(..., vec2(0.0F))`, area
lights won't be noisy, but will have harder falloff than they would
otherwise have.
@DOC_END */
LightContrib singleLightContribution(in Light light, in vec3 surfacePos, in vec3 surfaceNormal)
{
  return singleLightContribution(light, surfacePos, surfaceNormal, vec2(0.0F));
}
#endif