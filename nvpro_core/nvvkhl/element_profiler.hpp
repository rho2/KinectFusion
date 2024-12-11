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
#pragma once

/** @DOC_START
# class nvvkhl::ElementProfiler
>  This class is an element of the application that is responsible for the profiling of the application. It is using the `nvvk::ProfilerVK` to profile the time parts of the computation done on the GPU.

To use this class, you need to add it to the `nvvkhl::Application` using the `addElement` method.

The profiler element, is there to help profiling the time parts of the 
computation is done on the GPU. To use it, follow those simple steps

In the main() program, create an instance of the profiler and add it to the
nvvkhl::Application

```cpp
  std::shared_ptr<nvvkhl::ElementProfiler> profiler = std::make_shared<nvvkhl::ElementProfiler>();
  app->addElement(profiler);
```

In the application where profiling needs to be done, add profiling sections

```cpp
void mySample::onRender(VkCommandBuffer cmd)
{
  auto sec = m_profiler->timeRecurring(__FUNCTION__, cmd);
  ...
  // Subsection
  {
    auto sec = m_profiler->timeRecurring("Dispatch", cmd);
    vkCmdDispatch(cmd, (size.width + (GROUP_SIZE - 1)) / GROUP_SIZE, (size.height + (GROUP_SIZE - 1)) / GROUP_SIZE, 1);
  }
  ...
```

This is it and the execution time on the GPU for each part will be showing in the Profiler window.

@DOC_END */

#include <implot.h>
#include <imgui_internal.h>

#include "application.hpp"

#include "nvh/commandlineparser.hpp"
#include "nvh/nvprint.hpp"
#include "nvh/timesampler.hpp"
#include "nvpsystem.hpp"
#include "nvvk/error_vk.hpp"
#include "nvvk/profiler_vk.hpp"
#include "nvvk/debug_util_vk.hpp"

#define PROFILER_GRAPH_TEMPORAL_SMOOTHING 20.f
#define PROFILER_GRAPH_MINIMAL_LUMINANCE 0.1f

namespace nvvkhl {

class ElementProfiler : public nvvkhl::IAppElement, public nvvk::ProfilerVK
{
public:
  ElementProfiler(bool showWindow = true)
      : m_showWindow(showWindow)
  {
    addSettingsHandler();
  };
  ~ElementProfiler() = default;

  void onAttach(Application* app) override
  {
    m_app = app;

    nvvk::ProfilerVK::init(m_app->getDevice(), m_app->getPhysicalDevice());
    bool hasDebugUtils = nvvk::DebugUtil::isEnabled();
    nvvk::ProfilerVK::setLabelUsage(hasDebugUtils);
    nvvk::ProfilerVK::beginFrame();
  }

  void onDetach() override
  {
    nvvk::ProfilerVK::endFrame();
    vkDeviceWaitIdle(m_app->getDevice());
    nvvk::ProfilerVK::deinit();
  }

  void onUIMenu() override
  {
    if(ImGui::BeginMenu("View"))
    {
      ImGui::MenuItem("Profiler", "", &m_showWindow);
      ImGui::EndMenu();
    }
  }  // This is the menubar to create


  void onUIRender() override
  {
    constexpr float frequency    = (1.0f / 60.0f);
    static float    s_minElapsed = 0;
    s_minElapsed += ImGui::GetIO().DeltaTime;

    if(!m_showWindow)
      return;

    // Opening the window
    if(!ImGui::Begin("Profiler", &m_showWindow))
    {
      ImGui::End();
      return;
    }

    if(s_minElapsed >= frequency)
    {
      s_minElapsed = 0;
      m_node.child.clear();
      m_node.name    = "Frame";
      m_node.cpuTime = static_cast<float>(m_data->cpuTime.getAveraged() / 1000.);
      m_single.child.clear();
      m_single.name = "Single";
      addEntries(m_node.child, 0, m_data->numLastSections, 0);
    }

    bool copyToClipboard = ImGui::SmallButton("Copy");
    if(copyToClipboard)
      ImGui::LogToClipboard();

    if(ImGui::BeginTabBar("Profiler Tabs"))
    {
      if(ImGui::BeginTabItem("Table"))
      {
        renderTable();
        ImGui::EndTabItem();
      }
      if(ImGui::BeginTabItem("PieChart"))
      {
        renderPieChart();
        ImGui::EndTabItem();
      }
      if(ImGui::BeginTabItem("LineChart"))
      {
        renderLineChart();
        ImGui::EndTabItem();
      }
      ImGui::EndTabBar();
    }


    if(copyToClipboard)
      ImGui::LogFinish();

    ImGui::End();
  }


  void onRender(VkCommandBuffer /*cmd*/) override
  {
    nvvk::ProfilerVK::endFrame();
    nvvk::ProfilerVK::beginFrame();
  }

private:
  struct MyEntryNode
  {
    std::string              name;
    float                    cpuTime = 0.f;
    float                    gpuTime = -1.f;
    std::vector<MyEntryNode> child;
    Entry*                   entry = nullptr;
  };

  uint32_t addEntries(std::vector<MyEntryNode>& nodes, uint32_t startIndex, uint32_t endIndex, uint32_t currentLevel)
  {
    for(uint32_t curIndex = startIndex; curIndex < endIndex; curIndex++)
    {
      Entry& entry = m_data->entries[curIndex];
      if(entry.level < currentLevel)
        return curIndex;

      MyEntryNode entryNode;
      entryNode.name    = entry.name.empty() ? "N/A" : entry.name;
      entryNode.gpuTime = static_cast<float>(entry.gpuTime.getAveraged() / 1000.);
      entryNode.cpuTime = static_cast<float>(entry.cpuTime.getAveraged() / 1000.);
      entryNode.entry   = &entry;
      if(entry.level == LEVEL_SINGLESHOT)
      {
        m_single.child.push_back(entryNode);
        continue;
      }

      uint32_t nextLevel = curIndex + 1 < endIndex ? m_data->entries[curIndex + 1].level : currentLevel;
      if(nextLevel > currentLevel)
      {
        curIndex = addEntries(entryNode.child, curIndex + 1, endIndex, nextLevel);
      }
      nodes.push_back(entryNode);
      if(nextLevel < currentLevel)
        return curIndex;
    }
    return endIndex;
  }


  void displayTableNode(const MyEntryNode& node, uint32_t depth = 0)
  {
    ImGuiTableFlags flags = ImGuiTreeNodeFlags_SpanFullWidth | ImGuiTreeNodeFlags_SpanAllColumns;

    // Systematically open the first level
    if(depth < 1)
    {
      flags |= ImGuiTreeNodeFlags_DefaultOpen;
    }
    ImGui::TableNextRow();
    ImGui::TableNextColumn();
    const bool is_folder = (node.child.empty() == false);
    flags = is_folder ? flags : flags | ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_Bullet | ImGuiTreeNodeFlags_NoTreePushOnOpen;
    bool open = ImGui::TreeNodeEx(node.name.c_str(), flags);
    ImGui::TableNextColumn();
    if(node.gpuTime <= 0)
      ImGui::TextDisabled("--");
    else
      ImGui::Text("%3.3f", node.gpuTime);
    ImGui::TableNextColumn();
    if(node.cpuTime <= 0)
      ImGui::TextDisabled("--");
    else
      ImGui::Text("%3.3f", node.cpuTime);
    if((open) && is_folder)
    {
      for(int child_n = 0; child_n < static_cast<int>(node.child.size()); child_n++)
      {
        displayTableNode(node.child[child_n], depth + 1);
      }
      ImGui::TreePop();
    }
  }


  void renderTable()
  {
    // Using those as a base value to create width/height that are factor of the size of our font
    const float textBaseWidth = ImGui::CalcTextSize("A").x;

    static ImGuiTableFlags s_flags = ImGuiTableFlags_BordersV | ImGuiTableFlags_BordersOuterH | ImGuiTableFlags_Resizable
                                     | ImGuiTableFlags_RowBg | ImGuiTableFlags_NoBordersInBody;
    bool copy = false;
    if(ImGui::Button("Copy"))
    {
      ImGui::LogToClipboard();
      copy = true;
    }
    if(ImGui::BeginTable("EntryTable", 3, s_flags))
    {
      // The first column will use the default _WidthStretch when ScrollX is Off and _WidthFixed when ScrollX is On
      ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_NoHide);
      ImGui::TableSetupColumn("GPU", ImGuiTableColumnFlags_WidthFixed, textBaseWidth * 4.0f);
      ImGui::TableSetupColumn("CPU", ImGuiTableColumnFlags_WidthFixed, textBaseWidth * 4.0f);
      ImGui::TableHeadersRow();

      displayTableNode(m_node);

      // Display only if an element
      if(!m_single.child.empty())
      {
        displayTableNode(m_single);
      }

      ImGui::EndTable();
    }
    if(copy)
    {
      ImGui::LogFinish();
    }
  }

  //-------------------------------------------------------------------------------------------------
  // Rendering the data as a PieChart, showing the percentage of utilization
  //
  void renderPieChart()
  {
    static bool s_showSubLevel = false;
    ImGui::Checkbox("Show SubLevel 1", &s_showSubLevel);

    if(ImPlot::BeginPlot("##Pie1", ImVec2(-1, -1), ImPlotFlags_NoMouseText))
    {
      ImPlot::SetupAxes(nullptr, nullptr, ImPlotAxisFlags_NoDecorations | ImPlotAxisFlags_Lock,
                        ImPlotAxisFlags_NoDecorations | ImPlotAxisFlags_Lock);
      ImPlot::SetupAxesLimits(0, 1, 0, 1, ImPlotCond_Always);

      // Get all Level 0
      std::vector<const char*> labels1(m_node.child.size());
      std::vector<float>       data1(m_node.child.size());
      double                   angle0 = 90;
      for(size_t i = 0; i < m_node.child.size(); i++)
      {
        labels1[i] = m_node.child[i].name.c_str();
        data1[i]   = m_node.child[i].gpuTime / m_node.cpuTime;
      }

      ImPlot::PlotPieChart(labels1.data(), data1.data(), static_cast<int>(data1.size()), 0.5, 0.5, 0.4, "%.2f", angle0);

      // Level 1
      if(s_showSubLevel)
      {
        double a0 = angle0;
        for(size_t i = 0; i < m_node.child.size(); i++)
        {
          auto& currentNode = m_node.child[i];
          if(!currentNode.child.empty())
          {
            labels1.resize(currentNode.child.size());
            data1.resize(currentNode.child.size());
            for(size_t j = 0; j < currentNode.child.size(); j++)
            {
              labels1[j] = currentNode.child[j].name.c_str();
              data1[j]   = currentNode.child[j].gpuTime / m_node.cpuTime;
            }

            ImPlot::PlotPieChart(labels1.data(), data1.data(), static_cast<int>(data1.size()), 0.5, 0.5, 0.1, "", a0,
                                 ImPlotPieChartFlags_None);
          }

          // Increment the position of the next sub-element
          double percent = currentNode.gpuTime / m_node.cpuTime;
          a0 += a0 + 360 * percent;
        }
      }

      ImPlot::EndPlot();
    }
  }


  static uint32_t wangHash(uint32_t seed)
  {
    seed = (seed ^ 61) ^ (seed >> 16);
    seed *= 9;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2d;
    seed = seed ^ (seed >> 15);
    return seed;
  }

  static ImColor uintToColor(uint32_t v)
  {
    uint32_t hashed = wangHash(v);

    float r = (hashed & 0xFF) / 255.f;
    hashed  = hashed >> 8;
    float g = (hashed & 0xFF) / 255.f;
    hashed  = hashed >> 8;
    float b = (hashed & 0xFF) / 255.f;

    // Boost luminance of darker colors for visibility
    float luminance = (0.2126f * r + 0.7152f * g + 0.0722f * b);
    float boost     = std::max(1.f, PROFILER_GRAPH_MINIMAL_LUMINANCE / luminance);

    return ImColor(r * boost, g * boost, b * boost, 1.f);
  }

  //-------------------------------------------------------------------------------------------------
  // Rendering the data as a cumulated line chart
  //
  void renderLineChart()
  {
    std::vector<const char*>        gpuTimesLabels(m_node.child.size());
    std::vector<std::vector<float>> gpuTimes(m_node.child.size());
    std::vector<float>              cpuTimes(m_data->cpuTime.numValid);
    static float                    maxY       = 0.f;
    float                           avgCpuTime = 0.f;
    for(size_t i = 0; i < m_node.child.size(); i++)
    {
      gpuTimesLabels[i] = m_node.child[i].name.c_str();

      if(m_node.child[i].entry)
      {
        gpuTimes[i].resize(m_node.child[i].entry->gpuTime.numValid);
        for(size_t j = 0; j < m_node.child[i].entry->gpuTime.numValid; j++)
        {
          uint32_t index = (m_node.child[i].entry->gpuTime.index - m_node.child[i].entry->gpuTime.numValid + j) % m_data->numAveraging;
          gpuTimes[i][j] = float(m_node.child[i].entry->gpuTime.times[index] / 1000.0);
          if(i > 0)
          {
            gpuTimes[i][j] += gpuTimes[i - 1][j];
          }
        }
      }
    }

    for(size_t j = 0; j < m_data->cpuTime.numValid; j++)
    {
      uint32_t index = (m_data->cpuTime.index - m_data->cpuTime.numValid + j) % m_data->numAveraging;
      cpuTimes[j]    = float(m_data->cpuTime.times[index] / 1000.0);
      avgCpuTime += cpuTimes[j];
    }
    if(m_data->cpuTime.numValid > 0)
    {
      avgCpuTime /= m_data->cpuTime.numValid;
    }
    if(maxY == 0.f)
    {
      maxY = avgCpuTime;
    }
    else
    {
      maxY = (PROFILER_GRAPH_TEMPORAL_SMOOTHING * maxY + avgCpuTime) / (PROFILER_GRAPH_TEMPORAL_SMOOTHING + 1.f);
    }

    if(gpuTimes.size() > 0 && gpuTimes[0].size() > 0)
    {
      const ImPlotFlags     plotFlags = ImPlotFlags_NoBoxSelect | ImPlotFlags_NoMouseText | ImPlotFlags_Crosshairs;
      const ImPlotAxisFlags axesFlags = ImPlotAxisFlags_Lock | ImPlotAxisFlags_NoLabel;

      if(ImPlot::BeginPlot("##Line1", ImVec2(-1, -1), plotFlags))
      {
        ImPlot::SetupLegend(ImPlotLocation_NorthWest, ImPlotLegendFlags_NoButtons);
        ImPlot::SetupAxes(nullptr, "Count", axesFlags | ImPlotAxisFlags_NoTickLabels, axesFlags);
        ImPlot::SetupAxesLimits(0, m_node.child[0].entry->gpuTime.numValid, 0, maxY * 1.2f, ImPlotCond_Always);

        ImPlot::SetAxes(ImAxis_X1, ImAxis_Y1);
        ImPlot::SetNextLineStyle(ImColor(0.03f, 0.45f, 0.02f, 1.0f), 0.1f);

        ImPlot::PlotLine("CPU", cpuTimes.data(), (int)cpuTimes.size());

        ImPlot::PushStyleVar(ImPlotStyleVar_FillAlpha, 1.f);
        ImPlot::SetAxes(ImAxis_X1, ImAxis_Y1);

        for(size_t i = 0; i < m_node.child.size(); i++)
        {
          size_t index = m_node.child.size() - i - 1;

          uint32_t h = 0;
          for(size_t j = 0; j < m_node.child[index].name.size(); j++)
          {
            h = wangHash(h + m_node.child[index].name[j]);
          }
          ImPlot::SetNextFillStyle(uintToColor(h));
          ImPlot::PlotShaded(m_node.child[index].name.c_str(), gpuTimes[index].data(), (int)gpuTimes[index].size(),
                             -INFINITY, 1.0, 0.0, 0, 0);
        }
        ImPlot::PopStyleVar();

        if(ImPlot::IsPlotHovered())
        {
          ImPlotPoint        mouse       = ImPlot::GetPlotMousePos();
          int                mouseOffset = (int(mouse.x)) % (int)gpuTimes[0].size();
          std::vector<float> localTimes(m_node.child.size());
          ImGui::BeginTooltip();

          ImGui::Text("CPU: %.3f ms", cpuTimes[mouseOffset]);

          float totalGpu = 0.f;
          for(size_t i = 0; i < m_node.child.size(); i++)
          {
            if(i == 0)
            {
              localTimes[i] = gpuTimes[i][mouseOffset];
            }
            else
            {
              localTimes[i] = gpuTimes[i][mouseOffset] - gpuTimes[i - 1][mouseOffset];
            }
            totalGpu += localTimes[i];
          }
          ImGui::Text("GPU: %.3f ms", totalGpu);
          for(size_t i = 0; i < m_node.child.size(); i++)
          {
            ImGui::Text("  %s: %.3f ms (%.1f%%)", m_node.child[i].name.c_str(), localTimes[i], localTimes[i] * 100.f / totalGpu);
          }

          ImGui::EndTooltip();
        }

        ImPlot::EndPlot();
      }
    }
  }

  // This goes in the .ini file and remember the state of the window [open/close]
  void addSettingsHandler()
  {
    // Persisting the window
    ImGuiSettingsHandler iniHandler{};
    iniHandler.TypeName   = "ElementProfiler";
    iniHandler.TypeHash   = ImHashStr("ElementProfiler");
    iniHandler.ClearAllFn = [](ImGuiContext* ctx, ImGuiSettingsHandler*) {};
    iniHandler.ApplyAllFn = [](ImGuiContext* ctx, ImGuiSettingsHandler*) {};
    iniHandler.ReadOpenFn = [](ImGuiContext*, ImGuiSettingsHandler*, const char* name) -> void* { return (void*)1; };
    iniHandler.ReadLineFn = [](ImGuiContext*, ImGuiSettingsHandler* handler, void* entry, const char* line) {
      ElementProfiler* s = (ElementProfiler*)handler->UserData;
      int              x;
      if(sscanf(line, "ShowWindow=%d", &x) == 1)
      {
        s->m_showWindow = (x == 1);
      }
    };
    iniHandler.WriteAllFn = [](ImGuiContext* ctx, ImGuiSettingsHandler* handler, ImGuiTextBuffer* buf) {
      ElementProfiler* s = (ElementProfiler*)handler->UserData;
      buf->appendf("[%s][State]\n", handler->TypeName);
      buf->appendf("ShowWindow=%d\n", s->m_showWindow ? 1 : 0);
      buf->appendf("\n");
    };
    iniHandler.UserData = this;
    ImGui::AddSettingsHandler(&iniHandler);
  }


  //---
  Application* m_app{nullptr};
  MyEntryNode  m_node;
  MyEntryNode  m_single;
  bool         m_showWindow = true;
};

}  // namespace nvvkhl
