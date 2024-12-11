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

#include "imgui.h"
#include "imgui_internal.h"

namespace nvvkhl {

/** @DOC_START
# class nvvkhl::ElementLogger

>  This class is an element of the application that can redirect all logs to a ImGui window in the application

To use this class, you need to add it to the `nvvkhl::Application` using the `addElement` method.
  
Create the element such that it will be available to the target application

Example:
  ```cpp
  static nvvkhl::SampleAppLog g_logger;
  nvprintSetCallback([](int level, const char* fmt)
  {
    g_logger.addLog("%s", fmt);
  });
  
  app->addElement(std::make_unique<nvvkhl::ElementLogger>(&g_logger, true));  // Add logger window
  ```
  
@DOC_END */

// Helper structure to hold the log
struct SampleAppLog
{
public:
  SampleAppLog() { clear(); }

  void     setLogLevel(uint32_t level) { m_levelFilter = level; }
  uint32_t getLogLevel() { return m_levelFilter; }

  void clear()
  {
    m_buf.clear();
    m_lineOffsets.clear();
    m_lineOffsets.push_back(0);
    m_lineLevels.clear();
  }

  void addLog(uint32_t level, const char* fmt, ...)
  {
    if((m_levelFilter & (1 << level)) == 0)
      return;

    int     old_size = m_buf.size();
    va_list args     = {};
    va_start(args, fmt);
    m_buf.appendfv(fmt, args);
    va_end(args);
    for(int new_size = m_buf.size(); old_size < new_size; old_size++)
    {
      if(m_buf[old_size] == '\n')
      {
        m_lineOffsets.push_back(old_size + 1);
        m_lineLevels.push_back(level);
      }
    }
  }

  void initColors()
  {
    ImGuiContext& g = *GImGui;
    m_colors.resize(8);
    m_colors[LOGLEVEL_INFO]    = g.Style.Colors[ImGuiCol_Text];  // Default text color
    m_colors[LOGLEVEL_WARNING] = ImVec4(1.0, 0.5, 0.0, 1.0);     // Orange
    m_colors[LOGLEVEL_ERROR]   = ImVec4(1.0, 0.0, 0.0, 1.0);     // Bright Red
    m_colors[LOGLEVEL_DEBUG]   = ImVec4(0.5, 0.5, 1.0, 1.0);     // Light Blue
    m_colors[LOGLEVEL_STATS]   = ImVec4(0.0, 0.75, 0.0, 1.0);    // Light Green
    m_colors[LOGLEVEL_OK]      = ImVec4(0.0, 1.0, 0.0, 1.0);     // Green
  }

  void draw(const char* title, bool* p_open = nullptr)
  {
    if(m_colors.empty())  // Initialize colors late, as we need the ImGui context
      initColors();

    if(!ImGui::Begin(title, p_open))
    {
      ImGui::End();
      return;
    }

    // Options menu
    if(ImGui::BeginPopup("Options"))
    {
      ImGui::Checkbox("Auto-scroll", &m_autoScroll);
      ImGui::EndPopup();
    }

    // Main window
    if(ImGui::Button("Options"))
    {
      ImGui::OpenPopup("Options");
    }
    ImGui::SameLine();
    bool do_clear = ImGui::Button("Clear");
    ImGui::SameLine();
    bool copy = ImGui::Button("Copy");
    ImGui::SameLine();
    ImGui::CheckboxFlags("All", &m_levelFilter, LOGBITS_ALL);
    ImGui::SameLine();
    ImGui::CheckboxFlags("Stats", &m_levelFilter, LOGBIT_STATS);
    ImGui::SameLine();
    ImGui::CheckboxFlags("Debug", &m_levelFilter, LOGBIT_DEBUG);
    ImGui::SameLine();
    ImGui::CheckboxFlags("Info", &m_levelFilter, LOGBIT_INFO);
    ImGui::SameLine();
    ImGui::CheckboxFlags("Warnings", &m_levelFilter, LOGBIT_WARNING);
    ImGui::SameLine();
    ImGui::CheckboxFlags("Errors", &m_levelFilter, LOGBIT_ERROR);
    ImGui::SameLine();
    ImGui::Text("Filter");
    ImGui::SameLine();
    m_filter.Draw("##Filter", -100.0F);
    ImGui::SameLine();
    bool clear_filter = ImGui::SmallButton("X");

    ImGui::Separator();
    ImGui::BeginChild("scrolling", ImVec2(0, 0), false, ImGuiWindowFlags_HorizontalScrollbar);

    if(do_clear)
    {
      clear();
    }
    if(copy)
    {
      ImGui::SetClipboardText(m_buf.c_str());
    }
    if(clear_filter)
    {
      m_filter.Clear();
    }


    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0, 0));
    const char* buf     = m_buf.begin();
    const char* buf_end = m_buf.end();
    if(m_filter.IsActive())
    {
      // In this example we don't use the clipper when Filter is enabled.
      // This is because we don't have a random access on the result on our filter.
      // A real application processing logs with ten of thousands of entries may want to store the result of
      // search/filter.. especially if the filtering function is not trivial (e.g. reg-exp).
      for(int line_no = 0; line_no < m_lineOffsets.Size; line_no++)
      {
        const char*   line_start = buf + m_lineOffsets[line_no];
        const char*   line_end = (line_no + 1 < m_lineOffsets.Size) ? (buf + m_lineOffsets[line_no + 1] - 1) : buf_end;
        const int32_t level    = line_no < m_lineLevels.Size ? m_lineLevels[line_no] : 0;
        if(m_filter.PassFilter(line_start, line_end))
        {
          // Setting the color of the line
          ImGuiContext& g               = *GImGui;
          ImVec4        backupTextColor = g.Style.Colors[ImGuiCol_Text];
          g.Style.Colors[ImGuiCol_Text] = m_colors[level];

          ImGui::TextUnformatted(line_start, line_end);

          g.Style.Colors[ImGuiCol_Text] = backupTextColor;  // restore color
        }
      }
    }
    else
    {
      // The simplest and easy way to display the entire buffer:
      //   ImGui::TextUnformatted(buf_begin, buf_end);
      // And it'll just work. TextUnformatted() has specialization for large blob of text and will fast-forward
      // to skip non-visible lines. Here we instead demonstrate using the clipper to only process lines that are
      // within the visible area.
      // If you have tens of thousands of items and their processing cost is non-negligible, coarse clipping them
      // on your side is recommended. Using ImGuiListClipper requires
      // - A) random access into your data
      // - B) items all being the  same height,
      // both of which we can handle since we an array pointing to the beginning of each line of text.
      // When using the filter (in the block of code above) we don't have random access into the data to display
      // anymore, which is why we don't use the clipper. Storing or skimming through the search result would make
      // it possible (and would be recommended if you want to search through tens of thousands of entries).
      ImGuiListClipper clipper;
      clipper.Begin(m_lineOffsets.Size);
      while(clipper.Step())
      {
        for(int line_no = clipper.DisplayStart; line_no < clipper.DisplayEnd; line_no++)
        {
          const char* line_start = buf + m_lineOffsets[line_no];
          const char* line_end = (line_no + 1 < m_lineOffsets.Size) ? (buf + m_lineOffsets[line_no + 1] - 1) : buf_end;
          const int32_t level  = line_no < m_lineLevels.Size ? m_lineLevels[line_no] : 0;

          // Setting the color of the line
          ImGuiContext& g               = *GImGui;
          ImVec4        backupTextColor = g.Style.Colors[ImGuiCol_Text];
          g.Style.Colors[ImGuiCol_Text] = m_colors[level];

          ImGui::TextUnformatted(line_start, line_end);

          g.Style.Colors[ImGuiCol_Text] = backupTextColor;  // restore color
        }
      }
      clipper.End();
    }
    ImGui::PopStyleVar();

    if(m_autoScroll && ImGui::GetScrollY() >= ImGui::GetScrollMaxY())
    {
      ImGui::SetScrollHereY(1.0F);
    }

    ImGui::EndChild();
    ImGui::End();
  }

private:
  ImGuiTextBuffer m_buf{};
  ImGuiTextFilter m_filter{};
  ImVector<int>   m_lineOffsets;       // Index to lines offset. We maintain this with AddLog() calls.
  ImVector<int>   m_lineLevels;        // Log level per line.
  bool            m_autoScroll{true};  // Keep scrolling if already at the bottom.
  uint32_t        m_levelFilter = LOGBITS_WARNINGS;

  ImVector<ImVec4> m_colors;  // Line color based on log level
};


//extern SampleAppLog g_logger;
struct ElementLogger : public nvvkhl::IAppElement
{
  explicit ElementLogger(SampleAppLog* logger, bool show = false)
      : m_showLog(show)
      , m_logger(logger)
  {
    addSettingsHandler();
  }

  virtual ~ElementLogger() = default;

  void onUIRender() override
  {
    if(ImGui::IsKeyDown(ImGuiKey_ModCtrl) && ImGui::IsKeyDown(ImGuiKey_ModShift) && !ImGui::IsKeyDown(ImGuiKey_ModAlt))
    {
      if(ImGui::IsKeyPressed(ImGuiKey_L))
      {
        m_showLog = !m_showLog;
      }
    }

    if(!m_showLog)
    {
      return;
    }

    ImGui::SetNextWindowCollapsed(false, ImGuiCond_Appearing);
    ImGui::SetNextWindowSize({400, 200}, ImGuiCond_Appearing);
    ImGui::SetNextWindowBgAlpha(0.7F);
    m_logger->draw("Log", &m_showLog);

  }  // Called for anything related to UI

  void onUIMenu() override
  {
    if(ImGui::BeginMenu("View"))
    {
      ImGui::MenuItem("Log Window", "Ctrl+Shift+L", &m_showLog);
      ImGui::EndMenu();
    }
  }  // This is the menubar to create

  // This goes in the .ini file and remember the state of the window [open/close]
  void addSettingsHandler()
  {
    // Persisting the window
    ImGuiSettingsHandler ini_handler{};
    ini_handler.TypeName   = "LoggerEngine";
    ini_handler.TypeHash   = ImHashStr("LoggerEngine");
    ini_handler.ClearAllFn = [](ImGuiContext* ctx, ImGuiSettingsHandler*) {};
    ini_handler.ApplyAllFn = [](ImGuiContext* ctx, ImGuiSettingsHandler*) {};
    ini_handler.ReadOpenFn = [](ImGuiContext*, ImGuiSettingsHandler*, const char* name) -> void* { return (void*)1; };
    ini_handler.ReadLineFn = [](ImGuiContext*, ImGuiSettingsHandler* handler, void* entry, const char* line) {
      ElementLogger* s = (ElementLogger*)handler->UserData;
      int            x;
      if(sscanf(line, "ShowLoader=%d", &x) == 1)
      {
        s->m_showLog = (x == 1);
      }
      else if(sscanf(line, "Level=%d", &x) == 1)
      {
        s->m_logger->setLogLevel(x);
      }
    };
    ini_handler.WriteAllFn = [](ImGuiContext* ctx, ImGuiSettingsHandler* handler, ImGuiTextBuffer* buf) {
      ElementLogger* s = (ElementLogger*)handler->UserData;
      buf->appendf("[%s][State]\n", handler->TypeName);
      buf->appendf("ShowLoader=%d\n", s->m_showLog ? 1 : 0);
      buf->appendf("Level=%d\n", s->m_logger->getLogLevel());
      buf->appendf("\n");
    };
    ini_handler.UserData = this;
    ImGui::AddSettingsHandler(&ini_handler);
  }

private:
  bool          m_showLog{false};
  SampleAppLog* m_logger{nullptr};
};

}  // namespace nvvkhl
