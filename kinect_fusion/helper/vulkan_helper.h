#pragma once

#include "glm/glm.hpp"

#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>

#include <list>
#include <iostream>

struct VulkanMemory {
    vk::BufferCreateInfo _bufferCreateInfo;
    vk::DescriptorBufferInfo _descriptorBuffer;
    vk::raii::Buffer _buffer;

    vk::raii::DeviceMemory _deviceMemory;

    VulkanMemory(
        const size_t size,
        const vk::raii::Device& device,
        const uint32_t* compute_indeces,
        const uint32_t MemoryTypeIndex) : _bufferCreateInfo(
            vk::BufferCreateFlags(), size, vk::BufferUsageFlagBits::eStorageBuffer, vk::SharingMode::eExclusive, 1,
            compute_indeces), _buffer(device.createBuffer(_bufferCreateInfo)), _deviceMemory(nullptr) {
        vk::MemoryRequirements MemReqs = _buffer.getMemoryRequirements();
        vk::MemoryAllocateInfo MemoryAllocateInfo(MemReqs.size, MemoryTypeIndex);

        _deviceMemory = vk::raii::DeviceMemory(device, MemoryAllocateInfo);

        _buffer.bindMemory(*_deviceMemory, 0);

        _descriptorBuffer =  vk::DescriptorBufferInfo(*_buffer, 0, _bufferCreateInfo.size);
    }

    void fillWith(const void * data) const {
        auto *mapped = _deviceMemory.mapMemory(0, _bufferCreateInfo.size);
        memcpy(mapped, data, _bufferCreateInfo.size);
        _deviceMemory.unmapMemory();
    }

    template <typename T>
    struct MapProxy {
        const vk::raii::DeviceMemory& _mem;
        T* data;

        MapProxy(vk::raii::DeviceMemory& mem, uint32_t size): _mem(mem) {
            data = (T*) _mem.mapMemory(0, size);
        }

        ~MapProxy() {
            _mem.unmapMemory();
        }

        T& operator[](uint32_t index) {
            return data[index];
        }
    };

    template <typename T>
    MapProxy<T> map() {
        return MapProxy<T>(_deviceMemory, _bufferCreateInfo.size);
    }
};

struct VulkanWrapper {
    vk::raii::Context _vk_context;
    vk::raii::Instance _instance;
    vk::raii::Device _device;
    uint32_t ComputeQueueFamilyIndex;
    uint32_t MemoryTypeIndex;

    std::list<VulkanMemory> _buffers;

    vk::raii::DescriptorSetLayout _descriptor_set_layout;
    std::vector<vk::DescriptorSetLayoutBinding> _layout_bindings;

    vk::raii::PipelineLayout _pipelineLayout;
    vk::raii::PipelineCache _pipelineCache;
    vk::raii::DescriptorPool _descriptorPool;

    std::vector<vk::raii::DescriptorSet> _descriptorSets;

    vk::raii::CommandPool _commandPool;
    std::vector<vk::raii::CommandBuffer> _cmdBuffers;

    vk::raii::ShaderModule _shaderModule;
    vk::raii::Pipeline _pipeline;

    vk::raii::Queue _queue;
    vk::raii::Fence _fence;

    explicit VulkanWrapper(const char * ApplicationName): _instance(nullptr), _device(nullptr),
                                                          _buffers(), _descriptor_set_layout(nullptr),
                                                          _layout_bindings(), _pipelineLayout(nullptr), _pipelineCache(nullptr),
                                                          _descriptorPool(nullptr), _descriptorSets(), _commandPool(nullptr),
                                                          _shaderModule(nullptr),
                                                          _pipeline(nullptr),
                                                          _queue(nullptr),
                                                          _fence(nullptr) {
        vk::ApplicationInfo AppInfo{ApplicationName, 1, nullptr, 0,VK_API_VERSION_1_3};

        const std::vector<const char *> Layers = {
#ifndef _NDEBUG
            "VK_LAYER_KHRONOS_validation"
#endif
        };

        const vk::InstanceCreateInfo InstanceCreateInfo(vk::InstanceCreateFlags(), &AppInfo, Layers, {});
        _instance = vk::raii::Instance(_vk_context, InstanceCreateInfo);

        vk::raii::PhysicalDevice PhysicalDevice = _instance.enumeratePhysicalDevices().front();
        vk::PhysicalDeviceProperties DeviceProps = PhysicalDevice.getProperties();

        std::cout << "Device Name    : " << DeviceProps.deviceName << std::endl;
        const uint32_t ApiVersion = DeviceProps.apiVersion;
        std::cout << "Vulkan Version : " << VK_VERSION_MAJOR(ApiVersion) << "." << VK_VERSION_MINOR(ApiVersion) << "."
                << VK_VERSION_PATCH(ApiVersion) << std::endl;

        std::vector<vk::QueueFamilyProperties> QueueFamilyProps = PhysicalDevice.getQueueFamilyProperties();
        auto PropIt = std::find_if(QueueFamilyProps.begin(), QueueFamilyProps.end(),
                                   [](const vk::QueueFamilyProperties &Prop) {
                                       return Prop.queueFlags & vk::QueueFlagBits::eCompute;
                                   });
        ComputeQueueFamilyIndex = std::distance(QueueFamilyProps.begin(), PropIt);
        std::cout << "Compute Queue Family Index: " << ComputeQueueFamilyIndex << std::endl;

        constexpr float QueuePriority = 1.0f;
        vk::DeviceQueueCreateInfo DeviceQueueCreateInfo(vk::DeviceQueueCreateFlags(), // Flags
                                                        ComputeQueueFamilyIndex, // Queue Family Index
                                                        1, // Number of Queues
                                                        &QueuePriority);

        vk::PhysicalDeviceFeatures requestedFeatures = {};

        vk::DeviceCreateInfo DeviceCreateInfo(vk::DeviceCreateFlags(), // Flags
                                              DeviceQueueCreateInfo); // Device Queue Create Info struct
        DeviceCreateInfo.pEnabledFeatures = &requestedFeatures;

        _device = PhysicalDevice.createDevice(DeviceCreateInfo);

        vk::PhysicalDeviceMemoryProperties MemoryProperties = PhysicalDevice.getMemoryProperties();

        MemoryTypeIndex = uint32_t(~0);
        vk::DeviceSize MemoryHeapSize = uint32_t(~0);
        for (uint32_t CurrentMemoryTypeIndex = 0; CurrentMemoryTypeIndex < MemoryProperties.memoryTypeCount; ++
             CurrentMemoryTypeIndex) {
            vk::MemoryType MemoryType = MemoryProperties.memoryTypes[CurrentMemoryTypeIndex];
            if ((vk::MemoryPropertyFlagBits::eHostVisible & MemoryType.propertyFlags) &&
                (vk::MemoryPropertyFlagBits::eHostCoherent & MemoryType.propertyFlags)) {
                MemoryHeapSize = MemoryProperties.memoryHeaps[MemoryType.heapIndex].size;
                MemoryTypeIndex = CurrentMemoryTypeIndex;
                break;
            }
        }

        std::cout << "Memory Type Index: " << MemoryTypeIndex << std::endl;
        std::cout << "Memory Heap Size : " << MemoryHeapSize / 1024 / 1024 / 1024 << " GB" << std::endl;

        _queue = _device.getQueue(ComputeQueueFamilyIndex, 0);
        _fence = _device.createFence(vk::FenceCreateFlags());
    }

    VulkanMemory& addBuffer(size_t size) {
        auto& Buffer = _buffers.emplace_back(size, _device, &ComputeQueueFamilyIndex, MemoryTypeIndex);

        _layout_bindings.emplace_back(
            _layout_bindings.size(),
            vk::DescriptorType::eStorageBuffer,
            1,
            vk::ShaderStageFlagBits::eCompute
        );

        return Buffer;
    }


    void createPipeline(
        const uint32_t constant_size,
        const std::vector<uint32_t>& shader_code,
        const char* entry_point) {
        vk::DescriptorSetLayoutCreateInfo DescriptorSetLayoutCreateInfo(
                vk::DescriptorSetLayoutCreateFlags(),
                _layout_bindings);

        _descriptor_set_layout = _device.createDescriptorSetLayout(DescriptorSetLayoutCreateInfo);
        vk::PipelineLayoutCreateInfo PipelineLayoutCreateInfo(vk::PipelineLayoutCreateFlags(), *_descriptor_set_layout);

        vk::PushConstantRange pushConstantRange{
            vk::ShaderStageFlagBits::eCompute,
            0,
            constant_size};
        PipelineLayoutCreateInfo.setPushConstantRanges(pushConstantRange);

        _pipelineLayout = _device.createPipelineLayout(PipelineLayoutCreateInfo);
        _pipelineCache = _device.createPipelineCache(vk::PipelineCacheCreateInfo());

        vk::DescriptorPoolSize
                DescriptorPoolSize(vk::DescriptorType::eStorageBuffer,  _layout_bindings.size());
        vk::DescriptorPoolCreateInfo DescriptorPoolCreateInfo(
                vk::DescriptorPoolCreateFlags() | vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, 1,
                DescriptorPoolSize);
        _descriptorPool = _device.createDescriptorPool(DescriptorPoolCreateInfo);

        vk::DescriptorSetAllocateInfo DescriptorSetAllocInfo(*_descriptorPool, *_descriptor_set_layout);
        _descriptorSets = _device.allocateDescriptorSets(DescriptorSetAllocInfo);

        std::vector<vk::WriteDescriptorSet> WriteDescriptorSets;

        size_t index = 0;
        for (auto& Buffer: _buffers) {
            WriteDescriptorSets.emplace_back(
            *_descriptorSets[0], index++, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &Buffer._descriptorBuffer
            );
        }
        _device.updateDescriptorSets(WriteDescriptorSets, {});

        vk::CommandPoolCreateInfo CommandPoolCreateInfo(vk::CommandPoolCreateFlags(), ComputeQueueFamilyIndex);
        _commandPool = _device.createCommandPool(CommandPoolCreateInfo);

        vk::CommandBufferAllocateInfo CommandBufferAllocInfo(*_commandPool,vk::CommandBufferLevel::ePrimary,1);
        _cmdBuffers = _device.allocateCommandBuffers(CommandBufferAllocInfo);

        vk::ShaderModuleCreateInfo ShaderModuleCreateInfo(vk::ShaderModuleCreateFlags(), shader_code.size() * sizeof(shader_code[0]), shader_code.data());
        _shaderModule = _device.createShaderModule(ShaderModuleCreateInfo);

        vk::PipelineShaderStageCreateInfo PipelineShaderCreateInfo(vk::PipelineShaderStageCreateFlags(), vk::ShaderStageFlagBits::eCompute, *_shaderModule, entry_point);
        vk::ComputePipelineCreateInfo ComputePipelineCreateInfo(vk::PipelineCreateFlags(), PipelineShaderCreateInfo, *_pipelineLayout);
        _pipeline = _device.createComputePipeline(_pipelineCache, ComputePipelineCreateInfo);
    }

    vk::raii::CommandBuffer& startCommandBuffer() {
        vk::CommandBufferBeginInfo CmdBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);

        _cmdBuffers[0].begin(CmdBufferBeginInfo);
        _cmdBuffers[0].bindPipeline(vk::PipelineBindPoint::eCompute, *_pipeline);
        _cmdBuffers[0].bindDescriptorSets(vk::PipelineBindPoint::eCompute, *_pipelineLayout,  0, {*_descriptorSets.front()}, {});

        return _cmdBuffers[0];
    }

    template <typename T>
    void addCommandPushConstants(T t) {
        _cmdBuffers[0].pushConstants<T>(*_pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, t);
    }

    void submitAndWait(uint32_t sx, uint32_t sy, uint32_t sz) {
        _cmdBuffers[0].dispatch(sx, sy, sz);
        _cmdBuffers[0].end();

        vk::SubmitInfo SubmitInfo(nullptr, nullptr,*_cmdBuffers[0]);
        _queue.submit({SubmitInfo}, *_fence);

        auto result = _device.waitForFences({*_fence},  true,uint64_t(-1));

        _commandPool.reset();
        _device.resetFences(*_fence);
    }
};