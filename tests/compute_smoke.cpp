#include "vkexp/compute/HeadlessComputeContext.hpp"
#include "vkexp/neuro/NeuralNetwork.hpp"
#include "vkexp/simulation/AgentTypes.hpp"
#include "vkexp/simulation/CpuSimulation.hpp"

#include <vulkan/vulkan.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>

namespace {

struct GridSize {
    std::uint32_t width;
    std::uint32_t height;
};

void runGameOfLife(vkexp::HeadlessComputeContext& context) {
    constexpr GridSize grid{16, 16};
    constexpr std::size_t cellCount = grid.width * grid.height;
    constexpr VkDeviceSize byteSize = cellCount * sizeof(std::uint32_t);
    std::array<std::uint32_t, cellCount> initial{};
    const std::size_t center = (grid.height / 2) * grid.width + grid.width / 2;
    initial[center - 1] = 1;
    initial[center] = 1;
    initial[center + 1] = 1;

    vkexp::PingPongBuffer state;
    state.create(context.physicalDevice(), context.device(),
                 {byteSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                VK_BUFFER_USAGE_TRANSFER_DST_BIT});
    context.immediate().uploadBuffer(state.read(), initial.data(), byteSize);

    std::array<VkDescriptorSetLayoutBinding, 2> bindings{};
    for (std::uint32_t index = 0; index < bindings.size(); ++index) {
        bindings[index].binding = index;
        bindings[index].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[index].descriptorCount = 1;
        bindings[index].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layoutInfo.bindingCount = static_cast<std::uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();
    vkexp::UniqueDescriptorSetLayout setLayout;
    if (vkCreateDescriptorSetLayout(context.device(), &layoutInfo, nullptr,
                                    setLayout.put(context.device())) != VK_SUCCESS) {
        throw std::runtime_error("Unable to create smoke descriptor set layout");
    }

    vkexp::DescriptorAllocator descriptors{context.device(),
                                           {2, {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4}}}};
    vkexp::PingPongDescriptorSets descriptorSets;
    descriptorSets.createStorageBuffers(context.physicalDevice(), context.device(), descriptors,
                                        setLayout.get(), state);

    constexpr std::uint32_t aliveValue = 1;
    const vkexp::ComputePipeline pipeline =
        vkexp::ComputePipelineBuilder{context.physicalDevice(), context.device()}
            .shader(VKEXP_SHADER_DIR "/game_of_life.comp.spv")
            .addDescriptorSetLayout(setLayout.get())
            .addPushConstantRange(VK_SHADER_STAGE_COMPUTE_BIT, sizeof(GridSize))
            .specializationConstant(0, aliveValue)
            .build();

    context.immediate().execute([&](const VkCommandBuffer commands) {
        vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline());
        const VkPipelineLayout pipelineLayout = pipeline.layout();
        vkCmdPushConstants(commands, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(GridSize), &grid);
        for (int step = 0; step < 2; ++step) {
            const VkDescriptorSet descriptorSet = descriptorSets.current(state);
            vkCmdBindDescriptorSets(commands, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1,
                                    &descriptorSet, 0, nullptr);
            const std::array<VkDeviceSize, 2> storageRanges{state.read().size(),
                                                            state.write().size()};
            const vkexp::DispatchSize groups = vkexp::checkedDispatchSize(
                context.physicalDevice(),
                {{grid.width, grid.height, 1}, {8, 8, 1}, sizeof(GridSize), storageRanges});
            vkCmdDispatch(commands, groups.x, groups.y, groups.z);
            vkexp::cmdComputePingPongBarrier(commands, state);
            state.swap();
        }
    });

    std::array<std::uint32_t, cellCount> result{};
    context.immediate().readbackBuffer(state.read(), result.data(), byteSize);
    if (result != initial) {
        throw std::runtime_error("Two Game of Life GPU steps did not restore the blinker");
    }
}

void runImageRoundTrip(vkexp::HeadlessComputeContext& context) {
    constexpr VkExtent2D extent{4, 4};
    constexpr std::size_t byteCount = extent.width * extent.height * 4;
    std::array<std::uint8_t, byteCount> pixels{};
    for (std::size_t index = 0; index < pixels.size(); ++index) {
        pixels[index] = static_cast<std::uint8_t>(index * 3);
    }

    vkexp::PingPongImage images;
    images.create(context.physicalDevice(), context.device(),
                  {extent, VK_FORMAT_R8G8B8A8_UNORM,
                   VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                       VK_IMAGE_USAGE_TRANSFER_DST_BIT});
    context.immediate().uploadImage(images.read(), pixels.data(), pixels.size());

    std::array<VkDescriptorSetLayoutBinding, 2> bindings{};
    for (std::uint32_t index = 0; index < bindings.size(); ++index) {
        bindings[index].binding = index;
        bindings[index].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        bindings[index].descriptorCount = 1;
        bindings[index].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layoutInfo.bindingCount = static_cast<std::uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();
    vkexp::UniqueDescriptorSetLayout setLayout;
    if (vkCreateDescriptorSetLayout(context.device(), &layoutInfo, nullptr,
                                    setLayout.put(context.device())) != VK_SUCCESS) {
        throw std::runtime_error("Unable to create image descriptor set layout");
    }
    vkexp::DescriptorAllocator descriptors{context.device(),
                                           {2, {{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 4}}}};
    vkexp::PingPongDescriptorSets descriptorSets;
    descriptorSets.createStorageImages(context.device(), descriptors, setLayout.get(), images);
    const VkDescriptorSet firstSet = descriptorSets.current(images);
    images.swap();
    const VkDescriptorSet secondSet = descriptorSets.current(images);
    if (firstSet == secondSet) {
        throw std::runtime_error("Image ping-pong descriptors did not switch sets");
    }
    images.swap();

    std::array<std::uint8_t, byteCount> downloaded{};
    context.immediate().readbackImage(images.read(), downloaded.data(), downloaded.size());
    if (downloaded != pixels) {
        throw std::runtime_error("GPU image upload/readback did not preserve RGBA8 data");
    }
}

void runNeuralStepParity(vkexp::HeadlessComputeContext& context,
                         const vkexp::WorldShape worldShape) {
    vkexp::neuro::Weights weights{};
    for (std::size_t index = 0; index < weights.size(); ++index) {
        weights[index] = std::sin(static_cast<float>(index) * 0.37F) * 0.31F;
    }
    vkexp::AgentState initial{};
    initial.pose = {1.827F, 0.03F, 0.0F, 0.012F};
    initial.motion = {2.0F, -1.0F, 9.0F, 1.0F};
    initial.signal = {0.1F, 0.2F, 0.3F, 0.0F};
    initial.target = {0.62F, 0.41F, 0.0F, 0.0F};
    const float initialDx = initial.target.x - initial.pose.x;
    const float initialDy = initial.target.y - initial.pose.y;
    const float initialDistance = std::sqrt(initialDx * initialDx + initialDy * initialDy);
    initial.metrics = {initialDistance, initialDistance, 0.0F, 0.0F};
    vkexp::SimulationStep settings{};
    settings.worldShape = worldShape;
    vkexp::AgentState expected = initial;
    vkexp::stepAgentCpu(expected, weights, settings);
    const float expectedSpeed =
        std::sqrt(expected.motion.x * expected.motion.x + expected.motion.y * expected.motion.y);
    if (expectedSpeed > settings.maximumSpeed + 0.0001F ||
        std::abs(expected.motion.z) > settings.maximumAngularSpeed + 0.0001F) {
        throw std::runtime_error("CPU agent speed limits were not applied");
    }
    const float maximumCenterDistance = settings.worldRadius - expected.pose.w + 0.0001F;
    const bool insideWorld =
        worldShape == vkexp::WorldShape::Circle
            ? std::hypot(expected.pose.x, expected.pose.y) <= maximumCenterDistance
            : std::abs(expected.pose.x) <= maximumCenterDistance &&
                  std::abs(expected.pose.y) <= maximumCenterDistance;
    if (!insideWorld) {
        throw std::runtime_error("CPU agent escaped the selected world boundary");
    }

    vkexp::BufferResource agents;
    agents.create(context.physicalDevice(), context.device(),
                  {sizeof(initial), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                        VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                        VK_BUFFER_USAGE_TRANSFER_DST_BIT});
    vkexp::BufferResource genomes;
    genomes.create(
        context.physicalDevice(), context.device(),
        {sizeof(weights), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT});
    context.immediate().uploadBuffer(agents, &initial, sizeof(initial));
    context.immediate().uploadBuffer(genomes, weights.data(), sizeof(weights));

    std::array<VkDescriptorSetLayoutBinding, 2> bindings{};
    for (std::uint32_t binding = 0; binding < bindings.size(); ++binding) {
        bindings[binding].binding = binding;
        bindings[binding].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[binding].descriptorCount = 1;
        bindings[binding].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layoutInfo.bindingCount = static_cast<std::uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();
    vkexp::UniqueDescriptorSetLayout setLayout;
    if (vkCreateDescriptorSetLayout(context.device(), &layoutInfo, nullptr,
                                    setLayout.put(context.device())) != VK_SUCCESS) {
        throw std::runtime_error("Unable to create neural parity descriptor layout");
    }
    vkexp::DescriptorAllocator descriptors{context.device(),
                                           {1, {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2}}}};
    const VkDescriptorSet descriptorSet = descriptors.allocate(setLayout.get());
    vkexp::DescriptorSetWriter{}
        .writeBuffer(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, agents.buffer(), 0, agents.size())
        .writeBuffer(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, genomes.buffer(), 0, genomes.size())
        .update(context.device(), descriptorSet);
    const vkexp::ComputePipeline pipeline =
        vkexp::ComputePipelineBuilder{context.physicalDevice(), context.device()}
            .shader(VKEXP_SHADER_DIR "/agent_step.comp.spv")
            .addDescriptorSetLayout(setLayout.get())
            .addPushConstantRange(VK_SHADER_STAGE_COMPUTE_BIT, sizeof(vkexp::GpuStepParameters))
            .build();
    const vkexp::GpuStepParameters parameters{
        settings.deltaTime,
        settings.worldRadius,
        settings.thrust,
        settings.turnAcceleration,
        settings.linearDrag,
        settings.angularDrag,
        settings.sensorFieldOfView,
        settings.arrivalRadius,
        settings.maximumSpeed,
        settings.maximumAngularSpeed,
        0.0F,
        0.0F,
        1,
        static_cast<std::uint32_t>(vkexp::neuro::Topology::weightCount),
        1,
        static_cast<std::uint32_t>(worldShape)};
    context.immediate().execute([&](const VkCommandBuffer commands) {
        vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline());
        vkCmdBindDescriptorSets(commands, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.layout(), 0, 1,
                                &descriptorSet, 0, nullptr);
        vkCmdPushConstants(commands, pipeline.layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(parameters), &parameters);
        vkCmdDispatch(commands, 1, 1, 1);
        vkexp::cmdBufferBarrier(commands, agents.buffer(), VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                                VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
    });
    vkexp::AgentState actual{};
    context.immediate().readbackBuffer(agents, &actual, sizeof(actual));
    const auto close = [](const float a, const float b) { return std::abs(a - b) < 0.0002F; };
    const float* expectedValues = reinterpret_cast<const float*>(&expected);
    const float* actualValues = reinterpret_cast<const float*>(&actual);
    for (std::size_t index = 0; index < sizeof(actual) / sizeof(float); ++index) {
        if (!close(expectedValues[index], actualValues[index])) {
            throw std::runtime_error("CPU/GPU neural step mismatch at float " +
                                     std::to_string(index) + ": expected " +
                                     std::to_string(expectedValues[index]) + ", got " +
                                     std::to_string(actualValues[index]));
        }
    }
}

int run() {
    vkexp::HeadlessComputeContext context{{"vkexp compute smoke"}};
    runGameOfLife(context);
    runImageRoundTrip(context);
    runNeuralStepParity(context, vkexp::WorldShape::Circle);
    runNeuralStepParity(context, vkexp::WorldShape::Square);
    std::cout << "Headless compute and CPU/GPU neural parity tests passed on "
              << context.deviceName() << '\n';
    return 0;
}

} // namespace

int main() {
    try {
        return run();
    } catch (const vkexp::HeadlessComputeUnavailable& error) {
        std::cout << "SKIPPED: " << error.what() << '\n';
        return 77;
    } catch (const std::exception& error) {
        std::cerr << "FAILED: " << error.what() << '\n';
        return 1;
    }
}
