#include "vkexp/compute/HeadlessComputeContext.hpp"
#include "vkexp/neuro/NeuralNetwork.hpp"
#include "vkexp/simulation/AgentTypes.hpp"
#include "vkexp/simulation/CpuSimulation.hpp"
#include "vkexp/worlds/WorldScenario.hpp"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <vector>

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

void runNeuralStepParity(
    vkexp::HeadlessComputeContext& context, const vkexp::WorldShape worldShape,
    const vkexp::BeaconScenario beaconScenario = vkexp::BeaconScenario::Stationary) {
    vkexp::neuro::Weights weights{};
    for (std::size_t index = 0; index < weights.size(); ++index) {
        weights[index] = std::sin(static_cast<float>(index) * 0.37F) * 0.31F;
    }
    vkexp::AgentState initial{};
    initial.pose = {1.827F, 0.03F, 0.0F, 0.012F};
    initial.motion = {2.0F, -1.0F, 9.0F, 1.0F};
    initial.signal = {0.1F, 0.2F, 0.3F, 0.0F};
    initial.target = {0.62F, 0.41F, 0.0F, 0.0F};
    vkexp::SimulationStep settings{};
    settings.worldShape = worldShape;
    settings.beaconScenario = beaconScenario;
    if (beaconScenario == vkexp::BeaconScenario::Rotating) {
        settings.beaconRotationAngle = 0.73F;
    } else if (beaconScenario == vkexp::BeaconScenario::RandomMovement) {
        settings.beaconMotionTime = 4.2F;
        settings.beaconMotionSeed = 73U;
    } else if (beaconScenario == vkexp::BeaconScenario::ForageHome) {
        settings.beaconRotationAngle = 0.73F;
        settings.beaconMotionTime = vkexp::forageHomeRelocationSeconds;
        initial.internal = {0.8F, 1.0F, 0.2F, -0.3F};
        const vkexp::Float4 home = vkexp::homeBeaconPosition(initial, settings);
        initial.pose.x = home.x;
        initial.pose.y = home.y;
    }
    vkexp::SimulationStep initialSettings = settings;
    initialSettings.beaconPhase = 0;
    initialSettings.beaconRotationAngle = 0.0F;
    initialSettings.beaconMotionTime = 0.0F;
    const float initialDistance = vkexp::nearestBeaconDistance(initial, initialSettings);
    initial.metrics = {initialDistance, initialDistance, 0.0F, 0.0F};
    if (beaconScenario == vkexp::BeaconScenario::AlternatingDiagonals) {
        settings.beaconPhase = 1;
        settings.beaconPhaseChanged = true;
    }
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

    vkexp::BufferResource inputAgents;
    inputAgents.create(
        context.physicalDevice(), context.device(),
        {sizeof(initial), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT});
    vkexp::BufferResource outputAgents;
    outputAgents.create(
        context.physicalDevice(), context.device(),
        {sizeof(initial), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT});
    vkexp::BufferResource genomes;
    genomes.create(
        context.physicalDevice(), context.device(),
        {sizeof(weights), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT});
    constexpr float gridCellSize = 0.12F;
    const std::uint32_t gridWidth =
        static_cast<std::uint32_t>(std::ceil(settings.worldRadius * 2.0F / gridCellSize));
    const std::uint32_t gridCells = gridWidth * gridWidth;
    std::vector<std::int32_t> heads(gridCells, -1);
    const auto coordinate = [&](const float position) {
        return static_cast<std::uint32_t>(
            std::clamp(std::floor((position + settings.worldRadius) / gridCellSize), 0.0F,
                       static_cast<float>(gridWidth - 1)));
    };
    heads[coordinate(initial.pose.y) * gridWidth + coordinate(initial.pose.x)] = 0;
    const std::int32_t next = -1;
    vkexp::BufferResource gridHeads;
    gridHeads.create(context.physicalDevice(), context.device(),
                     {heads.size() * sizeof(std::int32_t),
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT});
    vkexp::BufferResource gridNext;
    gridNext.create(
        context.physicalDevice(), context.device(),
        {sizeof(next), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT});
    context.immediate().uploadBuffer(inputAgents, &initial, sizeof(initial));
    context.immediate().uploadBuffer(genomes, weights.data(), sizeof(weights));
    context.immediate().uploadBuffer(gridHeads, heads.data(), heads.size() * sizeof(std::int32_t));
    context.immediate().uploadBuffer(gridNext, &next, sizeof(next));

    std::array<VkDescriptorSetLayoutBinding, 5> bindings{};
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
                                           {1, {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 5}}}};
    const VkDescriptorSet descriptorSet = descriptors.allocate(setLayout.get());
    vkexp::DescriptorSetWriter{}
        .writeBuffer(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, inputAgents.buffer(), 0,
                     inputAgents.size())
        .writeBuffer(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, outputAgents.buffer(), 0,
                     outputAgents.size())
        .writeBuffer(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, genomes.buffer(), 0, genomes.size())
        .writeBuffer(3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, gridHeads.buffer(), 0, gridHeads.size())
        .writeBuffer(4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, gridNext.buffer(), 0, gridNext.size())
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
        vkexp::beaconArrivalRadius(settings),
        settings.maximumSpeed,
        settings.maximumAngularSpeed,
        settings.lightSensorRange,
        settings.lightExposure,
        settings.collisionRestitution,
        settings.collisionStiffness,
        gridCellSize,
        settings.wallCollisionPenalty,
        1,
        vkexp::neuro::packBrainLayout(vkexp::scenarioDefinition(settings.beaconScenario).brain),
        1,
        static_cast<std::uint32_t>(worldShape),
        gridWidth,
        gridCells,
        settings.agentCollisionsEnabled ? 1U : 0U,
        settings.agentLightEnabled ? 1U : 0U,
        static_cast<std::uint32_t>(settings.beaconScenario),
        settings.beaconPhase,
        settings.beaconPhaseChanged ? 1U : 0U,
        settings.beaconScenario == vkexp::BeaconScenario::RandomMovement
            ? settings.beaconRandomSpeed
            : settings.beaconRotationAngle,
        settings.beaconRadiusRatio,
        settings.beaconMotionTime,
        settings.beaconScenario == vkexp::BeaconScenario::ForageHome
            ? settings.forageCargoDecayRate
            : settings.beaconTeleportProbability,
        settings.beaconMotionSeed};
    context.immediate().execute([&](const VkCommandBuffer commands) {
        vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline());
        vkCmdBindDescriptorSets(commands, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.layout(), 0, 1,
                                &descriptorSet, 0, nullptr);
        vkCmdPushConstants(commands, pipeline.layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(parameters), &parameters);
        vkCmdDispatch(commands, 1, 1, 1);
        vkexp::cmdBufferBarrier(commands, outputAgents.buffer(),
                                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                                VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
    });
    vkexp::AgentState actual{};
    context.immediate().readbackBuffer(outputAgents, &actual, sizeof(actual));
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

void runAgentInteractionTest(vkexp::HeadlessComputeContext& context, const bool isolatedWorlds) {
    constexpr std::size_t agentCount = 2;
    std::array<vkexp::AgentState, agentCount> initial{};
    initial[0].pose = {-0.01F, 0.0F, 0.0F, 0.022F};
    initial[1].pose = {0.01F, 0.0F, 3.14159265F, 0.022F};
    for (std::size_t index = 0; index < agentCount; ++index) {
        initial[index].motion.w = 1.0F;
        initial[index].target = {-1.0F, 0.0F, 0.0F, static_cast<float>(index)};
        initial[index].metrics = {1.0F, 1.0F, 0.0F, 0.0F};
        initial[index].penalties.w = isolatedWorlds ? static_cast<float>(index) : 0.0F;
    }
    initial[1].signal = {1.0F, 0.0F, 0.0F, 1.0F};

    std::array<vkexp::neuro::Weights, agentCount> genomes{};
    constexpr std::size_t centerRedInput = 3 * vkexp::neuro::Topology::lightChannelsPerReceptor;
    constexpr vkexp::neuro::BrainShape brain{48, 20, 6};
    constexpr std::size_t outputWeights = brain.inputCount * brain.hiddenCount + brain.hiddenCount;
    genomes[0][centerRedInput] = 4.0F;
    genomes[0][outputWeights + 2 * brain.hiddenCount] = 4.0F;

    vkexp::BufferResource inputAgents;
    vkexp::BufferResource outputAgents;
    vkexp::BufferResource genomeBuffer;
    vkexp::BufferResource gridHeads;
    vkexp::BufferResource gridNext;
    const VkDeviceSize agentBytes = sizeof(initial);
    inputAgents.create(
        context.physicalDevice(), context.device(),
        {agentBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT});
    outputAgents.create(
        context.physicalDevice(), context.device(),
        {agentBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT});
    genomeBuffer.create(
        context.physicalDevice(), context.device(),
        {sizeof(genomes), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT});
    constexpr float cellSize = 0.12F;
    const vkexp::SimulationStep settings{};
    const std::uint32_t gridWidth =
        static_cast<std::uint32_t>(std::ceil(settings.worldRadius * 2.0F / cellSize));
    const std::uint32_t gridCells = gridWidth * gridWidth;
    const std::uint32_t worldCount = isolatedWorlds ? 2U : 1U;
    std::vector<std::int32_t> heads(gridCells * worldCount, -1);
    const std::uint32_t center = gridWidth / 2;
    const std::uint32_t centerCell = center * gridWidth + center;
    heads[centerCell] = isolatedWorlds ? 0 : 1;
    if (isolatedWorlds) {
        heads[gridCells + centerCell] = 1;
    }
    const std::array<std::int32_t, agentCount> links{-1, isolatedWorlds ? -1 : 0};
    gridHeads.create(context.physicalDevice(), context.device(),
                     {heads.size() * sizeof(std::int32_t),
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT});
    gridNext.create(
        context.physicalDevice(), context.device(),
        {sizeof(links), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT});
    context.immediate().uploadBuffer(inputAgents, initial.data(), agentBytes);
    context.immediate().uploadBuffer(genomeBuffer, genomes.data(), sizeof(genomes));
    context.immediate().uploadBuffer(gridHeads, heads.data(), heads.size() * sizeof(std::int32_t));
    context.immediate().uploadBuffer(gridNext, links.data(), sizeof(links));

    std::array<VkDescriptorSetLayoutBinding, 5> bindings{};
    for (std::uint32_t binding = 0; binding < bindings.size(); ++binding) {
        bindings[binding].binding = binding;
        bindings[binding].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[binding].descriptorCount = 1;
        bindings[binding].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layoutInfo.bindingCount = static_cast<std::uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();
    vkexp::UniqueDescriptorSetLayout layout;
    if (vkCreateDescriptorSetLayout(context.device(), &layoutInfo, nullptr,
                                    layout.put(context.device())) != VK_SUCCESS) {
        throw std::runtime_error("Unable to create interaction test descriptor layout");
    }
    vkexp::DescriptorAllocator descriptors{context.device(),
                                           {1, {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 5}}}};
    const VkDescriptorSet set = descriptors.allocate(layout.get());
    vkexp::DescriptorSetWriter{}
        .writeBuffer(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, inputAgents.buffer(), 0,
                     inputAgents.size())
        .writeBuffer(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, outputAgents.buffer(), 0,
                     outputAgents.size())
        .writeBuffer(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, genomeBuffer.buffer(), 0,
                     genomeBuffer.size())
        .writeBuffer(3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, gridHeads.buffer(), 0, gridHeads.size())
        .writeBuffer(4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, gridNext.buffer(), 0, gridNext.size())
        .update(context.device(), set);
    const vkexp::ComputePipeline pipeline =
        vkexp::ComputePipelineBuilder{context.physicalDevice(), context.device()}
            .shader(VKEXP_SHADER_DIR "/agent_step.comp.spv")
            .addDescriptorSetLayout(layout.get())
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
        vkexp::beaconArrivalRadius(settings),
        settings.maximumSpeed,
        settings.maximumAngularSpeed,
        settings.lightSensorRange,
        settings.lightExposure,
        settings.collisionRestitution,
        settings.collisionStiffness,
        cellSize,
        settings.wallCollisionPenalty,
        static_cast<std::uint32_t>(agentCount),
        vkexp::neuro::packBrainLayout(vkexp::scenarioDefinition(settings.beaconScenario).brain),
        1,
        static_cast<std::uint32_t>(settings.worldShape),
        gridWidth,
        gridCells,
        settings.agentCollisionsEnabled ? 1U : 0U,
        settings.agentLightEnabled ? 1U : 0U,
        static_cast<std::uint32_t>(settings.beaconScenario),
        settings.beaconPhase,
        settings.beaconPhaseChanged ? 1U : 0U,
        settings.beaconScenario == vkexp::BeaconScenario::RandomMovement
            ? settings.beaconRandomSpeed
            : settings.beaconRotationAngle,
        settings.beaconRadiusRatio,
        settings.beaconMotionTime,
        settings.beaconScenario == vkexp::BeaconScenario::ForageHome
            ? settings.forageCargoDecayRate
            : settings.beaconTeleportProbability,
        settings.beaconMotionSeed};
    context.immediate().execute([&](const VkCommandBuffer commands) {
        vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline());
        vkCmdBindDescriptorSets(commands, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.layout(), 0, 1,
                                &set, 0, nullptr);
        vkCmdPushConstants(commands, pipeline.layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(parameters), &parameters);
        vkCmdDispatch(commands, 1, 1, 1);
        vkexp::cmdBufferBarrier(commands, outputAgents.buffer(),
                                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                                VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
    });
    std::array<vkexp::AgentState, agentCount> result{};
    context.immediate().readbackBuffer(outputAgents, result.data(), sizeof(result));
    const float initialDistance = initial[1].pose.x - initial[0].pose.x;
    const float resultDistance = result[1].pose.x - result[0].pose.x;
    const float firstTouch =
        std::max({result[0].agentTouch0.x, result[0].agentTouch0.y, result[0].agentTouch0.z,
                  result[0].agentTouch0.w, result[0].agentTouch1.x, result[0].agentTouch1.y,
                  result[0].agentTouch1.z, result[0].agentTouch1.w});
    if (isolatedWorlds) {
        if (std::abs(resultDistance - initialDistance) > 0.0001F || firstTouch > 0.0F) {
            throw std::runtime_error("GPU agents interacted across logical world boundaries");
        }
        if (result[0].signal.x > 0.55F) {
            throw std::runtime_error("GPU photoreceptor observed light from another world");
        }
    } else {
        if (resultDistance <= initialDistance || firstTouch <= 0.0F) {
            throw std::runtime_error("GPU agents did not separate and report tactile contact");
        }
        if (result[0].signal.x <= 0.55F) {
            throw std::runtime_error("GPU photoreceptor did not observe another agent's red light");
        }
    }
}

int run() {
    vkexp::HeadlessComputeContext context{{"vkexp compute smoke"}};
    runGameOfLife(context);
    runImageRoundTrip(context);
    runNeuralStepParity(context, vkexp::WorldShape::Circle);
    runNeuralStepParity(context, vkexp::WorldShape::Square);
    runNeuralStepParity(context, vkexp::WorldShape::Circle,
                        vkexp::BeaconScenario::AlternatingDiagonals);
    runNeuralStepParity(context, vkexp::WorldShape::Circle, vkexp::BeaconScenario::Rotating);
    runNeuralStepParity(context, vkexp::WorldShape::Circle, vkexp::BeaconScenario::RandomMovement);
    runNeuralStepParity(context, vkexp::WorldShape::Circle, vkexp::BeaconScenario::ForageHome);
    runAgentInteractionTest(context, false);
    runAgentInteractionTest(context, true);
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
