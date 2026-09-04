#include "vkexp/compute/HeadlessComputeContext.hpp"

#include <vulkan/vulkan.h>

#include <array>
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

int run() {
    vkexp::HeadlessComputeContext context{{"vkexp compute smoke"}};
    runGameOfLife(context);
    runImageRoundTrip(context);
    std::cout << "Headless compute smoke test passed on " << context.deviceName() << '\n';
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
