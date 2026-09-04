# Vulkan Experiment Framework Plan

## Purpose

Build a compact C++20 playground for Vulkan experiments. The program owns one
application loop and composes independent modules for rendering, compute work,
the ImGui interface, and startup presets.

## Architecture

```text
vulkan_compute_boilerplate (composition root)
  -> vkexp_demo
       -> GraphicsModule
       -> ComputeModule
       -> DemoUiModule
  -> vkexp_imgui
       -> ImGuiModule
       -> ProfilerPanel
  -> vkexp_core
       -> Application
       -> Window
       -> VulkanContext
       -> VulkanResource
  -> vkexp_profiling
  -> vkexp_compute
       -> HeadlessComputeContext
       -> ComputePipelineBuilder
       -> DescriptorAllocator / DescriptorSetWriter
       -> ImmediateContext
       -> PingPongBuffer / PingPongImage
```

Every runtime module implements the `Module` lifecycle:

1. `onAttach(AppContext&)` creates long-lived resources.
2. `onFrameBegin(AppContext&, FrameInfo)` starts cross-cutting frame work.
3. `onUpdate(AppContext&, FrameInfo)` advances per-frame state.
4. `onRender(AppContext&, FrameInfo)` records its GPU work.
5. `onFrameEnd(AppContext&, FrameInfo)` completes cross-cutting frame work.
6. `onDetach(AppContext&)` releases resources in reverse order.

The composition root selects modules; the application owns them and guarantees
their lifetime. `AppContext` contains platform services only. Demo modules share
their explicit `DemoState` dependency rather than adding application-specific
fields to the core context. The graphics module publishes an off-screen image;
the demo UI displays it without owning the renderer.

## Initial milestones

- [x] Create the CMake project and source/include layout.
- [x] Add a GLFW application loop and module lifecycle.
- [x] Add a Vulkan context responsible for instance, surface, device, queues,
      swapchain, synchronization, and frame presentation.
- [x] Separate graphics and compute pipeline modules.
- [x] Add an ImGui module with Vulkan/GLFW backends.
- [x] Add named startup presets selected with `--preset`.
- [x] Render experiments into a resizable ImGui viewport texture.
- [x] Run a button-triggered compute blur into a second viewport texture.
- [x] Persist ImGui window layout and load the native window size from a preset.
- [x] Add scoped CPU/GPU profiling, timestamp queries, rolling statistics, and
      a persistent profiler panel.
- [x] Add runtime validation output through `VK_EXT_debug_utils`.
- [x] Add reusable RAII handles and image/shader resources.
- [x] Add a reusable RAII buffer resource with host-visible access.
- [x] Add synchronous staging upload and GPU readback.
- [x] Add a compute pipeline builder and descriptor allocator/writer.
- [x] Add synchronization2 buffer/image barrier helpers.
- [x] Add dispatch group calculation and buffer/image ping-pong resources.
- [x] Validate dispatch, local workgroups, push constants, and storage buffer
      ranges against physical-device limits.
- [x] Add a headless Game of Life smoke test.
- [x] Extract reusable headless Vulkan instance/device/queue setup.
- [x] Add tightly-packed colour image upload and readback.
- [x] Add compute-pipeline specialization constants.
- [x] Prebuild both descriptor sets for buffer and image ping-pong resources.
- [x] Add automated unit and CLI smoke tests.
- [ ] Add shader hot reload.
- [ ] Add growing frame-aware descriptor pools.
- [ ] Add off-screen compute-to-graphics image experiments.
- [ ] Add automated rendering/image-comparison tests.

## Frame flow

```text
poll events -> begin module frame -> update modules -> acquire image
            -> begin timestamp scopes
            -> render off-screen viewport
            -> optionally dispatch compute blur
            -> compose ImGui over background -> submit -> present
```

The compute layer keeps command recording explicit while removing repetitive
resource, descriptor, pipeline, dispatch, and ping-pong plumbing. Experiments
can extend a module or introduce a new one without changing the core loop.

## Presets

Presets are data-only startup configurations. The initial registry contains:

- `graphics`: graphics enabled, compute disabled.
- `compute`: compute enabled, graphics disabled.
- `mixed`: both pipelines enabled (default).

Future presets may also choose shaders, dispatch dimensions, clear colours,
camera state, and module-specific parameters.

## Directory layout

```text
include/vkexp/
  core/       application, context, module, window, Vulkan RAII resources
  demo/       demo state and demo UI
  graphics/   graphics pipeline module
  compute/    reusable compute resources and the demo compute module
  ui/         generic ImGui backend module
  presets/    preset definitions and registry
  profiling/  CPU/GPU scopes, timing history, profiler panel
src/          implementation files mirroring include/vkexp
shaders/      GLSL shader experiments
tests/        CPU unit, CLI, and headless Vulkan smoke tests
```

## Build strategy

CMake finds Vulkan, GLFW, and GLM as system packages and obtains pinned Dear
ImGui with `FetchContent`. Validation layers are enabled in debug builds when
available. Shader targets use `glslangValidator`.
