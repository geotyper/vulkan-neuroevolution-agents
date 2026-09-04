# Vulkan Neuroevolution Agents

An experimental C++20/Vulkan playground for evolving thousands of small agent
brains on the GPU. The first scenario is deliberately simple: agents use seven
directional photoreceptors to find four beacon positions, while a genetic
algorithm evolves the weights of a fixed dense neural network.

The repository is a working vertical slice and a base for later worlds with
walls, memory, inter-agent light perception, colonies, and richer fitness
functions. Simulation, evolution, visualization, UI, and Vulkan infrastructure
are separate targets rather than one application-specific module.

## Current experiment

- 512 genomes, each evaluated in four trials (2048 GPU agents);
- 7 light receptors plus speed, angular velocity, and energy inputs;
- `10 inputs -> 12 tanh neurons -> 6 outputs`;
- outputs control left/right motors, RGB emission, and emission intensity;
- inertial movement with linear/angular drag and hard linear/angular speed
  limits;
- selectable circular or square world, both twice the original span;
- average fitness across trials rewards approach and arrival and penalizes
  motor use;
- elitism, tournament selection, uniform crossover, Gaussian mutation;
- adjustable simulation speed, generation length, physics, and sensor FOV;
- separate simulation, genetic-algorithm, world, and profiler windows;
- fitness/arrival history graphs for completed generations;
- an off-screen Vulkan renderer with persistent circular bodies, heading
  markers, and independently rendered coloured halos.

Light emission is already part of the genome, agent state, and renderer. Other
agents do not perceive it yet; that will be added with a spatial acceleration
structure instead of an O(N²) scan.

## Architecture

```text
vulkan_neuroevolution_agents (composition root)
  |
  +-- vkneuro_domain          no Vulkan dependency
  |     NeuralNetwork         flattened brain contract + CPU evaluator
  |     Sensors               CPU reference perception
  |     CpuSimulation         CPU reference physics/fitness
  |     GeneticAlgorithm      selection/crossover/mutation
  |
  +-- vkneuro_simulation
  |     SimulationModule      GPU buffers + compute dispatch
  |     agent_step.comp       sensors + brain + physics
  |
  +-- vkneuro_visualization
  |     AgentRenderer         read-only consumer of the agent SSBO
  |
  +-- vkneuro_ui
  |     SimulationUiModule    controls/statistics/viewport only
  |
  +-- reusable infrastructure
        vkexp_core, vkexp_compute, vkexp_profiling, vkexp_imgui
```

The shared contracts are small:

- `SimulationState` carries controls, statistics, the published agent-buffer
  view, and the published viewport image;
- `AgentState` is an explicitly checked 80-byte std430-compatible structure;
- `Topology` defines one flattened 210-float weight layout used identically by
  the CPU evaluator and GLSL shader;
- module order in `main.cpp` is the composition graph: compute publishes the
  buffer, rendering reads it, and ImGui composites the viewport.

This lets a new sensor model, brain evaluator, selection policy, renderer, or
UI replace its counterpart without changing the application lifecycle.

## CPU/GPU correctness

`vkexp_compute_smoke` creates the same agent and genome on CPU and GPU, advances
both by one complete sensor/network/physics step, reads the SSBO back, and
compares every float with a small tolerance. Pure CPU unit tests also cover the
weight layout, neural evaluation, elite preservation, and reusable compute
validation.

## Build and run

Requirements: CMake 3.24+, Ninja, a C++20 compiler, Vulkan 1.3 development
files and driver, GLFW 3.3+, `glslangValidator`, and X11/Wayland for the GUI.
Dear ImGui v1.91.8 is fetched by CMake.

```bash
cmake --preset debug
cmake --build --preset debug
ctest --preset debug --output-on-failure
./build/debug/vulkan_neuroevolution_agents
```

Disable validation if the validation layer is unavailable:

```bash
./build/debug/vulkan_neuroevolution_agents --no-validation
```

The debug suite contains pure unit tests, a CLI smoke test, and a headless
Vulkan test. The latter returns CTest's skip code when no compute device exists.

## Extension points

The next world feature should enter through a focused contract:

- walls and occlusion extend sensor/world queries;
- recurrent state extends `AgentState` and the brain contract;
- inter-agent light uses a separate spatial-index compute pass;
- colony scoring replaces fitness aggregation without changing physics;
- a different topology can become another evaluator/shader pair;
- GPU-side evolution can later replace the synchronous generation boundary.

See [PLAN.md](PLAN.md) for the staged roadmap.
