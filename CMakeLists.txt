cmake_minimum_required(VERSION 3.24)

project(vulkan_compute_boilerplate VERSION 0.3.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(VKEXP_ENABLE_VALIDATION "Enable Vulkan validation layers in debug builds" ON)

include(CTest)

set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS config/window.preset)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/config)
configure_file(config/window.preset
               ${CMAKE_CURRENT_BINARY_DIR}/config/window.preset
               COPYONLY)

find_package(Vulkan 1.3 REQUIRED)
find_package(glfw3 3.3 REQUIRED)
find_package(glm REQUIRED)

include(FetchContent)
FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG v1.91.8
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(imgui)

add_library(imgui STATIC
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp
)
target_include_directories(imgui PUBLIC ${imgui_SOURCE_DIR} ${imgui_SOURCE_DIR}/backends)
target_link_libraries(imgui PUBLIC Vulkan::Vulkan glfw)

find_program(GLSLANG_VALIDATOR glslangValidator REQUIRED)
set(SHADER_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
set(SHADER_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders)

set(SHADER_SOURCES
    triangle.vert
    triangle.frag
    experiment.comp
    game_of_life.comp
)

set(SHADER_BINARIES)
foreach(SHADER ${SHADER_SOURCES})
    set(SPIRV ${SHADER_BINARY_DIR}/${SHADER}.spv)
    add_custom_command(
        OUTPUT ${SPIRV}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADER_BINARY_DIR}
        COMMAND ${GLSLANG_VALIDATOR} -V ${SHADER_SOURCE_DIR}/${SHADER} -o ${SPIRV}
        DEPENDS ${SHADER_SOURCE_DIR}/${SHADER}
        COMMENT "Compiling ${SHADER}"
        VERBATIM
    )
    list(APPEND SHADER_BINARIES ${SPIRV})
endforeach()
add_custom_target(vkexp_shaders ALL DEPENDS ${SHADER_BINARIES})

function(vkexp_enable_warnings target)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4 /permissive-)
    else()
        # Vulkan C structs intentionally use partial aggregate initialization.
        target_compile_options(${target} PRIVATE
            -Wall -Wextra -Wpedantic -Wno-missing-field-initializers
        )
    endif()
endfunction()

add_library(vkexp_profiling STATIC
    src/profiling/CpuProfiler.cpp
    src/profiling/GpuProfiler.cpp
    src/profiling/Profiler.cpp
)
target_include_directories(vkexp_profiling PUBLIC include)
target_link_libraries(vkexp_profiling PUBLIC Vulkan::Vulkan)
vkexp_enable_warnings(vkexp_profiling)

add_library(vkexp_core STATIC
    src/core/Application.cpp
    src/core/VulkanContext.cpp
    src/core/VulkanResource.cpp
    src/core/Window.cpp
)
target_include_directories(vkexp_core PUBLIC include)
target_link_libraries(vkexp_core PUBLIC Vulkan::Vulkan glfw vkexp_profiling)
vkexp_enable_warnings(vkexp_core)

add_library(vkexp_imgui STATIC
    src/ui/ImGuiModule.cpp
    src/profiling/ProfilerPanel.cpp
)
target_include_directories(vkexp_imgui PUBLIC include)
target_link_libraries(vkexp_imgui
    PUBLIC vkexp_core vkexp_profiling
    PRIVATE imgui
)
target_compile_definitions(vkexp_imgui PRIVATE
    VKEXP_IMGUI_INI="${CMAKE_CURRENT_BINARY_DIR}/imgui.ini"
)
vkexp_enable_warnings(vkexp_imgui)

add_library(vkexp_compute STATIC
    src/compute/ComputeResources.cpp
    src/compute/HeadlessComputeContext.cpp
)
target_include_directories(vkexp_compute PUBLIC include)
target_link_libraries(vkexp_compute PUBLIC vkexp_core)
vkexp_enable_warnings(vkexp_compute)

add_library(vkexp_demo STATIC
    src/graphics/GraphicsModule.cpp
    src/compute/ComputeModule.cpp
    src/demo/DemoUiModule.cpp
    src/presets/PresetRegistry.cpp
)
add_dependencies(vkexp_demo vkexp_shaders)
target_include_directories(vkexp_demo PUBLIC include)
target_link_libraries(vkexp_demo
    PUBLIC vkexp_compute vkexp_imgui glm::glm
    PRIVATE imgui
)
target_compile_definitions(vkexp_demo PRIVATE
    VKEXP_SHADER_DIR="${SHADER_BINARY_DIR}"
)
vkexp_enable_warnings(vkexp_demo)

add_executable(vulkan_compute_boilerplate src/main.cpp)
target_link_libraries(vulkan_compute_boilerplate PRIVATE vkexp_demo)
target_compile_definitions(vulkan_compute_boilerplate PRIVATE
    VKEXP_WINDOW_PRESET="${CMAKE_CURRENT_BINARY_DIR}/config/window.preset"
    $<$<AND:$<BOOL:${VKEXP_ENABLE_VALIDATION}>,$<CONFIG:Debug>>:VKEXP_ENABLE_VALIDATION>
)
vkexp_enable_warnings(vulkan_compute_boilerplate)

if(BUILD_TESTING)
    add_executable(vkexp_unit_tests tests/vkexp_unit_tests.cpp)
    target_link_libraries(vkexp_unit_tests PRIVATE vkexp_demo)
    target_compile_definitions(vkexp_unit_tests PRIVATE
        VKEXP_TEST_WINDOW_PRESET="${CMAKE_CURRENT_SOURCE_DIR}/config/window.preset"
    )
    vkexp_enable_warnings(vkexp_unit_tests)

    add_test(NAME vkexp_unit_tests COMMAND vkexp_unit_tests)
    add_test(NAME vkexp_cli_list_presets
             COMMAND vulkan_compute_boilerplate --list-presets)

    add_executable(vkexp_compute_smoke tests/compute_smoke.cpp)
    add_dependencies(vkexp_compute_smoke vkexp_shaders)
    target_link_libraries(vkexp_compute_smoke PRIVATE vkexp_compute)
    target_compile_definitions(vkexp_compute_smoke PRIVATE
        VKEXP_SHADER_DIR="${SHADER_BINARY_DIR}"
    )
    vkexp_enable_warnings(vkexp_compute_smoke)
    add_test(NAME vkexp_compute_smoke COMMAND vkexp_compute_smoke)
    set_tests_properties(vkexp_compute_smoke PROPERTIES SKIP_RETURN_CODE 77)
endif()
