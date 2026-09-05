// The network preset: how many sensors feed the brain, how wide it is, what its
// outputs mean, and how a flat genome is addressed.
//
// Compiled twice -- as C++ through BrainKernel.hpp and as GLSL through
// shaders/neuro/brain_kernel.glsl -- so both sides build the same network from
// the same declaration. Adding a sensor changes the counts here and nothing
// else: the construction below is written once and derives every offset.
//
// Same common-subset rules as ScenarioKernel.inl: VKEXP_BRAIN_FN in front of
// every function, only uint/bool across boundaries, no standard library. Note
// that GLSL reserves more words than C++ does -- `input`, `output`, `layout`,
// `filter`, `active` and friends cannot be identifiers here.

// --- preset: change these when the sensor suite or brain width changes -------

const uint BrainLightReceptorCount = 7u;
const uint BrainLightChannels = 4u; // RGB + luminance
const uint BrainTactileSectorCount = 8u;
const uint BrainTactileChannels = 2u;     // wall + agent
const uint BrainSelfInputCount = 4u;      // speed, turn rate, energy, own signal
const uint BrainTaskInputCount = 2u;      // cargo level, seeking-home flag
const uint BrainRecurrentCount = 2u;      // memory cells, fed back as inputs
const uint BrainActuatorOutputCount = 6u; // left, right, R, G, B, intensity
const uint BrainHiddenCapacity = 20u;

// --- derived layout: never edited by hand ------------------------------------

const uint BrainLightBlockSize = BrainLightReceptorCount * BrainLightChannels;
const uint BrainTactileBlockSize = BrainTactileSectorCount * BrainTactileChannels;

const uint BrainLightOffset = 0u;
const uint BrainTactileOffset = BrainLightOffset + BrainLightBlockSize;
const uint BrainSelfOffset = BrainTactileOffset + BrainTactileBlockSize;
const uint BrainTaskOffset = BrainSelfOffset + BrainSelfInputCount;
const uint BrainRecurrentInputOffset = BrainTaskOffset + BrainTaskInputCount;
const uint BrainInputCapacity = BrainRecurrentInputOffset + BrainRecurrentCount;

const uint BrainMotorLeftOutput = 0u;
const uint BrainMotorRightOutput = 1u;
const uint BrainSignalColorOutput = 2u; // three consecutive channels
const uint BrainSignalIntensityOutput = 5u;
const uint BrainRecurrentOutputOffset = BrainActuatorOutputCount;
const uint BrainOutputCapacity = BrainActuatorOutputCount + BrainRecurrentCount;

VKEXP_BRAIN_FN uint brainLightChannelIndex(uint receptor, uint channel) {
    return BrainLightOffset + receptor * BrainLightChannels + channel;
}

VKEXP_BRAIN_FN uint brainTactileChannelIndex(uint sector, uint channel) {
    return BrainTactileOffset + sector * BrainTactileChannels + channel;
}

// --- sensor response model ---------------------------------------------------
//
// What a receptor *is*: how sharply it is tuned, how light falls off with range,
// and how RGB collapses to luminance. These use float math, so they carry the
// VKEXP_BRAIN_MATH_FN marker instead -- the C++ side cannot make them constexpr.

const float BrainReceptorSharpness = 12.0f;
const float BrainLightFalloffWidth = 0.25f;
const float BrainDistanceAttenuation = 2.0f;
const float BrainLuminanceRed = 0.2126f;
const float BrainLuminanceGreen = 0.7152f;
const float BrainLuminanceBlue = 0.0722f;

// Smooth cut-off at the edge of sensor range.
VKEXP_BRAIN_MATH_FN float brainRangeFalloff(float distanceToLight, float range) {
    const float fade =
        clamp((range - distanceToLight) / (range * BrainLightFalloffWidth), 0.0f, 1.0f);
    return fade * fade * (3.0f - 2.0f * fade);
}

// Directional tuning of one receptor; `alignment` is a clamped cosine.
VKEXP_BRAIN_MATH_FN float brainReceptorResponse(float alignment) {
    return pow(alignment, BrainReceptorSharpness);
}

VKEXP_BRAIN_MATH_FN float brainDistanceAttenuation(float normalizedDistanceSquared) {
    return 1.0f / (1.0f + normalizedDistanceSquared * BrainDistanceAttenuation);
}

VKEXP_BRAIN_MATH_FN float brainLuminance(float red, float green, float blue) {
    return red * BrainLuminanceRed + green * BrainLuminanceGreen + blue * BrainLuminanceBlue;
}

// --- genome addressing: one dense network laid out flat ----------------------
//
// [input->hidden weights][hidden biases][hidden->output weights][output biases]

VKEXP_BRAIN_FN uint brainWeightCount(uint inputCount, uint hiddenCount, uint outputCount) {
    return inputCount * hiddenCount + hiddenCount + hiddenCount * outputCount + outputCount;
}

VKEXP_BRAIN_FN uint brainHiddenWeightIndex(uint base, uint inputCount, uint neuron,
                                           uint inputIndex) {
    return base + neuron * inputCount + inputIndex;
}

VKEXP_BRAIN_FN uint brainHiddenBiasIndex(uint base, uint inputCount, uint hiddenCount,
                                         uint neuron) {
    return base + inputCount * hiddenCount + neuron;
}

VKEXP_BRAIN_FN uint brainOutputWeightIndex(uint base, uint inputCount, uint hiddenCount,
                                           uint neuron, uint hiddenIndex) {
    return base + inputCount * hiddenCount + hiddenCount + neuron * hiddenCount + hiddenIndex;
}

VKEXP_BRAIN_FN uint brainOutputBiasIndex(uint base, uint inputCount, uint hiddenCount,
                                         uint outputCount, uint neuron) {
    return base + inputCount * hiddenCount + hiddenCount + hiddenCount * outputCount + neuron;
}

// --- active shape packed into one uint for the GPU ---------------------------

const uint BrainStrideMask = 0xfffu;
const uint BrainInputShift = 12u;
const uint BrainHiddenShift = 18u;
const uint BrainOutputShift = 24u;
const uint BrainCountMask = 0x3fu;
const uint BrainOutputCountMask = 0x1fu;

VKEXP_BRAIN_FN uint brainPackLayout(uint genomeStride, uint inputCount, uint hiddenCount,
                                    uint outputCount) {
    return genomeStride | (inputCount << BrainInputShift) | (hiddenCount << BrainHiddenShift) |
           (outputCount << BrainOutputShift);
}

VKEXP_BRAIN_FN uint brainLayoutStride(uint packed) { return packed & BrainStrideMask; }
VKEXP_BRAIN_FN uint brainLayoutInputCount(uint packed) {
    return (packed >> BrainInputShift) & BrainCountMask;
}
VKEXP_BRAIN_FN uint brainLayoutHiddenCount(uint packed) {
    return (packed >> BrainHiddenShift) & BrainCountMask;
}
VKEXP_BRAIN_FN uint brainLayoutOutputCount(uint packed) {
    return (packed >> BrainOutputShift) & BrainOutputCountMask;
}
