# VRayMtl GLSL

A reference universal GLSL implementation of the VRayMtl material.

The provided implementation is a universal base for implementing VRayMtl shaders in GLSL-based rasteriser engines. Look at the mainImage function for how to set up a VRayMtlInitParams struct, set up a VRayMtlContext and compute the shading for the current fragment. You must implement the functions starting with "eng" according to how your rendering engine provides environment light sampling. You can tune the number of environment samples using `NUM_ENV_SAMPLES` and `NUM_ENV_ROUGH_SAMPLES` to fit your performance needs.

The shader evaluates a single direct light contribution, plus environment reflection and refraction. It does not handle shadowing or screen-space effects, since every rasteriser engine does those differently -- you should adapt the shade() function to your needs.

You can see this shader in action at [https://www.shadertoy.com/view/3dfcRr](https://www.shadertoy.com/view/3dfcRr).

