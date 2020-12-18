# V-Ray Material GLSL Implementation #

## General Information ##

vraymtl.glsl contains a GLSL implementation of the V-Ray 5 VRayMtl done in a single fragment shader pass.
This implementation can be run directly in either shadertoy.com or Visual Studio Code using the ShaderToy extension.

## Quality and Performance ##

This implementation is intended as a fast and good approximation of the full-featured VRayMtl in V-Ray. It is used as a base for multiple viewport/preview integrations of VRayMtl:
- VRayMtl in Maya's Viewport 2.0
- VRayMtl in Mari 4.6+
- VRayMtl in Substance Painter (TODO: version)

Note that all of the material's layers are computed for each fragment and indirect lighting can require a lot of samples for a clean result. This might be too heavy for some realtime applications and can be either omitted or replaced with a faster approximation.

## Engine specific functions and inputs ##

Look at the mainImage function for how to set up a VRayMtlInitParams struct, set up a VRayMtlContext and compute the shading for the current fragment. You must implement the functions starting with "eng" according to how your rendering engine provides environment light sampling. You can tune the number of environment samples using `NUM_ENV_SAMPLES` and `NUM_ENV_ROUGH_SAMPLES` to fit your performance needs.

## Supported features ##

The implementation supports the following major features:
- All of the layers (diffuse, reflection, refraction, sheen, coat) in V-Ray 5's VRayMtl are supported and react to direct lighting (from lights such as point or directional) and indirect lighting (from environment)
- Glossy reflections and refractions
- Metalness
- Anisotropic reflections

## Unsupported features ##

- Glossy Fresnel (supported only in sheen) because the calculations are a bit heavy for real-time rendering.
- SSS and fog parameters
- Normal/Bump Mapping - this is outside of the scope of the current implementation because the normal is usually computed separately before any BRDF calculations
- Shadows, AO and other global illumination effects - outside of the scope of the current implementation
- Displacement

## Demo setup ##

The demo uses material presets with constant (non-textured) values for various properties, one directional light and textured environment in order to demonstrate various materials under direct and indirect illumination.

Most of the material parameters can use variable inputs such as file or procedural textures. The respective value needs to be read or calculated and set into the VRayMtlInitParams structure before calling initVRayMtlContext().

You can see this shader in action at [https://www.shadertoy.com/view/3dfcRr](https://www.shadertoy.com/view/3dfcRr).
