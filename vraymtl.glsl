//********************************************************************
// V-Ray Material Shader
//
// Copyright (c) 2020 Chaos Software Ltd
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//********************************************************************

precision highp float;

#define PI 3.1415926535897932384626433832795
#define INV_PI 0.31830988618
#define INV_2PI 0.15915494309
// A spherical env map affects how the LOD is computed based on normal
#define ENV_MAP_SPHERICAL 0
// How many env samples to take - increase for rougher surfaces
#define NUM_ENV_SAMPLES 8
// Additional samples added for rough reflection & refraction
#define NUM_ENV_SAMPLES_ROUGH 16
// Set to 1 to use a procedural checker environment (useful for local testing)
#define PROCEDURAL_ENV 1

// Conductor Fresnel values for sheen
#define SHEEN_N 2.9114
#define SHEEN_K 3.0893

// color conversion
vec3 srgb_from_rgb(vec3 rgb) {
	vec3 a = vec3(0.055, 0.055, 0.055);
	vec3 ap1 = vec3(1.0, 1.0, 1.0) + a;
	vec3 g = vec3(2.4, 2.4, 2.4);
	vec3 ginv = 1.0 / g;
	vec3 select = step(vec3(0.0031308, 0.0031308, 0.0031308), rgb);
	vec3 lo = rgb * 12.92;
	vec3 hi = ap1 * pow(rgb, ginv) - a;
	return mix(lo, hi, select);
}

vec3 rgb_from_srgb(vec3 srgb) {
	vec3 a = vec3(0.055, 0.055, 0.055);
	vec3 ap1 = vec3(1.0, 1.0, 1.0) + a;
	vec3 g = vec3(2.4, 2.4, 2.4);
	vec3 select = step(vec3(0.04045, 0.04045, 0.04045), srgb);
	vec3 lo = srgb / 12.92;
	vec3 hi = pow((srgb + a) / ap1, g);
	return mix(lo, hi, select);
}


// Engine-specific functions {{{
// These must be implemented depending on your engine
// The implementation here is for use in ShaderToy, with env map in iChannel0
// Note that the TextureEnvMapLOD and EnvIrradiance functions must return linear rgb

// Get maximum lod for texture env
float engGetMaxEnvLOD() {
	ivec2 envSize = textureSize(iChannel0, 0);
	float minsz = float(max(envSize.x, envSize.y));
	return log2(minsz);
}

// Convert Cartesian vector to spherical coordinates
vec2 toSpherical(vec3 dir) {
	float alpha, beta;
	if (dir.z * dir.z + dir.x * dir.x < 1e-12) {
		alpha = 0.0;
		beta = (dir.y > 0.0) ? 0.0 : 1.0;
	} else {
		alpha = atan(dir.z, dir.x) * INV_PI;
		beta = acos(clamp(dir.y, -1.0, 1.0)) * INV_PI;
	}

	float u = alpha * 0.5 + 0.5;
	float v = beta;
	return vec2(u, v);
}

vec3 sampleCheckerEnv(vec3 dir) {
	vec2 uv = toSpherical(dir);
	float size = 0.1;
	int x = int(floor(uv.x / size));
	int y = int(floor(uv.y / size));
	return vec3(1.0 - float((x + y) & 1));
}

// Sample environment with LOD
vec3 engTextureEnvMapLOD(vec3 dir, float lod) {
#if PROCEDURAL_ENV == 1
	return sampleCheckerEnv(dir);
#else
    vec3 color;
#   if ENV_MAP_SPHERICAL
    vec2 texcoord = toSpherical(dir);
    texcoord.y = 1.0 - texcoord.y;
    color = textureLod(iChannel0, texcoord, lod).xyz;
#   else
    color = textureLod(iChannel0, dir, lod).xyz;
#   endif
    return rgb_from_srgb(color);
#endif
}

// Diffuse environment light (averaged over the hemisphere)
vec3 engEnvIrradiance(vec3 dir) {
#if PROCEDURAL_ENV == 1
	return sampleCheckerEnv(dir);
#else
	float lod = max(0.0, engGetMaxEnvLOD() - 1.0);
	return rgb_from_srgb(textureLod(iChannel0, dir, lod).xyz);
#endif
}

/// Get the LOD for sampling the environment
/// @param Wn World-space normal
/// @param p Probability of this direction (from sampleBRDF)
/// @param numSamples Number of environment samples for the BRDF
float computeEnvLOD(vec3 Wn, float p, int numSamples) {
#if ENV_MAP_SPHERICAL
	float distortion = sqrt(max(0.0, 1.0 - Wn.y * Wn.y));
#else
	float distortion = 1.0;
#endif
	if (numSamples < 2) {
		return 0.0;
	} else {
		return max(0.0, (engGetMaxEnvLOD() - 0.5 * log2(1.0 + float(numSamples) * p * INV_2PI * distortion)));
	}
}

// }}} engine-specific functions

struct VRayMtlInitParams {
	vec3 Vw;
	vec3 geomNormal;
	vec3 diffuseColor;
	float diffuseAmount;
	float roughness;
	vec3 selfIllum;
	vec3 reflColor;
	float reflAmount;
	float reflGloss;
	bool traceReflections;
	float metalness;
	float aniso;
	float anisoRotation;
	int anisoAxis;
	vec3 opacity;
	vec3 refractionColor;
	float refractionAmount;
	float refrGloss;
	bool traceRefractions;
	float refractionIOR;
	bool useFresnel;
	float fresnelIOR;
	bool lockFresnelIOR;
	bool doubleSided;
	bool useRoughness;
	float gtrGamma;
	int brdfType;
	vec3 fogColor;
	float fogMult;
	float fogBias;
	bool sssOn;
	vec3 translucencyColor;
	float sssFwdBackCoeff;
	float sssScatterCoeff;
	float thickness;
	float distToCamera;
	vec3 sheenColor;
	float sheenAmount;
	float sheenGlossiness;
	vec3 coatColor;
	float coatAmount;
	float coatGlossiness;
	float coatIOR;
};

struct VRayMtlContext {
	vec3 geomNormal;
	float gloss1;
	float roughnessSqr;
	float reflGloss;
	vec3 e;
	vec3 diff;
	float fresnel;
	vec3 refl;
	vec3 refr;
	vec3 illum;
	vec3 opacity;
	float rtermA;
	float rtermB;
	float gtrGamma;
	float fragmentNoise; // per-fragment noise value
	mat3 nm;
	mat3 inm;
	vec3 sheen;
	bool hasSheen;
	float sheenGloss;
	vec3 coat;
	mat3 coatNM;
	float coatRoughnessSqr;
	bool hasCoat;
};

vec3 sampleBRDF(VRayMtlInitParams params, VRayMtlContext ctx,
		int sampleIdx, int nbSamples, out float brdfContrib);
vec3 sampleRefractBRDF(VRayMtlInitParams params, VRayMtlContext ctx,
		int sampleIdx, int nbSamples, out bool totalInternalReflection);

VRayMtlContext initVRayMtlContext(VRayMtlInitParams initParams);

vec3 computeDirectDiffuseContribution(VRayMtlInitParams params, VRayMtlContext ctx, vec3 lightDir);
vec3 computeDirectReflectionContribution(VRayMtlInitParams params, VRayMtlContext ctx, vec3 lightDir);
vec3 computeDirectSheenContribution(VRayMtlInitParams params, VRayMtlContext ctx, vec3 lightDir);
vec3 computeDirectCoatContribution(VRayMtlInitParams params, VRayMtlContext ctx, vec3 lightDir);

vec3 computeIndirectDiffuseContribution(VRayMtlInitParams params, VRayMtlContext ctx);
vec3 computeIndirectReflectionContribution(VRayMtlInitParams params, VRayMtlContext ctx);
vec3 computeIndirectRefractionContribution(VRayMtlInitParams params, VRayMtlContext ctx, float alpha, vec3 alphaDir);
vec3 computeIndirectSheenContribution(VRayMtlInitParams params, VRayMtlContext ctx);
vec3 computeIndirectCoatContribution(VRayMtlInitParams params, VRayMtlContext ctx);

vec3 computeRefractFogContrib(VRayMtlInitParams params, VRayMtlContext ctx, vec3 diffuseContrib);

// utility functions {{{

float sqr(float x) {
	return x * x;
}

// return random number in [0, 1)
float hashRand(vec2 co) {
	return fract(sin(dot(co.xy, vec2(12.9898, 78.233))) * 43758.5453);
}

// return random vector in [0, 1)
vec2 rand(VRayMtlContext ctx, int sampleIdx, int nbSamples) {
	// fibonacci spiral distribution using the plastic constant
	const float plast = 1.324717957244746;
	const float invPlast = 1.0/plast;
	return vec2(
		fract(float(sampleIdx + 1) * invPlast),
		float(sampleIdx) / float(nbSamples) + ctx.fragmentNoise
	);
}

float intensity(vec3 v) {
	return (v.x + v.y + v.z) / 3.0;
}

vec3 whiteComplement(vec3 x) {
	return clamp(1.0 - x, 0.0, 1.0);
}

// }}} end utility functions

/// Compute the two orthogonal vectors to a given input vector
/// @param n Input vector
/// @param[out] u The first orthogonal vector
/// @param[out] v The second orthogonal vector
void computeTangentVectors(vec3 n, out vec3 u, out vec3 v) {
	// It doesn't matter what these vectors are, the result vectors just need to be perpendicular to the normal and to
	// each other
	u = cross(n, vec3(0.643782, 0.98432, 0.324632));
	if (length(u) < 1e-6)
		u = cross(n, vec3(0.432902, 0.43223, 0.908953));
	u = normalize(u);
	v = normalize(cross(n, u));
}

/// Make an orthogonal matrix given a surface normal
/// @param n The normal vector
/// @param[out] m The output orthogonal matrix with n in the third column
void makeNormalMatrix(in vec3 n, out mat3 m) {
	computeTangentVectors(n, m[0], m[1]);
	m[2] = n;
}

/// Get the Fresnel reflectance for a dielectric.
/// @param fresnelIOR Surface index of refraction
/// @param e View direction
/// @param n Surface normal
/// @param refractDir Refracted view direction
/// @return Fresnel reflectance
float getFresnelCoeff(float fresnelIOR, vec3 e, vec3 n, vec3 refractDir) {
	if (abs(fresnelIOR - 1.0) < 1e-6)
		return 0.0;

	float cosIn = -dot(e, n);
	float cosR = -dot(refractDir, n);

	if (cosIn > 1.0 - 1e-12 || cosR > 1.0 - 1e-12) { // View direction is perpendicular to the surface
		float f = (fresnelIOR - 1.0) / (fresnelIOR + 1.0);
		return f * f;
	}

	float ks = (cosR / cosIn) * fresnelIOR;
	float fs2 = (ks - 1.0) / (ks + 1.0);
	float Fs = fs2 * fs2;

	float kp = (cosIn / cosR) * fresnelIOR;
	float fp2 = (kp - 1.0) / (kp + 1.0);
	float Fp = fp2 * fp2;

	return 0.5 * (Fs + Fp);
}

/// Get the Fresnel reflectance for a conductor.
/// Accurate values for n and k can be obtained from https://refractiveindex.info/
/// For some conductors the n and k parameters vary with the light wavelength so the
/// Fresnel reflectance should be computed separately for R,G and B.
/// @param cosTheta Cosine of the angle between the view direction and the normal
/// @param n Refractive index
/// @param k Extinction coefficient
/// @return Fresnel reflectance.
float getConductorFresnel(float cosTheta, float n, float k) {
	float c2 = cosTheta * cosTheta;
	float n2k2 = n * n + k * k;
	float nc2 = 2.0f * n * cosTheta;
	float rsa = n2k2 + c2;
	float rpa = n2k2 * c2 + 1.0f;
	float rs = (rsa - nc2) / (rsa + nc2);
	float rp = (rpa - nc2) / (rpa + nc2);
	return 0.5f * (rs + rp);
}

vec3 getSpecularDir(float u, float v, float k) {
	float thetaSin = clamp(pow(u, 1.0 / (k + 1.0)), 0.0, 1.0);
	float thetaCos = sqrt(1.0 - thetaSin * thetaSin);
	float phi = 2.0 * PI * v;
	return vec3(cos(phi) * thetaCos, sin(phi) * thetaCos, thetaSin);
}

vec3 getPhongDir(float uc, float vc, float glossiness, vec3 view, mat3 nm) {
	vec3 reflectDir = reflect(-view, nm[2]);
	vec3 s = cross(vec3(0, 1, 0), reflectDir);
	vec3 s1 = cross(reflectDir, s);
	mat3 m;
	m[0] = normalize(s);
	m[1] = normalize(s1);
	m[2] = normalize(reflectDir);
	vec3 sampleDir = getSpecularDir(uc, vc, glossiness);
	return m * sampleDir;
}

vec3 getBlinnDir(float uc, float vc, float glossiness, vec3 view, mat3 nm) {
	vec3 nn = getSpecularDir(uc, vc, glossiness);
	vec3 h = normalize(nm * nn);
	float cs = 2.0 * dot(h, view);
	vec3 dir = normalize(-view + cs * h);
	return dir;
}

vec3 getSphereDir(float u, float v) {
	float thetaSin = u;
	float thetaCos = sqrt(1.0 - thetaSin * thetaSin);
	float phi = 2.0 * PI * v;
	return vec3(cos(phi) * thetaCos, sin(phi) * thetaCos, thetaSin);
}

vec3 getDiffuseDir(float u, float v) {
	float thetaSin = sqrt(u);
	float thetaCos = sqrt(1.0 - u);
	float phi = 2.0 * PI * v;
	return vec3(cos(phi) * thetaCos, sin(phi) * thetaCos, thetaSin);
}

vec3 getWardDir(float u, float v, float glossiness, vec3 view, mat3 nm) {
	if (u >= 1.0)
		u -= 1.0;

	float k = -log(1.0 - u);
	if (k < 0.0)
		k = 0.0;

	float thetaCos = sqrt(1.0 / (glossiness * k + 1.0));
	vec3 hn = getSphereDir(thetaCos, v);
	vec3 hw = normalize(nm * hn);
	vec3 dir = reflect(-view, hw);
	return dir;
}

vec3 getGTR1MicroNormal(float uc, float vc, float sharpness) {
	float sharpness2 = min(sharpness * sharpness, 0.999);
	float thetaCosSqr = (1.0 - pow(sharpness2, 1.0 - uc)) / (1.0 - sharpness2);
	float thetaCos = sqrt(thetaCosSqr);
	float thetaSin = sqrt(max(1.0 - thetaCosSqr, 0.0));

	float phi = 2.0 * PI * vc;
	return vec3(cos(phi) * thetaSin, sin(phi) * thetaSin, thetaCos);
}

// Specific implementation when gamma == 2. See section B.2 Physically-Based Shading at Disney from SIGGRAPH 2012
vec3 getGTR2MicroNormal(float uc, float vc, float sharpness) {
	float thetaCosSqr = (1.0 - uc) / (1.0 + (sharpness * sharpness - 1.0) * uc);
	float thetaCos = sqrt(thetaCosSqr);
	float thetaSin = sqrt(max(1.0 - thetaCosSqr, 0.0));

	float phi = 2.0 * PI * vc;
	return vec3(cos(phi) * thetaSin, sin(phi) * thetaSin, thetaCos);
}

// General implementation  when gamma != 1 and != 2. See section B.2 Physically-Based Shading at Disney from SIGGRAPH 2012
vec3 getGTRMicroNormal(float uc, float vc, float sharpness, float gtrGamma) {
	float sharpness2 = min(sharpness * sharpness, 0.999);
	float thetaCosSqr =
		(1.0 - pow(pow(sharpness2, 1.0 - gtrGamma) * (1.0 - uc) + uc, 1.0 / (1.0 - gtrGamma))) / (1.0 - sharpness2);
	float thetaCos = sqrt(thetaCosSqr);
	float thetaSin = sqrt(max(1.0 - thetaCosSqr, 0.0));

	float phi = 2.0 * PI * vc;
	return vec3(cos(phi) * thetaSin, sin(phi) * thetaSin, thetaCos);
}

vec3 getGGXMicroNormal(float uc, float vc, float sharpness, float gtrGamma) {
	if (abs(gtrGamma - 1.0) < 1e-3)
		return getGTR1MicroNormal(uc, vc, sharpness);
	else if (abs(gtrGamma - 2.0) < 1e-3)
		return getGTR2MicroNormal(uc, vc, sharpness);
	else // if (gtrLowerLimit <= gtrGamma && gtrGamma <= gtrUpperLimit)
		return getGTRMicroNormal(uc, vc, sharpness, gtrGamma);
}

float getGTR1MicrofacetDistribution(float mz, float sharpness) {
	float cosThetaM = mz; // dotf(microNormal, normal);
	if (cosThetaM <= 1e-3)
		return 0.0;

	float cosThetaM2 = sqr(cosThetaM);
	float tanThetaM2 = (1.0 / cosThetaM2) - 1.0;
	float sharpness2 = sqr(sharpness);
	float div = PI * log(sharpness2) * cosThetaM2 * (sharpness2 + tanThetaM2);
	// when div<(sharpness2-1.0)*1e-6 no division by zero will occur (the dividend and the divisor are always negative);
	// div can get 0 in rare situation when the sharpness read from texture mapped in reflection glossines is 0
	// and cosThetaM is 1 (and consequently tanThetaM2 is 0);
	float res = (div < (sharpness2 - 1.0) * 1e-6) ? (sharpness2 - 1.0) / div : 0.0;

	return res;
}

float getGTR2MicrofacetDistribution(float mz, float sharpness) {
	float cosThetaM = mz; // dotf(microNormal, normal);
	if (cosThetaM <= 1e-3)
		return 0.0;

	float cosThetaM2 = sqr(cosThetaM);
	float tanThetaM2 = (1.0 / cosThetaM2) - 1.0;
	float sharpness2 = sqr(sharpness);
	float div = PI * sqr(cosThetaM2 * (sharpness2 + tanThetaM2));
	// when div>sharpness2*1e-6 no division by zero will occur (the dividend and the divisor are always positive);
	// div canget0 in rare situation when the sharpness read from texture mapped in reflection glossines is 0
	// and cosThetaM is 1 (and consequently tanThetaM2 is 0);
	float res = (div > sharpness2 * 1e-6) ? sharpness2 / div : 0.0;

	return res;
}

float getGTRMicrofacetDistribution(float mz, float sharpness, float gtrGamma) {
	float cosThetaM = mz; // dotf(microNormal, normal);
	if (cosThetaM <= 1e-3)
		return 0.0;

	float cosThetaM2 = sqr(cosThetaM);
	float tanThetaM2 = (1.0 / cosThetaM2) - 1.0;
	float sharpness2 = sqr(sharpness);
	float divisor =
		PI * (1.0 - pow(sharpness2, 1.0 - gtrGamma)) * pow(cosThetaM2 * (sharpness2 + tanThetaM2), gtrGamma);
	float dividend = (gtrGamma - 1.0) * (sharpness2 - 1.0);
	// when fabsf(divisor)>fabsf(dividend)*1e-6 no division by zero will occur
	// (the dividend and the divisor are always either both positive or both negative);
	// divisor canget0 in rare situation when the sharpness read from texture mapped in reflection glossines is 0
	// and cosThetaM is 1 (and consequently tanThetaM2 is 0);
	float res = (abs(divisor) > abs(dividend) * 1e-6) ? dividend / divisor : 0.0;

	return res;
}

float getGGXMicrofacetDistribution(float cosNH, float sharpness, float gtrGamma) {
	if (abs(gtrGamma - 1.0) < 1e-3)
		return getGTR1MicrofacetDistribution(cosNH, sharpness);
	else if (abs(gtrGamma - 2.0) < 1e-3)
		return getGTR2MicrofacetDistribution(cosNH, sharpness);
	else // if (gtrLowerLimit <= gtrGamma && gtrGamma <= gtrUpperLimit)
		return getGTRMicrofacetDistribution(cosNH, sharpness, gtrGamma);
}

float getGTRMonodirectionalShadowing0(float cotThetaV) {
	return 2.0 / (1.0 + sqrt(1.0 + 1.0 / (cotThetaV * cotThetaV)));
}

float getGTRMonodirectionalShadowing1(float sharpness, float cotThetaV) {
	float cotThetaV2 = sqr(cotThetaV);
	float sharpness2 = min(0.999, sqr(sharpness));
	float a = sqrt(cotThetaV2 + sharpness2);
	float b = sqrt(cotThetaV2 + 1.0);
	return cotThetaV * log(sharpness2) / (a - b + cotThetaV * log(sharpness2 * (cotThetaV + b) / (cotThetaV + a)));
}

float getGTRMonodirectionalShadowing2(float sharpness, float cotThetaV) {
	return 2.0 / (1.0 + sqrt(1.0 + sqr(sharpness / cotThetaV)));
}

float getGTRMonodirectionalShadowing3(float sharpness, float cotThetaV) {
	float cotThetaV2 = sqr(cotThetaV);
	float sharpness2 = min(0.999, sqr(sharpness));
	float a = sqrt(cotThetaV2 + sharpness2);
	float b = sharpness2 + 1.0;
	return 4.0 * cotThetaV * a * b / (2.0 * cotThetaV * b * (cotThetaV + a) + sharpness2 * (3.0 * sharpness2 + 1.0));
}

float getGTRMonodirectionalShadowing4(float sharpness, float cotThetaV) {
	float cotThetaV2 = cotThetaV * cotThetaV;
	float sharpness2 = min(0.999, sqr(sharpness));
	float sharpness4 = sharpness2 * sharpness2;
	float a = 8.0 * (sharpness4 + sharpness2 + 1.0);
	float b = sqrt(cotThetaV2 + sharpness2);
	float b3 = b * (cotThetaV2 + sharpness2);
	return 2.0 * cotThetaV * a * b3
		/ (a * cotThetaV * (b3 + cotThetaV * cotThetaV2)
		   + 3.0 * sharpness2
			   * (4.0 * cotThetaV2 * (2.0 * sharpness4 + sharpness2 + 1.0)
				  + sharpness2 * (5.0 * sharpness4 + 2.0 * sharpness2 + 1.0)));
}

float getGGXMonodirectionalShadowing(vec3 dir, vec3 hw, vec3 normal, float sharpness, float gtrGamma) {
	float cosThetaV = dot(dir, normal);

	if (cosThetaV <= 1e-3)
		return 0.0;

	if (dot(dir, hw) * cosThetaV <= 0.0) // Note: technically this is a division, but since we are only interested in
										 // the sign, we can do multiplication
		return 0.0;

	// when direction is collinear to the normal there is no shadowing
	// moreover if this case is not handled a division by zero will happen on the next line
	if (cosThetaV >= 1.0 - 1e-6)
		return 1.0;

	float cotThetaV = cosThetaV / sqrt(1.0 - sqr(cosThetaV));

	float res = 0.0;

	// when gamma is any of the integer values 0, 1, 2, 3, 4 apply analytical solution
	if (gtrGamma <= 0.01)
		res = getGTRMonodirectionalShadowing0(cotThetaV);
	else if (abs(gtrGamma - 1.0) <= 1e-2)
		res = getGTRMonodirectionalShadowing1(sharpness, cotThetaV);
	else if (abs(gtrGamma - 2.0) <= 1e-2)
		res = getGTRMonodirectionalShadowing2(sharpness, cotThetaV);
	else if (abs(gtrGamma - 3.0) <= 1e-2)
		res = getGTRMonodirectionalShadowing3(sharpness, cotThetaV);
	else if (gtrGamma >= 4.0 - 1e-2)
		res = getGTRMonodirectionalShadowing4(sharpness, cotThetaV);
	else {
		// gamma is not an integer. interpolate
		// gtrGamma is not an integer. interpolate
		// If we get here gtrGamma is in (0.01, 3.99).

		// We use a cubic spline curve with 5 knots to evaluate the shadowing, based on the results for integer values.
		// The original code used a CubicSpline<5> object to construct and evaluate the spline, but Vladimir Nedev
		// derived a simplified version for Lavina, which is used below.

		// knots[i].x is implicit and is equal to 'i', so we store only knots[i].y in knots[i].
		float knots[5];
		knots[0]=getGTRMonodirectionalShadowing0(cotThetaV);
		knots[1]=getGTRMonodirectionalShadowing1(sharpness, cotThetaV);
		knots[2]=getGTRMonodirectionalShadowing2(sharpness, cotThetaV);
		knots[3]=getGTRMonodirectionalShadowing3(sharpness, cotThetaV);
		knots[4]=getGTRMonodirectionalShadowing4(sharpness, cotThetaV);

		// The code that follows is a simplified version of the code for CubicSpline<5> that constructs the spline,
		// using the fact that the distance between the spline knots in x is always 1.
		// We also directly compute which polynomial we are going to use instead of binary searching.
		// From CubicSpline::construct: h[i] = 1
		// From evalMomentsNatural: m[i] = 4
		// We compute f, instead of 'c', since 'f' is the argument name of solveTridiagonal.
		// From solveTridiagonal:
		// a[i] = h[i] = 1
		// b[i] = h[i+1] = 1
		// c[i] = m[i+1] = 4;
		float f[3];
		f[0]=knots[2]-knots[1]-knots[1]+knots[0];
		f[1]=knots[3]-knots[2]-knots[2]+knots[1];
		f[2]=knots[4]-knots[3]-knots[3]+knots[2];

		f[1]-=0.25*f[0];
		f[2]-=0.26666666666666666666666666666667*f[1];

		// Reuse 'f'.
		f[2]=f[2]*0.26785714285714285714285714285715;
		f[1]=(f[1]-f[2])*0.26666666666666666666666666666667;
		f[0]=(f[0]-f[1])*0.25;

		int i=int(floor(gtrGamma));

		float mi=(i>0 ? f[i-1] : 0.0);
		float mi1=(i<3 ? f[i] : 0.0);
		float a=(mi1-mi);
		float b=(3.0*mi);
		float c=(knots[i+1]-knots[i])-(2.0*mi+mi1);
		float d=knots[i];

		float x=gtrGamma-float(i);
		res=((a*x+b)*x+c)*x+d;
	}

	return clamp(res, 0.0, 1.0);
}

float getGGXBidirectionalShadowingMasking(
	vec3 view, vec3 dir, vec3 hw, vec3 normal, float sharpness, float gtrGamma) {
	return getGGXMonodirectionalShadowing(view, hw, normal, sharpness, gtrGamma)
		* getGGXMonodirectionalShadowing(dir, hw, normal, sharpness, gtrGamma);
}

float getGGXContribution(
	vec3 view,
	vec3 dir,
	vec3 hw,
	vec3 hl,
	float sharpness,
	float gtrGamma,
	vec3 normal,
	out float partialProb,
	out float D) {
	float cosIN = abs(dot(view, normal));
	float cosON = abs(dot(dir, normal));

	if (cosIN <= 1e-6 || cosON <= 1e-6)
		return 0.0;

	float partialBrdf = 0.0;

	float hn = hl.z;
	D = getGGXMicrofacetDistribution(hn, sharpness, gtrGamma);
	// division by cosON is omitted because we would have to multiply by the same below
	partialBrdf =
		0.25 * getGGXBidirectionalShadowingMasking(view, dir, hw, normal, sharpness, gtrGamma) / cosIN;

	if (hn > 0.0) {
		partialProb = hn;

		float ho = dot(hw, dir);
		partialProb *= ho > 0.0 ? 0.25 / ho : 0.0;
	}

	// reduce some multiplications in the final version
	// partialBrdf *= cosON; - omitted

	return partialBrdf;
}

vec3 getGGXDir(
	float u, float v, float sharpness, float gtrGamma, vec3 view, mat3 nm, out float prob, out float brdfDivByProb) {
	vec3 microNormalLocal = getGGXMicroNormal(u, v, sharpness, gtrGamma);
	if (microNormalLocal.z < 0.0)
		return nm[2];

	vec3 microNormal = nm * microNormalLocal;

	// Compute and keep the length of the half-vector in local space; needed for anisotropy correction
	float L2 = dot(microNormal, microNormal);
	float L = sqrt(L2);
	microNormal /= L;

	vec3 dir = reflect(-view, microNormal);

	float Dval = 0.0;
	float partialProb = 0.0;
	float partialBrdf =
		getGGXContribution(view, dir, microNormal, microNormalLocal, sharpness, gtrGamma, nm[2], partialProb, Dval);
	partialProb *= L * L2; // take anisotropy in consideration
	prob = (Dval >= 1e-6) ? partialProb * Dval * 2.0 * PI
						  : 1e18; // compute full probability and apply vray specific corrections
	brdfDivByProb = (partialProb >= 1e-6) ? partialBrdf / partialProb : 0.0;
	return dir;
}

vec3 sampleBRDF(
	VRayMtlInitParams params, VRayMtlContext ctx, int sampleIdx, int nbSamples, out float rayProb, out float brdfContrib) {
	vec3 geomNormal = params.geomNormal;
	float ggxTail = params.gtrGamma;
	int brdfType = params.brdfType;
	vec2 uv = rand(ctx, sampleIdx, nbSamples);
	float u = uv.x, v = uv.y;

	vec3 dir = vec3(0.0);
	rayProb = 1.0;
	brdfContrib = 1.0;
	if (brdfType == 0) {
		dir = getPhongDir(u, v, ctx.gloss1, -ctx.e, ctx.nm);
	} else if (brdfType == 1) {
		dir = getBlinnDir(u, v, ctx.gloss1, -ctx.e, ctx.nm);
	} else if (brdfType == 2) {
		dir = getWardDir(u, v, ctx.roughnessSqr, -ctx.e, ctx.nm);
	} else /* brdfType==3 */ {
		dir = getGGXDir(u, v, ctx.roughnessSqr, ctx.gtrGamma, -ctx.e, ctx.nm, rayProb, brdfContrib);
	}

	if (dot(dir, geomNormal) < 0.0) {
		brdfContrib = 0.0;
	}
	return dir;
}

vec3 sampleCoatBRDF(
	VRayMtlInitParams params, VRayMtlContext ctx, int sampleIdx, int nbSamples, out float rayProb, out float brdfContrib) {
	vec3 geomNormal = params.geomNormal;
	vec2 uv = rand(ctx, sampleIdx, nbSamples);
	float u = uv.x, v = uv.y;

	vec3 dir = vec3(0.0);
	rayProb = 1.0;
	brdfContrib = 1.0;
	dir = getGGXDir(u, v, ctx.coatRoughnessSqr, 2.0, -ctx.e, ctx.coatNM, rayProb, brdfContrib);

	if (dot(dir, geomNormal) < 0.0) {
		brdfContrib = 0.0;
	}
	return dir;
}

vec3 sampleRefractBRDF(
	VRayMtlInitParams params, VRayMtlContext ctx, int sampleIdx, int nbSamples, out bool totalInternalReflection) {
	vec3 geomNormal = params.geomNormal;
	vec3 refractDir = refract(ctx.e, geomNormal, 1.0 / params.refractionIOR);
	totalInternalReflection = false;
	if (refractDir == vec3(0.0)) {
		refractDir = reflect(ctx.e, geomNormal);
		totalInternalReflection = true;
	}

	vec3 s = cross(vec3(0, 1, 0), refractDir);
	vec3 s1 = cross(refractDir, s);
	mat3 m;
	m[0] = normalize(s);
	m[1] = normalize(s1);
	m[2] = normalize(refractDir);

	vec2 uv = rand(ctx, sampleIdx, nbSamples);
	float u = uv.x, v = uv.y;
	float gloss = 1.0 / pow(max(1.0 - params.refrGloss, 1e-4), 3.5) - 1.0;
	vec3 sampleDir = getSpecularDir(u, v, gloss);

	return m * sampleDir;
}

/// Sheen BRDF functions based on the Production Friendly Microfacet Sheen BRDF paper
/// Implementation of the curve fitting polynomial (Table 1 and Section 3)
float sheenP(float a, float b, float c, float d, float e, float x) {
	return a / (1.0 + b * pow(x, c)) + d * x + e;
}

/// Implementation of the lambda curve fitting and interpolation (Table 1 and Section 3)
float sheenL(float x, float roughness) {
	float a0 = 25.3245;
	float b0 = 3.32435;
	float c0 = 0.16801;
	float d0 = -1.27393;
	float e0 = -4.85967;
	float a1 = 21.5473;
	float b1 = 3.82987;
	float c1 = 0.19823;
	float d1 = -1.97760;
	float e1 = -4.32054;

	float t = (1.0 - roughness) * (1.0 - roughness);
	float p0 = sheenP(a0, b0, c0, d0, e0, x);
	float p1 = sheenP(a1, b1, c1, d1, e1, x);
	return t * p0 + (1.0 - t) * p1;
}

/// Implementation of the lambda term (Section 3)
float sheenLambda(float cosTheta, float roughness) {
	if (cosTheta < 0.5) {
		return exp(sheenL(cosTheta, roughness));
	} else {
		return exp(2.0 * sheenL(0.5, roughness) - sheenL(1.0 - cosTheta, roughness));
	}
}

/// Implementation of the full shadowing term (Section 3 and Section 4)
float sheenShadowingMasking(float cosIN, float cosON, float roughness) {
	float c1 = 1.0 - cosON;
	float c2 = c1 * c1;
	float c4 = c2 * c2;
	float c8 = c4 * c4;
	float lambdaON = pow(sheenLambda(cosON, roughness), 1.0 + 2.0 * c8);
	float lambdaIN = sheenLambda(cosIN, roughness);
	return 1.0 / (1.0 + lambdaIN + lambdaON);
}

/// Implementation of the full sheen BRDF including the cos(N,L) multiplication and 
/// VRay's probability transformation (2pi multiplication)
/// Glossiness must be in the [0, 1) range. In theory the result is undefined for glossiness = 1
/// but in practice the highlight disappears and we check for that as soon as we read the glossiness texture.
float sheenProbability(vec3 viewDir, vec3 lightDir, vec3 normal, float glossiness) {
	vec3 incomingDir = -viewDir;

	float cosIN = min(1.0, dot(incomingDir, normal));
	float cosON = min(1.0, dot(lightDir, normal));
	if (cosIN <= 1e-6 || cosON <= 1e-6)
		return 0.0;

	float roughness = 1.0 - glossiness;
	vec3 halfVector = normalize(lightDir - viewDir);

	float cosTheta = clamp(dot(halfVector, normal), 0.0, 1.0);
	// This should be fine because we expect theta in [0, pi/2] range and both 
	// sin(theta) and cos(theta) are non-negative in this case
	float sinThetaSq = clamp(1.0 - cosTheta * cosTheta, 0.0, 1.0);
	// Compute the microfacet distribution (Section 2)
	// The 2pi divide is cancelled by VRay's probability transformation
	float invRoughness = 1.0 / roughness;
	float D = (2.0 + invRoughness) * pow(sinThetaSq, 0.5 * invRoughness);
	float G = sheenShadowingMasking(cosIN, cosON, roughness);
	// cosON divide will be cancelled by cosON multiplication later so just skip both.
	float res = 0.25 * D * G / cosIN;
	return res;
}

/// Size of the sheen albedo LUT
#define SHEEN_LUT_SIZE 16

/// Directional sheen albedo LUT where the row index corresponds to roughness and the column index corresponds to cosTheta
/// Conductor Fresnel for wavelength 650nm with n=2.9114 and k=3.0893 (Iron) is used instead of the usual dielectric Fresnel.
/// Conductor Fresnel inputs taken from https://refractiveindex.info/
/// It's computed according to Section 2.1.5. in the paper
/// "A Microfacet Based Coupled Specular-Matte BRDF Model with Importance Sampling"
float sheenAlbedoLUT[SHEEN_LUT_SIZE * SHEEN_LUT_SIZE] = float[] (
	0.64503, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000,
	0.44977, 0.26630, 0.18104, 0.12713, 0.08979, 0.06302, 0.04360, 0.02954, 0.01947, 0.01245, 0.00762, 0.00438, 0.00229, 0.00103, 0.00035, 0.00005,
	0.38310, 0.26927, 0.20305, 0.15812, 0.12440, 0.09790, 0.07660, 0.05932, 0.04531, 0.03419, 0.02527, 0.01810, 0.01234, 0.00780, 0.00429, 0.00170,
	0.36921, 0.27749, 0.21857, 0.17769, 0.14612, 0.12046, 0.09902, 0.08085, 0.06543, 0.05263, 0.04182, 0.03255, 0.02453, 0.01759, 0.01161, 0.00650,
	0.37188, 0.28926, 0.23348, 0.19442, 0.16387, 0.13861, 0.11709, 0.09843, 0.08221, 0.06847, 0.05657, 0.04605, 0.03662, 0.02811, 0.02040, 0.01344,
	0.38180, 0.30284, 0.24825, 0.20989, 0.17968, 0.15446, 0.13273, 0.11365, 0.09683, 0.08242, 0.06979, 0.05843, 0.04806, 0.03849, 0.02961, 0.02137,
	0.39514, 0.31710, 0.26267, 0.22437, 0.19410, 0.16870, 0.14666, 0.12715, 0.10981, 0.09488, 0.08168, 0.06971, 0.05867, 0.04836, 0.03868, 0.02957,
	0.40983, 0.33122, 0.27639, 0.23780, 0.20725, 0.18153, 0.15912, 0.13919, 0.12140, 0.10601, 0.09236, 0.07993, 0.06839, 0.05755, 0.04731, 0.03762,
	0.42452, 0.34459, 0.28906, 0.25001, 0.21907, 0.19299, 0.17020, 0.14987, 0.13168, 0.11592, 0.10191, 0.08911, 0.07719, 0.06598, 0.05535, 0.04527,
	0.43822, 0.35667, 0.30038, 0.26082, 0.22948, 0.20304, 0.17990, 0.15923, 0.14070, 0.12464, 0.11035, 0.09727, 0.08509, 0.07361, 0.06272, 0.05239,
	0.45014, 0.36703, 0.31005, 0.27004, 0.23836, 0.21161, 0.18820, 0.16726, 0.14848, 0.13220, 0.11771, 0.10444, 0.09208, 0.08043, 0.06937, 0.05890,
	0.45964, 0.37532, 0.31783, 0.27751, 0.24559, 0.21866, 0.19507, 0.17397, 0.15503, 0.13861, 0.12400, 0.11064, 0.09818, 0.08643, 0.07530, 0.06476,
	0.46621, 0.38123, 0.32352, 0.28309, 0.25111, 0.22412, 0.20048, 0.17933, 0.16034, 0.14389, 0.12926, 0.11587, 0.10340, 0.09164, 0.08050, 0.06997,
	0.46947, 0.38454, 0.32698, 0.28669, 0.25484, 0.22796, 0.20442, 0.18335, 0.16443, 0.14805, 0.13348, 0.12016, 0.10775, 0.09606, 0.08499, 0.07452,
	0.46915, 0.38511, 0.32810, 0.28824, 0.25674, 0.23016, 0.20687, 0.18602, 0.16730, 0.15109, 0.13670, 0.12353, 0.11127, 0.09971, 0.08876, 0.07842,
	0.46510, 0.38285, 0.32683, 0.28769, 0.25679, 0.23070, 0.20784, 0.18736, 0.16896, 0.15304, 0.13891, 0.12599, 0.11395, 0.10260, 0.09184, 0.08168
);

/// Average sheen albedo LUT used to normalize the diffuse scaling factor.
/// Each element corresponds to a roughness value.
/// Check Section 2.2. in the paper "A Microfacet Based Coupled Specular-Matte BRDF Model with Importance Sampling"
/// The LUT stores 1 - average albedo as a small optimization for the renderer
float sheenAlbedoAvg[SHEEN_LUT_SIZE] = float[] (
	1.00000,
	0.97841,
	0.96276,
	0.94874,
	0.93569,
	0.92349,
	0.91217,
	0.90178,
	0.89235,
	0.88392,
	0.87652,
	0.87017,
	0.86488,
	0.86065,
	0.85749,
	0.85538
);

/// Sample the sheen albedo LUT for a given incident angle and glossiness
/// @param cosTheta Cosine of the angle between the incident direction and the surface normal
/// @param glossiness Sheen glossiness
/// @return Directional sheen albedo for the given incident angle and glossiness.
float sheenDirectionalAlbedo(float cosTheta, float glossiness) {
	float roughness = (1.0 - glossiness);
	float x = cosTheta * float(SHEEN_LUT_SIZE - 1);
	float y = roughness * float(SHEEN_LUT_SIZE - 1);
	int ix = int(x);
	int iy = int(y);
	int ix2 = clamp(ix + 1, 0, SHEEN_LUT_SIZE - 1);
	int iy2 = clamp(iy + 1, 0, SHEEN_LUT_SIZE - 1);
	float fx = x - float(ix);
	float fy = y - float(iy);

	float v1 = (1.0 - fx) * sheenAlbedoLUT[iy  * SHEEN_LUT_SIZE + ix] + fx * sheenAlbedoLUT[iy  * SHEEN_LUT_SIZE + ix2];
	float v2 = (1.0 - fx) * sheenAlbedoLUT[iy2 * SHEEN_LUT_SIZE + ix] + fx * sheenAlbedoLUT[iy2 * SHEEN_LUT_SIZE + ix2];
	float albedo = (1.0 - fy) * v1 + fy * v2;

	return clamp(albedo, 0.0, 1.0);
}

/// Sample the average sheen albedo from the LUT for a given glossiness value
/// @param glossiness Sheen glossiness
/// @return Average sheen albedo for the given glossiness
float getSheenAlbedoAverage(float glossiness) {
	float roughness = 1.0 - glossiness;
	float y = roughness * float(SHEEN_LUT_SIZE - 1);
	int iy0 = int(y);
	int iy1 = clamp(iy0 + 1, 0, SHEEN_LUT_SIZE - 1);
	float fy = y - float(iy0);
	float avg0 = sheenAlbedoAvg[iy0];
	float avg1 = sheenAlbedoAvg[iy1];
	float albedoAvg = (1.0 - fy) * avg0 + fy * avg1;
	return albedoAvg;
}

/// Get the partial sheen albedo scaling factor (without the view direction albedo).
/// Used to dim the diffuse according to section 2.2. in the paper
/// "A Microfacet Based Coupled Specular-Matte BRDF Model with Importance Sampling"
/// The light direction albedo needs to be computed per light direction but
/// the view direction albedo can be computed earlier and used to split the samples 
/// between the diffuse and the sheen layer.
/// @param sheenColor Sheen color
/// @param cosTheta Cosine of the angle between the light direction and the normal
/// @param glossiness Sheen glossiness
/// @return Partial albedo scaling factor
vec3 getSheenAlbedoLightDim(vec3 sheenColor, float cosTheta, float glossiness) {
	float albedoLight = sheenDirectionalAlbedo(max(0.0, cosTheta), glossiness);
	float avgAlbedo = getSheenAlbedoAverage(glossiness); // This is 1 - average albedo
	// No need to check the divisor because it's always large enough for this BRDF
	return (1.0 - sheenColor * albedoLight) / avgAlbedo;
}

vec3 sampleSheenBRDF(VRayMtlInitParams params, VRayMtlContext ctx, int sampleIdx, int nbSamples, out float rayProb, out float brdfContrib) {
	// Sample the hemisphere uniformly
	mat3 localToWorld;
	makeNormalMatrix(ctx.geomNormal, localToWorld);
	vec2 uv = rand(ctx, sampleIdx, nbSamples);
	vec3 dir = localToWorld * getSphereDir(uv.x, uv.y);
	rayProb = INV_2PI;
	float glossyFresnelCoeff = getConductorFresnel(-dot(ctx.e, normalize(dir - ctx.e)), SHEEN_N, SHEEN_K);
	brdfContrib = sheenProbability(ctx.e, dir, ctx.geomNormal, ctx.sheenGloss);
	brdfContrib *= glossyFresnelCoeff;
	return dir;
}

vec3 sampleDiffuseBRDF(VRayMtlInitParams params, VRayMtlContext ctx, int sampleIdx, int nbSamples, out float rayProb, out float brdfContrib) {
	// Sample the hemisphere with cosine distribution
	mat3 localToWorld;
	makeNormalMatrix(ctx.geomNormal, localToWorld);
	vec2 uv = rand(ctx, sampleIdx, nbSamples);
	vec3 dir = localToWorld * getDiffuseDir(uv.x, uv.y);
	rayProb = INV_2PI;
	brdfContrib = 1.0;
	return dir;
}

float pow35(float x) {
	return x * x * x * sqrt(x);
}

/// Blend between a metal color and reflection color based on a dielectric Fresnel value
/// This approximates the tinted reflection visible in some conductors
/// @param metalColor Metallic color, currently derived from the diffuse (base) color and the metalness value.
/// @param reflectionColor The color of the specular highlight
/// @param fresnel Dielectric fresnel coefficient
/// @return Blended reflection color.
vec3 computeMetallicReflection(vec3 metalColor, vec3 reflectionColor, float fresnel) {
	vec3 reflectionDim = reflectionColor * fresnel;
	return metalColor * (1.0 - reflectionDim) + reflectionDim;
}

VRayMtlContext initVRayMtlContext(VRayMtlInitParams initParams) {
	float reflGloss = initParams.reflGloss;
	vec3 Vw = initParams.Vw;
	vec3 geomNormal = initParams.geomNormal;
	vec3 selfIllum = initParams.selfIllum;
	vec3 diffuseColor = initParams.diffuseColor;
	float diffuseAmount = initParams.diffuseAmount;
	vec3 reflColor = initParams.reflColor;
	float reflAmount = initParams.reflAmount;
	bool traceReflections = initParams.traceReflections;
	float metalness = initParams.metalness;
	float aniso = initParams.aniso;
	float anisoRotation = initParams.anisoRotation;
	int anisoAxis = initParams.anisoAxis;
	vec3 opacity = initParams.opacity;
	float roughness = initParams.roughness;
	vec3 refractionColor = initParams.refractionColor;
	float refractionAmount = initParams.refractionAmount;
	bool traceRefractions = initParams.traceRefractions;
	float refractionIOR = initParams.refractionIOR;
	bool useFresnel = initParams.useFresnel;
	float fresnelIOR = initParams.fresnelIOR;
	bool lockFresnelIOR = initParams.lockFresnelIOR;
	bool doubleSided = initParams.doubleSided;
	bool useRoughness = initParams.useRoughness;
	float gtrGamma = initParams.gtrGamma;
	int brdfType = initParams.brdfType;
	float sheenGloss = initParams.sheenGlossiness;
	float coatGloss = initParams.coatGlossiness;

	VRayMtlContext result;
	if (initParams.lockFresnelIOR)
		initParams.fresnelIOR = initParams.refractionIOR;

	result.e = -normalize(Vw);
	 // Invert glossiness (turn it into roughness)
	if (useRoughness) {
		reflGloss = 1.0 - reflGloss;
		coatGloss = 1.0 - coatGloss;
		sheenGloss = 1.0 - sheenGloss;
	}

	result.reflGloss = reflGloss;
	result.opacity = opacity;
	result.diff = diffuseColor * diffuseAmount * result.opacity;
	result.illum = selfIllum * result.opacity;
	// roughness
	float sqrRough = roughness * roughness;
	result.rtermA = 1.0 - 0.5 * (sqrRough / (sqrRough + 0.33));
	result.rtermB = 0.45 * (sqrRough / (sqrRough + 0.09));

	if (doubleSided && dot(geomNormal, result.e) > 0.0)
		geomNormal = -geomNormal;

	vec3 reflectDir = reflect(result.e, geomNormal);
	result.geomNormal = geomNormal;

	// check for internal reflection
	bool internalReflection;
	vec3 refractDir;
	bool outToIn = (dot(geomNormal, result.e) < 0.0);
	float ior = (outToIn ? 1.0 / refractionIOR : refractionIOR);
	vec3 normal = (outToIn ? geomNormal : -geomNormal);

	float cost = -dot(result.e, normal);
	float sintSqr = 1.0 - ior * ior * (1.0 - cost * cost);
	if (sintSqr > 1e-6) {
		internalReflection = false;
		refractDir = ior * result.e + (ior * cost - sqrt(sintSqr)) * normal;
	} else {
		internalReflection = true;
		refractDir = reflectDir;
	}
	float fresnel = 1.0;
	if (useFresnel && !internalReflection)
		fresnel = clamp(getFresnelCoeff(fresnelIOR, result.e, normal, refractDir), 0.0, 1.0);

	vec3 reflNoFresnel = reflColor * reflAmount * result.opacity;
	result.refl = reflNoFresnel * fresnel;

	// Reflection calculation including metalness. Taken from VRayMtl's original implementation.
	vec3 metalColor = result.diff * metalness;

	vec3 dielectricReflectionTransparency = traceReflections ? (1.0 - result.refl) : vec3(1.0);
	vec3 reflectionTransparency = (1.0 - metalness) * dielectricReflectionTransparency;
	if (traceRefractions) {
		result.refr = refractionColor * refractionAmount * result.opacity * reflectionTransparency;
	} else {
		result.refr = vec3(0.0);
	}
	result.diff *= reflectionTransparency - result.refr;

	result.refl = computeMetallicReflection(metalColor, reflNoFresnel, fresnel);

	vec3 sheenColor = initParams.sheenColor * initParams.sheenAmount;
	result.hasSheen = ((sheenColor.x + sheenColor.y + sheenColor.z) > 1e-6) && (1.0 - sheenGloss > 1e-5);
	if (result.hasSheen) {
		float albedoView = sheenDirectionalAlbedo(max(0.0, dot(-result.e, geomNormal)), sheenGloss);
		vec3 sheenViewDim = 1.0 - initParams.sheenColor * albedoView;
		result.diff *= sheenViewDim;
		result.sheen = initParams.sheenColor * (reflectionTransparency - result.refr);
	}

	result.hasCoat = (initParams.coatAmount > 1e-6);
	if (result.hasCoat && !internalReflection) {
		float coatFresnel = clamp(getFresnelCoeff(initParams.coatIOR, result.e, normal, refractDir), 0.0, 1.0);
		float coatAmount = initParams.coatAmount;
		vec3 coatColor = initParams.coatColor * (1.0 - coatFresnel);
		vec3 coatDim = traceReflections ? ((1.0 - coatAmount) + coatAmount * coatColor) : vec3(1.0);
		// Dim all layers below the coat
		result.refl *= coatDim;
		result.refr *= coatDim;
		result.sheen *= coatDim;
		result.diff *= coatDim;
		makeNormalMatrix(geomNormal, result.coatNM);
		result.coat = vec3(1.0) * initParams.coatAmount * coatFresnel;
	}

	result.gloss1 = max(0.0, 1.0 / pow35(max(1.0 - reflGloss, 1e-4)) - 1.0); // [0, 1] -> [0, inf)
	result.roughnessSqr = max(1.0 - reflGloss, 1e-4);
	result.roughnessSqr *= result.roughnessSqr;
	result.coatRoughnessSqr = max(1.0 - coatGloss, 1e-4);
	result.coatRoughnessSqr *= result.coatRoughnessSqr;
	result.sheenGloss = sheenGloss;
	result.gtrGamma = gtrGamma;

	// Set up the normal/inverse normal matrices for BRDFs that support anisotropy
	vec3 anisoDirection = vec3(0.0, 0.0, 1.0);
	if (anisoAxis == 0)
		anisoDirection = vec3(1.0, 0.0, 0.0);
	else if (anisoAxis == 1)
		anisoDirection = vec3(0.0, 1.0, 0.0);
	float anisoAbs = abs(aniso);
	if (anisoAbs < 1e-12 || anisoAbs >= 1.0 - 1e-6) {
		makeNormalMatrix(geomNormal, result.nm);
		result.inm = transpose(result.nm); // inverse = transpose for orthogonal matrix
	} else if (!internalReflection) {
		vec3 base0, base1;
		base0 = normalize(cross(geomNormal, anisoDirection));
		base1 = normalize(cross(base0, geomNormal));
		float anisor = anisoRotation * 6.2831853;
		if (abs(anisor) > 1e-6) {
			float cs = cos(anisor);
			float sn = sin(anisor);
			vec3 nu = base0 * cs - base1 * sn;
			vec3 nv = base0 * sn + base1 * cs;
			base0 = nu;
			base1 = nv;
		}

		if (length(cross(base0, base1)) < 1e-6)
			computeTangentVectors(geomNormal, base0, base1);
		if (aniso > 0.0) {
			float a = 1.0 / (1.0 - aniso);
			base0 *= a;
			base1 /= a;
		} else {
			float a = 1.0 / (1.0 + aniso);
			base0 /= a;
			base1 *= a;
		}
		result.nm[0] = base0;
		result.nm[1] = base1;
		result.nm[2] = geomNormal;
		result.inm = inverse(result.nm);
	}

	return result;
}

/// Lambertian BRDF contribution
vec3 vrayMtlDiffuse(vec3 lightDir, vec3 normal) {
	return vec3(max(0.0, dot(lightDir, normal)));
}

/// Oren-Nayar BRDF contribution
vec3 vrayMtlDiffuseRoughness(vec3 lightDir, VRayMtlContext ctx) {
	float lightNdotL = max(0.0, dot(lightDir, ctx.geomNormal));
	float rmult = 1.0;
	vec3 vecV = -ctx.e;
	float NV = clamp(dot(ctx.geomNormal, vecV), 0.0, 1.0);
	float theta_i = acos(lightNdotL);
	float theta_r = acos(NV);
	float alpha = max(theta_i, theta_r);
	if (alpha > 1.571) { // 1.571==pi/2
		rmult = 0.0;
	} else {
		float beta = min(theta_i, theta_r);
		vec3 vecVtan = vecV - ctx.geomNormal * NV;
		vec3 vecLtan = lightDir - ctx.geomNormal * lightNdotL;
		float fMult = length(vecVtan) * length(vecLtan);
		float cosDeltaPhi = fMult < 0.000001 ? 1.0 : dot(vecVtan, vecLtan) / fMult;
		rmult = (ctx.rtermA + ctx.rtermB * sin(alpha) * tan(beta) * max(0.0, cosDeltaPhi));
	}
	return vec3(lightNdotL * rmult);
}

/// Blinn BRDF contribution
vec3 vrayMtlBlinn(vec3 lightDir, VRayMtlContext ctx) {
	float k = max(0.0, ctx.gloss1);
	vec3 hw = lightDir - ctx.e;
	vec3 hn = normalize(ctx.inm * hw);
	float cs1 = hn.z;
	if (cs1 > 1e-6) {
		float lightNdotL = dot(ctx.geomNormal, lightDir);
		if (cs1 > 1.0)
			cs1 = 1.0;
		float cs = -dot(normalize(hw), ctx.e);
		k = cs < 1e-6 ? 0.0 : pow(cs1, k) * (k + 1.0) * 0.125 / cs;
		k *= lightNdotL;
		if (k > 0.0)
			return vec3(k);
	}
	return vec3(0.0);
}

/// Phong BRDF contribution
vec3 vrayMtlPhong(vec3 lightDir, VRayMtlContext ctx) {
	vec3 reflectDir = reflect(ctx.e, ctx.geomNormal);
	float cs1 = dot(lightDir, reflectDir);
	if (cs1 > 0.0) {
		float lightNdotL = dot(ctx.geomNormal, lightDir);
		if (cs1 > 1.0)
			cs1 = 1.0;
		float k = pow(cs1, ctx.gloss1) * (ctx.gloss1 + 1.0) * 0.5; // phong k
		k *= lightNdotL;
		if (k > 0.0)
			return vec3(k);
	}
	return vec3(0.0);
}

/// Ward BRDF contribution
vec3 vrayMtlWard(vec3 lightDir, VRayMtlContext ctx) {
	float cs1 = -dot(ctx.e, ctx.geomNormal);
	float lightNdotL = dot(ctx.geomNormal, lightDir);
	if (lightNdotL > 1e-6 && cs1 > 1e-6) {
		vec3 hw = lightDir - ctx.e;
		vec3 hn = normalize(ctx.inm * hw);
		if (hn.z > 1e-3) {
			float tanhSqr = (1.0 / (hn.z * hn.z) - 1.0);
			float divd = cs1 * ctx.roughnessSqr;
			float k = exp(-tanhSqr / ctx.roughnessSqr) / divd;
			k *= lightNdotL;
			if (k > 0.0)
				return vec3(k);
		}
	}
	return vec3(0.0);
}

/// GTR BRDF contribution
vec3 vrayMtlGGX(vec3 lightDir, VRayMtlContext ctx) {
	float cs1 = -dot(ctx.e, ctx.geomNormal);
	float lightNdotL = dot(ctx.geomNormal, lightDir);
	if (lightNdotL > 1e-6 && cs1 > 1e-6) {
		vec3 hw = normalize(lightDir - ctx.e);
		vec3 hn = normalize(ctx.inm * hw);
		if (hn.z > 1e-3) {
			float D = getGGXMicrofacetDistribution(hn.z, ctx.roughnessSqr, ctx.gtrGamma);
			float G =
				getGGXBidirectionalShadowingMasking(-ctx.e, lightDir, hw, ctx.geomNormal, ctx.roughnessSqr, ctx.gtrGamma);
			vec3 micron = ctx.nm * hn;
			float L2 = dot(micron, micron);
			float anisotropyCorrection = L2 * sqrt(L2);
			float k = 0.25 * D * G * anisotropyCorrection * PI / cs1; // anisotropy correction
			if (k > 0.0)
				return vec3(k);
		}
	}
	return vec3(0.0);
}

/// GGX BRDF contribution that uses the coat layer parameters
vec3 vrayMtlGGXCoat(vec3 lightDir, VRayMtlContext ctx) {
	float cs1 = -dot(ctx.e, ctx.geomNormal);
	float lightNdotL = dot(ctx.geomNormal, lightDir);
	if (lightNdotL > 1e-6 && cs1 > 1e-6) {
		vec3 hw = normalize(lightDir - ctx.e);
		vec3 hn = normalize(ctx.inm * hw);
		if (hn.z > 1e-3) {
			float D = getGGXMicrofacetDistribution(hn.z, ctx.coatRoughnessSqr, 2.0);
			float G = getGGXBidirectionalShadowingMasking(-ctx.e, lightDir, hw, ctx.geomNormal, ctx.coatRoughnessSqr, 2.0);
			vec3 micron = ctx.coatNM * hn;
			float k = 0.25 * D * G * PI / cs1;
			if (k > 0.0)
				return vec3(k);
		}
	}
	return vec3(0.0);
}

vec3 computeRefractFogContrib(VRayMtlInitParams params, VRayMtlContext ctx, vec3 diffuseContrib) {
	if (intensity(ctx.diff) < 0.001)
		return vec3(0.0);

	vec3 fogColor = params.fogColor;
	float fogMult = max(1e-6, params.fogMult);
	float fogBias = params.fogBias;
	if (fogBias > 0.0) {
		fogBias = 1.0 / (1.0 + fogBias);
	} else {
		fogBias = 1.0 - fogBias;
	}

	float fogDist = params.distToCamera * 0.001;
	fogDist = pow(fogDist, fogBias);
	fogColor = pow(fogColor, vec3(fogMult * fogDist));
	return fogColor * ctx.diff * diffuseContrib;
}

vec3 computeDirectDiffuseContribution(VRayMtlInitParams params, VRayMtlContext ctx, vec3 lightDir) {
	vec3 res = vec3(0.0);
	if (params.roughness < 1e-6) {
		res = vrayMtlDiffuse(lightDir, ctx.geomNormal);
	} else {
		res = vrayMtlDiffuseRoughness(lightDir, ctx);
	}

	if (ctx.hasSheen) {
		vec3 sheenLightDim = getSheenAlbedoLightDim(params.sheenColor, dot(lightDir, ctx.geomNormal), ctx.sheenGloss);
		res *= sheenLightDim;
	}

	return res;
}

vec3 computeDirectReflectionContribution(VRayMtlInitParams params, VRayMtlContext ctx, vec3 lightDir) {
	vec3 res = vec3(0.0);

	if (params.brdfType == 0) {
		res = vrayMtlPhong(lightDir, ctx);
	} else if (params.brdfType == 1) {
		res = vrayMtlBlinn(lightDir, ctx);
	} else if (params.brdfType == 2) {
		res = vrayMtlWard(lightDir, ctx);
	} else /* if (params.brdfType==3) */ {
		res = vrayMtlGGX(lightDir, ctx);
	}
	return res;
}

// Sheen BRDF contribution based on the "Production Friendly Microfacet Sheen BRDF" paper
vec3 computeDirectSheenContribution(VRayMtlInitParams params, VRayMtlContext ctx, vec3 lightDir) {
	vec3 res = vec3(0.0);

	// Use fixed IOR for sheen
	float glossyFresnelCoeff = getConductorFresnel(-dot(ctx.e, normalize(lightDir - ctx.e)), SHEEN_N, SHEEN_K);
	float k = sheenProbability(ctx.e, lightDir, ctx.geomNormal, ctx.sheenGloss);
	res = vec3(k) * glossyFresnelCoeff * 0.5;

	return res;
}

vec3 computeDirectCoatContribution(VRayMtlInitParams params, VRayMtlContext ctx, vec3 lightDir) {
	return vrayMtlGGXCoat(lightDir, ctx);
}

vec3 computeIndirectDiffuseContribution(VRayMtlInitParams params, VRayMtlContext ctx) {
	vec3 res = vec3(0.0);

	if (ctx.hasSheen) {
		int numSamples = NUM_ENV_SAMPLES + int(float(NUM_ENV_SAMPLES_ROUGH) * 2.0);
		float invNumSamples = 1.0 / float(numSamples);
		vec3 envSum = vec3(0.0);
		for (int i = 0; i < numSamples; ++i) {
			float brdfContrib = 0.0;
			float rayProb = 0.0;
			vec3 dir = sampleDiffuseBRDF(params, ctx, i, numSamples, rayProb, brdfContrib);
			vec3 sheenLightDim = getSheenAlbedoLightDim(params.sheenColor, dot(dir, ctx.geomNormal), ctx.sheenGloss);
			float lod = computeEnvLOD(dir, rayProb, numSamples);
			envSum += engTextureEnvMapLOD(dir, lod) * brdfContrib * sheenLightDim;
		}
		res += envSum * invNumSamples;
	} else {
		res = engEnvIrradiance(params.geomNormal);
	}

	return res;
}

vec3 computeIndirectReflectionContribution(VRayMtlInitParams params, VRayMtlContext ctx) {
	vec3 res = vec3(0.0);

	if (!params.traceReflections)
		return res;

	int numSamples = NUM_ENV_SAMPLES + int(float(NUM_ENV_SAMPLES_ROUGH) * (params.aniso + 0.5 * ctx.roughnessSqr));
	if (ctx.roughnessSqr < 0.0001)
		numSamples = 1;
	float invNumSamples = 1.0 / float(numSamples);
	vec3 envSum = vec3(0.0);
	for (int i = 0; i < numSamples; ++i) {
		float brdfContrib = 0.0;
		float rayProb = 0.0;
		vec3 dir = sampleBRDF(params, ctx, i, numSamples, rayProb, brdfContrib);
		if (brdfContrib < 1e-6)
			continue;
		float lod = computeEnvLOD(dir, rayProb, numSamples);
		envSum += engTextureEnvMapLOD(dir, lod) * brdfContrib;
	}
	res += envSum * invNumSamples;

	return res;
}

vec3 computeIndirectRefractionContribution(
	VRayMtlInitParams params, VRayMtlContext ctx, float alpha, vec3 alphaDir) {
	vec3 res = vec3(0.0);

	if (!params.traceRefractions)
		return res;

	int numSamples = NUM_ENV_SAMPLES + int(float(NUM_ENV_SAMPLES_ROUGH) * params.refrGloss);
	float invNumSamples = 1.0 / float(numSamples);
	vec3 view = -params.Vw;

	if (alpha <= 0.999) {
		res += engTextureEnvMapLOD(alphaDir, 0.0);
	} else {
		vec3 envSum = vec3(0.0);
		for (int i = 0; i < numSamples; ++i) {
			bool totalInternalReflection;
			vec3 dir = sampleRefractBRDF(params, ctx, i, numSamples, totalInternalReflection);
			if (totalInternalReflection) {
				envSum += engTextureEnvMapLOD(dir, 0.0);
			} else {
				envSum += engTextureEnvMapLOD(dir, 0.0);
			}
		}
		res += envSum * invNumSamples;
		vec3 diffuseContrib = computeIndirectDiffuseContribution(params, ctx);
		res += computeRefractFogContrib(params, ctx, diffuseContrib);
	}

	return res;
}

vec3 computeIndirectSheenContribution(VRayMtlInitParams params, VRayMtlContext ctx) {
	vec3 res = vec3(0.0);

	if (!params.traceReflections || !ctx.hasSheen)
		return res;

	int numSamples = NUM_ENV_SAMPLES + int(float(NUM_ENV_SAMPLES_ROUGH) * 0.5 * (1.0 - ctx.sheenGloss));
	float invNumSamples = 1.0 / float(numSamples);
	vec3 envSum = vec3(0.0);
	for (int i = 0; i < numSamples; ++i) {
		float brdfContrib = 0.0;
		float rayProb = 0.0;
		vec3 dir = sampleSheenBRDF(params, ctx, i, numSamples, rayProb, brdfContrib);
		if (brdfContrib < 1e-6)
			continue;
		float lod = computeEnvLOD(dir, rayProb, numSamples);
		envSum += engTextureEnvMapLOD(dir, lod) * brdfContrib;
	}
	res += envSum * invNumSamples;

	return res;
}

vec3 computeIndirectCoatContribution(VRayMtlInitParams params, VRayMtlContext ctx) {
	vec3 res = vec3(0.0);

	if (!params.traceReflections || !ctx.hasCoat)
		return res;

	int numSamples = NUM_ENV_SAMPLES + int(float(NUM_ENV_SAMPLES_ROUGH) * 0.5 * ctx.coatRoughnessSqr);
	if (ctx.coatRoughnessSqr < 0.0001)
		numSamples = 1;
	float invNumSamples = 1.0 / float(numSamples);
	vec3 envSum = vec3(0.0);
	for (int i = 0; i < numSamples; ++i) {
		float brdfContrib = 0.0;
		float rayProb = 0.0;
		vec3 dir = sampleCoatBRDF(params, ctx, i, numSamples, rayProb, brdfContrib);
		if (brdfContrib < 1e-6)
			continue;
		float lod = computeEnvLOD(dir, rayProb, numSamples);
		envSum += engTextureEnvMapLOD(dir, lod) * brdfContrib;
	}
	res += envSum * invNumSamples;

	return res;
}

//////////////////////////////////////////////////////////////////////
// End of VRayMtl implementation.
// Following code implements a very simple raytracer
// and sets up the material parameters.
// You can run it on ShaderToy or with the VSCode Shader Toy plugin
//////////////////////////////////////////////////////////////////////



// presets

struct VRayMtlPreset {
	vec3 diffuseColor;
	float roughness;
	vec3 reflColor;
	float reflGloss;
	float metalness;
	float aniso;
	float anisoRotation;
	int anisoAxis;
	vec3 refractionColor;
	float refrGloss;
	float refractionIOR;
	bool useRoughness;
	vec3 fogColor;
	float fogMult;
	vec3 sheenColor;
	float sheenGlossiness;
	vec3 coatColor;
	float coatAmount;
	float coatGlossiness;
};

#define PRESET_COUNT 24

const VRayMtlPreset gPresets[PRESET_COUNT] = VRayMtlPreset[PRESET_COUNT](
// aluminium
	VRayMtlPreset(
/* diffuseColor    */ vec3(0.9019, 0.9137, 0.9215),
/* roughness	   */ 0.0,
/* reflColor	   */ vec3(1.0, 1.0, 1.0),
/* reflGloss	   */ 0.0,
/* metalness	   */ 1.0,
/* aniso		   */ 0.0,
/* anisoRotation   */ 0.0,
/* anisoAxis	   */ 2,
/* refractionColor */ vec3(0.0, 0.0, 0.0),
/* refrGloss	   */ 1.0,
/* refractionIOR   */ 1.002,
/* useRoughness    */ true,
/* fogColor		   */ vec3(1.0, 1.0, 1.0),
/* fogMult		   */ 1.0,
/* sheenColor	   */ vec3(0.0, 0.0, 0.0),
/* sheenGlossiness */ 1.0,
/* coatColor	   */ vec3(0.0, 0.0, 0.0),
/* coatAmount      */ 0.0f,
/* coatGlossiness  */ 1.0),

// aluminium (rough)
	VRayMtlPreset(
/* diffuseColor    */ vec3(0.9019, 0.9137, 0.9215),
/* roughness	   */ 0.0,
/* reflColor	   */ vec3(1.0, 1.0, 1.0),
/* reflGloss	   */ 0.12,
/* metalness	   */ 1.0,
/* aniso		   */ 0.0,
/* anisoRotation   */ 0.0,
/* anisoAxis	   */ 2,
/* refractionColor */ vec3(0.0, 0.0, 0.0),
/* refrGloss	   */ 1.0,
/* refractionIOR   */ 1.002,
/* useRoughness    */ true,
/* fogColor		   */ vec3(1.0, 1.0, 1.0),
/* fogMult		   */ 1.0,
/* sheenColor	   */ vec3(0.0, 0.0, 0.0),
/* sheenGlossiness */ 1.0,
/* coatColor	   */ vec3(0.0, 0.0, 0.0),
/* coatAmount      */ 0.0f,
/* coatGlossiness  */ 1.0),

// aluminium (brushed)
	VRayMtlPreset(
/* diffuseColor    */ vec3(0.9019, 0.9137, 0.9215),
/* roughness	   */ 0.0,
/* reflColor	   */ vec3(1.0, 1.0, 1.0),
/* reflGloss	   */ 0.3,
/* metalness	   */ 1.0,
/* aniso		   */ 0.8,
/* anisoRotation   */ 0.0,
/* anisoAxis	   */ 1,
/* refractionColor */ vec3(0.0, 0.0, 0.0),
/* refrGloss	   */ 1.0,
/* refractionIOR   */ 1.002,
/* useRoughness    */ true,
/* fogColor		   */ vec3(1.0, 1.0, 1.0),
/* fogMult		   */ 1.0,
/* sheenColor	   */ vec3(0.0, 0.0, 0.0),
/* sheenGlossiness */ 1.0,
/* coatColor	   */ vec3(0.0, 0.0, 0.0),
/* coatAmount      */ 0.0f,
/* coatGlossiness  */ 1.0),

// chrome
	VRayMtlPreset(
/* diffuseColor    */ vec3(0.5529, 0.5529, 0.5529),
/* roughness	   */ 0.0,
/* reflColor	   */ vec3(1.0, 1.0, 1.0),
/* reflGloss	   */ 0.0,
/* metalness	   */ 1.0,
/* aniso		   */ 0.0,
/* anisoRotation   */ 0.0,
/* anisoAxis	   */ 2,
/* refractionColor */ vec3(0.0, 0.0, 0.0),
/* refrGloss	   */ 1.0,
/* refractionIOR   */ 1.03,
/* useRoughness    */ true,
/* fogColor		   */ vec3(1.0, 1.0, 1.0),
/* fogMult		   */ 1.0,
/* sheenColor	   */ vec3(0.0, 0.0, 0.0),
/* sheenGlossiness */ 1.0,
/* coatColor	   */ vec3(0.0, 0.0, 0.0),
/* coatAmount      */ 0.0f,
/* coatGlossiness  */ 1.0),

// copper
	VRayMtlPreset(
/* diffuseColor    */ vec3(0.9333, 0.6196, 0.5372),
/* roughness	   */ 0.0,
/* reflColor	   */ vec3(1.0, 1.0, 1.0),
/* reflGloss	   */ 0.0,
/* metalness	   */ 1.0,
/* aniso		   */ 0.0,
/* anisoRotation   */ 0.0,
/* anisoAxis	   */ 2,
/* refractionColor */ vec3(0.0, 0.0, 0.0),
/* refrGloss	   */ 1.0,
/* refractionIOR   */ 1.21901,
/* useRoughness    */ true,
/* fogColor		   */ vec3(1.0, 1.0, 1.0),
/* fogMult		   */ 1.0,
/* sheenColor	   */ vec3(0.0, 0.0, 0.0),
/* sheenGlossiness */ 1.0,
/* coatColor	   */ vec3(0.0, 0.0, 0.0),
/* coatAmount      */ 0.0f,
/* coatGlossiness  */ 1.0),

// copper (rough)
	VRayMtlPreset(
/* diffuseColor    */ vec3(0.9333, 0.6196, 0.5372),
/* roughness	   */ 0.0,
/* reflColor	   */ vec3(1.0, 1.0, 1.0),
/* reflGloss	   */ 0.1,
/* metalness	   */ 1.0,
/* aniso		   */ 0.0,
/* anisoRotation   */ 0.0,
/* anisoAxis	   */ 2,
/* refractionColor */ vec3(0.0, 0.0, 0.0),
/* refrGloss	   */ 1.0,
/* refractionIOR   */ 1.21901,
/* useRoughness    */ true,
/* fogColor		   */ vec3(1.0, 1.0, 1.0),
/* fogMult		   */ 1.0,
/* sheenColor	   */ vec3(0.0, 0.0, 0.0),
/* sheenGlossiness */ 1.0,
/* coatColor	   */ vec3(0.0, 0.0, 0.0),
/* coatAmount      */ 0.0f,
/* coatGlossiness  */ 1.0),

// gold
	VRayMtlPreset(
/* diffuseColor    */ vec3(0.9529, 0.7882, 0.4078),
/* roughness	   */ 0.0,
/* reflColor	   */ vec3(1.0, 1.0, 1.0),
/* reflGloss	   */ 0.0,
/* metalness	   */ 1.0,
/* aniso		   */ 0.0,
/* anisoRotation   */ 0.0,
/* anisoAxis	   */ 2,
/* refractionColor */ vec3(0.0, 0.0, 0.0),
/* refrGloss	   */ 1.0,
/* refractionIOR   */ 1.35002,
/* useRoughness    */ true,
/* fogColor		   */ vec3(1.0, 1.0, 1.0),
/* fogMult		   */ 1.0,
/* sheenColor	   */ vec3(0.0, 0.0, 0.0),
/* sheenGlossiness */ 1.0,
/* coatColor	   */ vec3(0.0, 0.0, 0.0),
/* coatAmount      */ 0.0f,
/* coatGlossiness  */ 1.0),

// gold (rough)
	VRayMtlPreset(
/* diffuseColor    */ vec3(0.9529, 0.7882, 0.4078),
/* roughness	   */ 0.0,
/* reflColor	   */ vec3(1.0, 1.0, 1.0),
/* reflGloss	   */ 0.15,
/* metalness	   */ 1.0,
/* aniso		   */ 0.0,
/* anisoRotation   */ 0.0,
/* anisoAxis	   */ 2,
/* refractionColor */ vec3(0.0, 0.0, 0.0),
/* refrGloss	   */ 1.0,
/* refractionIOR   */ 1.35002,
/* useRoughness    */ true,
/* fogColor		   */ vec3(1.0, 1.0, 1.0),
/* fogMult		   */ 1.0,
/* sheenColor	   */ vec3(0.0, 0.0, 0.0),
/* sheenGlossiness */ 1.0,
/* coatColor	   */ vec3(0.0, 0.0, 0.0),
/* coatAmount      */ 0.0f,
/* coatGlossiness  */ 1.0),

// iron
	VRayMtlPreset(
/* diffuseColor    */ vec3(0.8862, 0.8745, 0.8235),
/* roughness	   */ 0.0,
/* reflColor	   */ vec3(1.0, 1.0, 1.0),
/* reflGloss	   */ 0.0,
/* metalness	   */ 1.0,
/* aniso		   */ 0.0,
/* anisoRotation   */ 0.0,
/* anisoAxis	   */ 2,
/* refractionColor */ vec3(0.0, 0.0, 0.0),
/* refrGloss	   */ 1.0,
/* refractionIOR   */ 1.006,
/* useRoughness    */ true,
/* fogColor		   */ vec3(1.0, 1.0, 1.0),
/* fogMult		   */ 1.0,
/* sheenColor	   */ vec3(0.0, 0.0, 0.0),
/* sheenGlossiness */ 1.0,
/* coatColor	   */ vec3(0.0, 0.0, 0.0),
/* coatAmount      */ 0.0f,
/* coatGlossiness  */ 1.0),

// lead
	VRayMtlPreset(
/* diffuseColor    */ vec3(0.6549, 0.6588, 0.6901),
/* roughness	   */ 0.0,
/* reflColor	   */ vec3(1.0, 1.0, 1.0),
/* reflGloss	   */ 0.0,
/* metalness	   */ 1.0,
/* aniso		   */ 0.0,
/* anisoRotation   */ 0.0,
/* anisoAxis	   */ 2,
/* refractionColor */ vec3(0.0, 0.0, 0.0),
/* refrGloss	   */ 1.0,
/* refractionIOR   */ 1.016,
/* useRoughness    */ true,
/* fogColor		   */ vec3(1.0, 1.0, 1.0),
/* fogMult		   */ 1.0,
/* sheenColor	   */ vec3(0.0, 0.0, 0.0),
/* sheenGlossiness */ 1.0,
/* coatColor	   */ vec3(0.0, 0.0, 0.0),
/* coatAmount      */ 0.0f,
/* coatGlossiness  */ 1.0),

// silver
	VRayMtlPreset(
/* diffuseColor    */ vec3(0.9882, 0.9803, 0.9764),
/* roughness	   */ 0.0,
/* reflColor	   */ vec3(1.0, 1.0, 1.0),
/* reflGloss	   */ 0.0,
/* metalness	   */ 1.0,
/* aniso		   */ 0.0,
/* anisoRotation   */ 0.0,
/* anisoAxis	   */ 2,
/* refractionColor */ vec3(0.0, 0.0, 0.0),
/* refrGloss	   */ 1.0,
/* refractionIOR   */ 1.082,
/* useRoughness    */ true,
/* fogColor		   */ vec3(1.0, 1.0, 1.0),
/* fogMult		   */ 1.0,
/* sheenColor	   */ vec3(0.0, 0.0, 0.0),
/* sheenGlossiness */ 1.0,
/* coatColor	   */ vec3(0.0, 0.0, 0.0),
/* coatAmount      */ 0.0f,
/* coatGlossiness  */ 1.0),

// silver (rough)
	VRayMtlPreset(
/* diffuseColor    */ vec3(0.9882, 0.9803, 0.9764),
/* roughness	   */ 0.0,
/* reflColor	   */ vec3(1.0, 1.0, 1.0),
/* reflGloss	   */ 0.11,
/* metalness	   */ 1.0,
/* aniso		   */ 0.0,
/* anisoRotation   */ 0.0,
/* anisoAxis	   */ 2,
/* refractionColor */ vec3(0.0, 0.0, 0.0),
/* refrGloss	   */ 1.0,
/* refractionIOR   */ 1.082,
/* useRoughness    */ true,
/* fogColor		   */ vec3(1.0, 1.0, 1.0),
/* fogMult		   */ 1.0,
/* sheenColor	   */ vec3(0.0, 0.0, 0.0),
/* sheenGlossiness */ 1.0,
/* coatColor	   */ vec3(0.0, 0.0, 0.0),
/* coatAmount      */ 0.0f,
/* coatGlossiness  */ 1.0),

// diamond
	VRayMtlPreset(
/* diffuseColor    */ vec3(0.0, 0.0, 0.0),
/* roughness	   */ 0.0,
/* reflColor	   */ vec3(1.0, 1.0, 1.0),
/* reflGloss	   */ 0.98,
/* metalness	   */ 0.0,
/* aniso		   */ 0.0,
/* anisoRotation   */ 0.0,
/* anisoAxis	   */ 2,
/* refractionColor */ vec3(1.0, 1.0, 1.0),
/* refrGloss	   */ 1.0,
/* refractionIOR   */ 2.42,
/* useRoughness    */ false,
/* fogColor		   */ vec3(1.0, 1.0, 1.0),
/* fogMult		   */ 1.0,
/* sheenColor	   */ vec3(0.0, 0.0, 0.0),
/* sheenGlossiness */ 1.0,
/* coatColor	   */ vec3(0.0, 0.0, 0.0),
/* coatAmount      */ 0.0f,
/* coatGlossiness  */ 1.0),

// glass
	VRayMtlPreset(
/* diffuseColor    */ vec3(0.0, 0.0, 0.0),
/* roughness	   */ 0.0,
/* reflColor	   */ vec3(0.914, 0.914, 0.914),
/* reflGloss	   */ 1.0,
/* metalness	   */ 0.0,
/* aniso		   */ 0.0,
/* anisoRotation   */ 0.0,
/* anisoAxis	   */ 2,
/* refractionColor */ vec3(0.977, 0.977, 0.977),
/* refrGloss	   */ 1.0,
/* refractionIOR   */ 1.517,
/* useRoughness    */ false,
/* fogColor		   */ vec3(1.0, 1.0, 1.0),
/* fogMult		   */ 1.0,
/* sheenColor	   */ vec3(0.0, 0.0, 0.0),
/* sheenGlossiness */ 1.0,
/* coatColor	   */ vec3(0.0, 0.0, 0.0),
/* coatAmount      */ 0.0f,
/* coatGlossiness  */ 1.0),

// glass (frosted)
	VRayMtlPreset(
/* diffuseColor    */ vec3(0.0, 0.0, 0.0),
/* roughness	   */ 0.0,
/* reflColor	   */ vec3(0.914, 0.914, 0.914),
/* reflGloss	   */ 0.75,
/* metalness	   */ 0.0,
/* aniso		   */ 0.0,
/* anisoRotation   */ 0.0,
/* anisoAxis	   */ 2,
/* refractionColor */ vec3(0.977, 0.977, 0.977),
/* refrGloss	   */ 0.8,
/* refractionIOR   */ 1.517,
/* useRoughness    */ false,
/* fogColor		   */ vec3(1.0, 1.0, 1.0),
/* fogMult		   */ 1.0,
/* sheenColor	   */ vec3(0.0, 0.0, 0.0),
/* sheenGlossiness */ 1.0,
/* coatColor	   */ vec3(0.0, 0.0, 0.0),
/* coatAmount      */ 0.0f,
/* coatGlossiness  */ 1.0),

// glass (tinted)
	VRayMtlPreset(
/* diffuseColor    */ vec3(0.0, 0.0, 0.0),
/* roughness	   */ 0.0,
/* reflColor	   */ vec3(0.914, 0.914, 0.914),
/* reflGloss	   */ 1.0,
/* metalness	   */ 0.0,
/* aniso		   */ 0.0,
/* anisoRotation   */ 0.0,
/* anisoAxis	   */ 2,
/* refractionColor */ vec3(0.977, 0.977, 0.977),
/* refrGloss	   */ 1.0,
/* refractionIOR   */ 1.517,
/* useRoughness    */ false,
/* fogColor		   */ vec3(0.702, 0.95, 0.702),
/* fogMult		   */ 1.0,
/* sheenColor	   */ vec3(0.0, 0.0, 0.0),
/* sheenGlossiness */ 1.0,
/* coatColor	   */ vec3(0.0, 0.0, 0.0),
/* coatAmount      */ 0.0f,
/* coatGlossiness  */ 1.0),

// water
	VRayMtlPreset(
/* diffuseColor    */ vec3(0.0, 0.0, 0.0),
/* roughness	   */ 0.0,
/* reflColor	   */ vec3(0.784, 0.784, 0.784),
/* reflGloss	   */ 1.0,
/* metalness	   */ 0.0,
/* aniso		   */ 0.0,
/* anisoRotation   */ 0.0,
/* anisoAxis	   */ 2,
/* refractionColor */ vec3(0.0, 0.0, 0.0),
/* refrGloss	   */ 1.0,
/* refractionIOR   */ 1.333,
/* useRoughness    */ false,
/* fogColor		   */ vec3(1.0, 1.0, 1.0),
/* fogMult		   */ 1.0,
/* sheenColor	   */ vec3(0.0, 0.0, 0.0),
/* sheenGlossiness */ 1.0,
/* coatColor	   */ vec3(0.0, 0.0, 0.0),
/* coatAmount      */ 0.0f,
/* coatGlossiness  */ 1.0),

// chocolate
	VRayMtlPreset(
/* diffuseColor    */ vec3(0.032, 0.019, 0.009),
/* roughness	   */ 0.0,
/* reflColor	   */ vec3(0.442, 0.442, 0.442),
/* reflGloss	   */ 0.68,
/* metalness	   */ 0.0,
/* aniso		   */ 0.0,
/* anisoRotation   */ 0.0,
/* anisoAxis	   */ 2,
/* refractionColor */ vec3(0.195, 0.195, 0.195),
/* refrGloss	   */ 0.6,
/* refractionIOR   */ 1.59,
/* useRoughness    */ false,
/* fogColor		   */ vec3(1.0, 1.0, 1.0),
/* fogMult		   */ 1.0,
/* sheenColor	   */ vec3(0.0, 0.0, 0.0),
/* sheenGlossiness */ 1.0,
/* coatColor	   */ vec3(0.0, 0.0, 0.0),
/* coatAmount      */ 0.0f,
/* coatGlossiness  */ 1.0),

// ceramic
	VRayMtlPreset(
/* diffuseColor    */ vec3(0.7764, 0.6941, 0.6352),
/* roughness	   */ 0.6,
/* reflColor	   */ vec3(0.996, 1.0, 0.988),
/* reflGloss	   */ 0.99,
/* metalness	   */ 0.0,
/* aniso		   */ 0.0,
/* anisoRotation   */ 0.0,
/* anisoAxis	   */ 2,
/* refractionColor */ vec3(0.0, 0.0, 0.0),
/* refrGloss	   */ 1.0,
/* refractionIOR   */ 1.6,
/* useRoughness    */ false,
/* fogColor		   */ vec3(1.0, 1.0, 1.0),
/* fogMult		   */ 1.0,
/* sheenColor	   */ vec3(0.0, 0.0, 0.0),
/* sheenGlossiness */ 1.0,
/* coatColor	   */ vec3(0.0, 0.0, 0.0),
/* coatAmount      */ 0.0f,
/* coatGlossiness  */ 1.0),

// plastic
	VRayMtlPreset(
/* diffuseColor    */ vec3(0.0627, 0.0588, 0.0627),
/* roughness	   */ 0.0,
/* reflColor	   */ vec3(0.9725, 0.9607, 1.0),
/* reflGloss	   */ 0.98,
/* metalness	   */ 0.0,
/* aniso		   */ 0.0,
/* anisoRotation   */ 0.0,
/* anisoAxis	   */ 2,
/* refractionColor */ vec3(0.0, 0.0, 0.0),
/* refrGloss	   */ 0.6,
/* refractionIOR   */ 1.46,
/* useRoughness    */ false,
/* fogColor		   */ vec3(1.0, 1.0, 1.0),
/* fogMult		   */ 1.0,
/* sheenColor	   */ vec3(0.0, 0.0, 0.0),
/* sheenGlossiness */ 1.0,
/* coatColor	   */ vec3(0.0, 0.0, 0.0),
/* coatAmount      */ 0.0f,
/* coatGlossiness  */ 1.0),

// rubber
	VRayMtlPreset(
/* diffuseColor    */ vec3(0.008, 0.01, 0.01),
/* roughness	   */ 0.0,
/* reflColor	   */ vec3(0.929, 0.975, 1.0),
/* reflGloss	   */ 0.472,
/* metalness	   */ 0.0,
/* aniso		   */ 0.0,
/* anisoRotation   */ 0.0,
/* anisoAxis	   */ 2,
/* refractionColor */ vec3(0.0, 0.0, 0.0),
/* refrGloss	   */ 1.0,
/* refractionIOR   */ 1.468,
/* useRoughness    */ false,
/* fogColor		   */ vec3(1.0, 1.0, 1.0),
/* fogMult		   */ 1.0,
/* sheenColor	   */ vec3(0.0, 0.0, 0.0),
/* sheenGlossiness */ 1.0,
/* coatColor	   */ vec3(0.0, 0.0, 0.0),
/* coatAmount      */ 0.0f,
/* coatGlossiness  */ 1.0),

// generic rough white ball
	VRayMtlPreset(
/* diffuseColor    */ vec3(1.0, 1.0, 1.0),
/* roughness	   */ 1.0,
/* reflColor	   */ vec3(1.0, 1.0, 1.0),
/* reflGloss	   */ 0.0,
/* metalness	   */ 0.0,
/* aniso		   */ 0.0,
/* anisoRotation   */ 0.0,
/* anisoAxis	   */ 2,
/* refractionColor */ vec3(0.0, 0.0, 0.0),
/* refrGloss	   */ 1.0,
/* refractionIOR   */ 1.6,
/* useRoughness    */ false,
/* fogColor		   */ vec3(1.0, 1.0, 1.0),
/* fogMult		   */ 1.0,
/* sheenColor	   */ vec3(0.0, 0.0, 0.0),
/* sheenGlossiness */ 1.0,
/* coatColor	   */ vec3(0.0, 0.0, 0.0),
/* coatAmount      */ 0.0f,
/* coatGlossiness  */ 1.0),

// basic sheen
	VRayMtlPreset(
/* diffuseColor    */ vec3(0.25, 0.25, 0.25),
/* roughness	   */ 0.0,
/* reflColor	   */ vec3(0.0, 0.0, 0.0),
/* reflGloss	   */ 0.0,
/* metalness	   */ 0.0,
/* aniso		   */ 0.0,
/* anisoRotation   */ 0.0,
/* anisoAxis	   */ 2,
/* refractionColor */ vec3(0.0, 0.0, 0.0),
/* refrGloss	   */ 1.0,
/* refractionIOR   */ 1.6,
/* useRoughness    */ false,
/* fogColor		   */ vec3(1.0, 1.0, 1.0),
/* fogMult		   */ 1.0,
/* sheenColor	   */ vec3(0.0, 0.0, 1.0),
/* sheenGlossiness */ 0.85,
/* coatColor	   */ vec3(0.0, 0.0, 0.0),
/* coatAmount      */ 0.0,
/* coatGlossiness  */ 1.0),

// basic coat
	VRayMtlPreset(
/* diffuseColor    */ vec3(0.0, 1.0, 1.0),
/* roughness	   */ 0.0,
/* reflColor	   */ vec3(1.0, 1.0, 1.0),
/* reflGloss	   */ 0.4,
/* metalness	   */ 0.0,
/* aniso		   */ 0.0,
/* anisoRotation   */ 0.0,
/* anisoAxis	   */ 2,
/* refractionColor */ vec3(0.0, 0.0, 0.0),
/* refrGloss	   */ 1.0,
/* refractionIOR   */ 1.6,
/* useRoughness    */ false,
/* fogColor		   */ vec3(1.0, 1.0, 1.0),
/* fogMult		   */ 1.0,
/* sheenColor	   */ vec3(0.0, 0.0, 0.0),
/* sheenGlossiness */ 1.0,
/* coatColor	   */ vec3(1.0, 0.0, 1.0),
/* coatAmount      */ 1.0,
/* coatGlossiness  */ 0.95)

); // end presets

// Get the index of the preset to apply
// @param sweepFactor Number from 0 to 1 used to animate the preset switch with a screen sweep effect. Set to fragCoord.x / iResolution.x
int getPresetIdx(float sweepFactor) {
	const int totalPresets = PRESET_COUNT;
	const float secsPerPreset = 5.0;
	const float secsSweep = 0.75; // included in secsPerPreset
	const float secsPerCycle = secsPerPreset * float(totalPresets);
	float cycleTime = mod(iTime, secsPerCycle);
	int presetIdx = int((cycleTime + secsSweep * sweepFactor) / secsPerPreset);
	if (presetIdx >= totalPresets) {
		presetIdx = 0;
	}
	return presetIdx;
}


void initPresetParams(inout VRayMtlInitParams initParams, float sweepFactor) {
	int presetIdx = getPresetIdx(sweepFactor);
	if (presetIdx >= 0 && presetIdx < PRESET_COUNT) {
		initParams.diffuseColor = gPresets[presetIdx].diffuseColor;
		initParams.roughness = gPresets[presetIdx].roughness;
		initParams.reflColor = gPresets[presetIdx].reflColor;
		initParams.reflGloss = gPresets[presetIdx].reflGloss;
		initParams.metalness = gPresets[presetIdx].metalness;
		initParams.aniso = gPresets[presetIdx].aniso;
		initParams.anisoRotation = gPresets[presetIdx].anisoRotation;
		initParams.anisoAxis = gPresets[presetIdx].anisoAxis;
		initParams.refractionColor = gPresets[presetIdx].refractionColor;
		initParams.refrGloss = gPresets[presetIdx].refrGloss;
		initParams.refractionIOR = gPresets[presetIdx].refractionIOR;
		initParams.useRoughness = gPresets[presetIdx].useRoughness;
		initParams.fogColor = gPresets[presetIdx].fogColor;
		initParams.fogMult = gPresets[presetIdx].fogMult;
		initParams.sheenColor = gPresets[presetIdx].sheenColor;
		initParams.sheenGlossiness = gPresets[presetIdx].sheenGlossiness;
		initParams.coatColor = gPresets[presetIdx].coatColor;
		initParams.coatAmount = gPresets[presetIdx].coatAmount;
		initParams.coatGlossiness = gPresets[presetIdx].coatGlossiness;
	}
}


vec3 shade(vec3 point, vec3 normal, vec3 eyeDir, float distToCamera, float sweepFactor, float fragmentNoise) {
	// Init VRayMtl with defaults
	VRayMtlInitParams initParams;
	initParams.Vw = normalize(eyeDir);
	initParams.geomNormal = normal;
	initParams.diffuseColor = vec3(0.5, 0.5, 0.5);
	initParams.diffuseAmount = 1.0;
	initParams.roughness = 0.0;
	initParams.selfIllum = vec3(0);
	initParams.reflColor = vec3(0.5, 0.5, 0.5);
	initParams.reflAmount = 1.0;
	initParams.reflGloss = 1.0;
	initParams.traceReflections = true;
	initParams.metalness = 0.0;
	initParams.aniso = 0.0;
	initParams.anisoRotation = 0.0;
	initParams.anisoAxis = 2;
	initParams.opacity = vec3(1, 1, 1);
	initParams.refractionColor = vec3(0.0, 0.0, 0.0);
	initParams.refractionAmount = 1.0;
	initParams.refrGloss = 1.0;
	initParams.refractionIOR = 1.6;
	initParams.traceRefractions = true;
	initParams.useFresnel = true;
	initParams.fresnelIOR = 1.6;
	initParams.lockFresnelIOR = true;
	initParams.doubleSided = true;
	initParams.useRoughness = false;
	initParams.gtrGamma = 2.0;
	initParams.brdfType = 3;
	initParams.fogColor = vec3(0.0, 0.0, 0.0);
	initParams.fogBias = 0.0;
	initParams.sssOn = false;
	initParams.sheenAmount = 1.0;
	initParams.coatIOR = 1.6;
	// unused yet - corresponds to translucency color in SSS
	initParams.translucencyColor = vec3(1.0);
	initParams.sssFwdBackCoeff = 0.0;
	initParams.sssScatterCoeff = 0.0;
	initParams.thickness = 1000.0;
	initParams.distToCamera = distToCamera;
	initPresetParams(initParams, sweepFactor);

	// Init context and sample material
	VRayMtlContext ctx = initVRayMtlContext(initParams);
	ctx.fragmentNoise = fragmentNoise;
	vec3 lightDir = normalize(vec3(1, 1, 0.2));
	vec3 diffuseDirect = computeDirectDiffuseContribution(initParams, ctx, lightDir);
	vec3 diffuseIndirect = computeIndirectDiffuseContribution(initParams, ctx);
	vec3 diffuse = diffuseDirect + diffuseIndirect;
	vec3 reflDirect = computeDirectReflectionContribution(initParams, ctx, lightDir);
	vec3 reflIndirect = computeIndirectReflectionContribution(initParams, ctx);
	vec3 reflection = reflDirect + reflIndirect;
	vec3 sheen = vec3(0.0);
	if (ctx.hasSheen) {
		vec3 sheenDirect = computeDirectSheenContribution(initParams, ctx, lightDir);
		vec3 sheenIndirect = computeIndirectSheenContribution(initParams, ctx);
		sheen = sheenDirect + sheenIndirect;
	}

	vec3 coat = vec3(0.0);
	if (ctx.hasCoat) {
		vec3 coatDirect = computeDirectCoatContribution(initParams, ctx, lightDir);
		vec3 coatIndirect = computeIndirectCoatContribution(initParams, ctx);
		coat = coatDirect + coatIndirect;
	}
	
	float alpha = intensity(ctx.opacity);
	vec3 refraction = computeRefractFogContrib(initParams, ctx, diffuseDirect)
		+ computeIndirectRefractionContribution(initParams, ctx, alpha, -initParams.Vw);

	return diffuse * ctx.diff + reflection * ctx.refl + ctx.illum + refraction * ctx.refr + sheen * ctx.sheen + coat * ctx.coat;
}


// simple raytracing of a sphere
const float INFINITY = 100000.0;
float raySphere(vec3 rpos, vec3 rdir, vec3 sp, float radius, inout vec3 point, inout vec3 normal) {
	radius = radius * radius;
	vec3 tmp = rpos - sp;
	float dt = dot(rdir, -tmp);
	if (dt < 0.0)
		return INFINITY;
	tmp.x = dot(tmp, tmp);
	tmp.x = tmp.x - dt * dt;
	if (tmp.x >= radius)
		return INFINITY;
	dt = dt - sqrt(radius - tmp.x);
	point = rpos + rdir * dt;
	normal = normalize(point - sp);
	return dt;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
	// simple sphere
	float wh = min(iResolution.x, iResolution.y);
	vec3 rayDir = normalize(vec3((fragCoord * 2.0 - iResolution.xy) / wh, -0.85));
	rayDir = normalize(rayDir);
	vec3 rayOrigin = vec3(0, 0, 2.25);
	// rotate camera with time
	float angle = iTime * 0.25;
	float ca = cos(angle);
	float sa = sin(angle);
	mat2 camRot = mat2(ca, sa, -sa, ca);
	rayOrigin.xz = camRot * rayOrigin.xz;
	rayDir.xz = camRot * rayDir.xz;
	const vec3 sphereO = vec3(0.0, 0.0, 0.0);
	const float sphereR = 1.3;
	vec3 point;
	vec3 normal;
	float distToCamera = raySphere(rayOrigin, rayDir, sphereO, sphereR, point, normal);
	vec3 linColor;
	if (distToCamera < INFINITY) {
		float sweepFactor = 1.0 - abs(dot(normal.xz, camRot[0]));
		// Ideally this would be blue noise, but regular hash random also works.
		float fragmentNoise = hashRand(fragCoord + vec2(0.01, 0.023) * float(iTime));
		linColor = shade(point, normal, rayOrigin - point, distToCamera, sweepFactor, fragmentNoise);
	} else {
		linColor = engTextureEnvMapLOD(rayDir, 0.0);
	}
	fragColor = vec4(srgb_from_rgb(linColor), 1.0);
}

