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
};

struct VRayMtlContext {
	vec3 geomNormal;
	float gloss1;
	float gloss2;
	float reflGloss;
	vec3 e;
	vec3 diff;
	float fresnel;
	vec3 reflNoFresnel;
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
};

vec3 sampleBRDF(VRayMtlInitParams params, VRayMtlContext ctx,
		int sampleIdx, int nbSamples, out float brdfContrib);
vec3 sampleRefractBRDF(VRayMtlInitParams params, VRayMtlContext ctx,
		int sampleIdx, int nbSamples, out bool totalInternalReflection);

VRayMtlContext initVRayMtlContext(VRayMtlInitParams initParams);

vec3 computeDirectDiffuseContribution(VRayMtlInitParams params, VRayMtlContext ctx, vec3 lightDir);
vec3 computeDirectReflectionContribution(VRayMtlInitParams params, VRayMtlContext ctx, vec3 lightDir);

vec3 computeIndirectDiffuseContribution(VRayMtlInitParams params, VRayMtlContext ctx);
vec3 computeIndirectReflectionContribution(VRayMtlInitParams params, VRayMtlContext ctx);
vec3 computeIndirectRefractionContribution(
	VRayMtlInitParams params, VRayMtlContext ctx, float alpha, vec3 alphaDir);

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

void computeTangentVectors(vec3 n, out vec3 u, out vec3 v) {
	// It doesn't matter what these vectors are, the result vectors just need to be perpendicular to the normal and to
	// each other
	u = cross(n, vec3(0.643782, 0.98432, 0.324632));
	if (length(u) < 1e-6)
		u = cross(n, vec3(0.432902, 0.43223, 0.908953));
	u = normalize(u);
	v = normalize(cross(n, u));
}

void makeNormalMatrix(in vec3 n, out mat3 m) {
	computeTangentVectors(n, m[0], m[1]);
	m[2] = n;
}

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
		dir = getWardDir(u, v, ctx.gloss2, -ctx.e, ctx.nm);
	} else /* brdfType==3 */ {
		dir = getGGXDir(u, v, ctx.gloss2, ctx.gtrGamma, -ctx.e, ctx.nm, rayProb, brdfContrib);
	}

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

float pow35(float x) {
	return x * x * x * sqrt(x);
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

	VRayMtlContext result;
	if (initParams.lockFresnelIOR)
		initParams.fresnelIOR = initParams.refractionIOR;

	result.e = -normalize(Vw);
	if (useRoughness)
		reflGloss = 1.0 - reflGloss; // Invert glossiness (turn it into roughness)

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
	result.fresnel = 1.0;
	if (useFresnel && !internalReflection)
		result.fresnel = clamp(getFresnelCoeff(fresnelIOR, result.e, normal, refractDir), 0.0, 1.0);

	result.reflNoFresnel = reflColor * reflAmount * result.opacity;
	result.refl = result.reflNoFresnel * result.fresnel;

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

	result.refl = mix(metalColor, result.reflNoFresnel, result.fresnel);

	result.gloss1 = max(0.0, 1.0 / pow35(max(1.0 - reflGloss, 1e-4)) - 1.0); // [0, 1] -> [0, inf)
	result.gloss2 = max(1.0 - reflGloss, 1e-4);
	result.gloss2 *= result.gloss2;
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

vec3 vrayMtlDiffuse(vec3 lightDir, vec3 normal) {
	return vec3(max(0.0, dot(lightDir, normal)));
}

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

vec3 vrayMtlWard(vec3 lightDir, VRayMtlContext ctx) {
	float cs1 = -dot(ctx.e, ctx.geomNormal);
	float lightNdotL = dot(ctx.geomNormal, lightDir);
	if (lightNdotL > 1e-6 && cs1 > 1e-6) {
		vec3 hw = lightDir - ctx.e;
		vec3 hn = normalize(ctx.inm * hw);
		if (hn.z > 1e-3) {
			float tanhSqr = (1.0 / (hn.z * hn.z) - 1.0);
			float divd = cs1 * ctx.gloss2;
			float k = exp(-tanhSqr / ctx.gloss2) / divd;
			k *= lightNdotL;
			if (k > 0.0)
				return vec3(k);
		}
	}
	return vec3(0.0);
}

vec3 vrayMtlGGX(vec3 lightDir, VRayMtlContext ctx) {
	float cs1 = -dot(ctx.e, ctx.geomNormal);
	float lightNdotL = dot(ctx.geomNormal, lightDir);
	if (lightNdotL > 1e-6 && cs1 > 1e-6) {
		vec3 hw = normalize(lightDir - ctx.e);
		vec3 hn = normalize(ctx.inm * hw);
		if (hn.z > 1e-3) {
			float D = getGGXMicrofacetDistribution(hn.z, ctx.gloss2, ctx.gtrGamma);
			float G =
				getGGXBidirectionalShadowingMasking(-ctx.e, lightDir, hw, ctx.geomNormal, ctx.gloss2, ctx.gtrGamma);
			float cs2 = max(dot(hw, lightDir), 0.0001);
			vec3 micron = ctx.nm * hn;
			float L2 = dot(micron, micron);
			float anisotropyCorrection = L2 * sqrt(L2);
			float k = 0.25 * D * G * anisotropyCorrection / cs1; // anisotropy correction
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

vec3 computeIndirectDiffuseContribution(VRayMtlInitParams params, VRayMtlContext ctx) {
	vec3 res = vec3(0.0);
	res = engEnvIrradiance(params.geomNormal);
	return res;
}

vec3 computeIndirectReflectionContribution(VRayMtlInitParams params, VRayMtlContext ctx) {
	vec3 res = vec3(0.0);

	if (!params.traceReflections)
		return res;

	int numSamples = NUM_ENV_SAMPLES + int(float(NUM_ENV_SAMPLES_ROUGH) * (params.aniso + 0.5 * (1.0 - ctx.reflGloss)));
	if (ctx.gloss2 < 0.0001)
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
};

#define PRESET_COUNT 22

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
/* fogMult		   */ 1.0),

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
/* fogMult		   */ 1.0),

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
/* fogMult		   */ 1.0),

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
/* fogMult		   */ 1.0),

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
/* fogMult		   */ 1.0),

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
/* fogMult		   */ 1.0),

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
/* fogMult		   */ 1.0),

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
/* fogMult		   */ 1.0),

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
/* fogMult		   */ 1.0),

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
/* fogMult		   */ 1.0),

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
/* fogMult		   */ 1.0),

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
/* fogMult		   */ 1.0),

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
/* fogMult		   */ 1.0),

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
/* fogMult		   */ 1.0),

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
/* fogMult		   */ 1.0),

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
/* fogMult		   */ 1.0),

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
/* fogMult		   */ 1.0),

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
/* fogMult		   */ 1.0),

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
/* fogMult		   */ 1.0),

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
/* fogMult		   */ 1.0),

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
/* fogMult		   */ 1.0),

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
/* fogMult		   */ 1.0)

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
	vec3 reflection =
		(computeDirectReflectionContribution(initParams, ctx, lightDir)
		 + computeIndirectReflectionContribution(initParams, ctx));
	float alpha = intensity(ctx.opacity);
	vec3 refraction = computeRefractFogContrib(initParams, ctx, diffuseDirect)
		+ computeIndirectRefractionContribution(initParams, ctx, alpha, -initParams.Vw);

	return +ctx.diff * (diffuseDirect + diffuseIndirect) + reflection * ctx.refl + ctx.illum + refraction * ctx.refr;
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

