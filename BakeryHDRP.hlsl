#ifndef BAKERY_INCLUDED
#define BAKERY_INCLUDED

float bakeryLightmapMode;
//float2 bakeryLightmapSize;
#define BAKERYMODE_DEFAULT 0
#define BAKERYMODE_VERTEXLM 1.0f
#define BAKERYMODE_RNM 2.0f
#define BAKERYMODE_SH 3.0f

#if defined(BAKERY_RNM) && defined(BAKERY_LMSPEC)
#define BAKERY_RNMSPEC
#endif

#ifndef BAKERY_VERTEXLM
    #undef BAKERY_VERTEXLMDIR
    #undef BAKERY_VERTEXLMSH
    #undef BAKERY_VERTEXLMMASK
#endif

#define lumaConv float3(0.2125f, 0.7154f, 0.0721f)

#if defined(BAKERY_SH) || defined(BAKERY_VERTEXLMSH)
float shEvaluateDiffuseL1Geomerics(float L0, float3 L1, float3 n)
{
    // average energy
    float R0 = L0;

    // avg direction of incoming light
    float3 R1 = 0.5f * L1;

    // directional brightness
    float lenR1 = length(R1);

    // linear angle between normal and direction 0-1
    //float q = 0.5f * (1.0f + dot(R1 / lenR1, n));
    //float q = dot(R1 / lenR1, n) * 0.5 + 0.5;
    float q = dot(normalize(R1), n) * 0.5 + 0.5;

    // power for q
    // lerps from 1 (linear) to 3 (cubic) based on directionality
    float p = 1.0f + 2.0f * lenR1 / R0;

    // dynamic range constant
    // should vary between 4 (highly directional) and 0 (ambient)
    float a = (1.0f - lenR1 / R0) / (1.0f + lenR1 / R0);

    return R0 * (a + (1.0f - a) * (p + 1.0f) * pow(q, p));
}
#endif

#ifdef BAKERY_VERTEXLM
    float4 unpack4NFloats(float src) {
        //return fmod(float4(src / 262144.0, src / 4096.0, src / 64.0, src), 64.0)/64.0;
        return frac(float4(src / (262144.0*64), src / (4096.0*64), src / (64.0*64), src));
    }
    float3 unpack3NFloats(float src) {
        float r = frac(src);
        float g = frac(src * 256.0);
        float b = frac(src * 65536.0);
        return float3(r, g, b);
    }
#if defined(BAKERY_VERTEXLMDIR)
    void BakeryVertexLMDirection(inout float3 diffuseColor, inout float3 specularColor, float3 lightDirection, float3 vertexNormalWorld, float3 normalWorld, float3 viewDir, float smoothness)
    {
        float3 dominantDir = Unity_SafeNormalize(lightDirection);
        half halfLambert = dot(normalWorld, dominantDir) * 0.5 + 0.5;
        half flatNormalHalfLambert = dot(vertexNormalWorld, dominantDir) * 0.5 + 0.5;

        #ifdef BAKERY_LMSPEC
            half3 halfDir = Unity_SafeNormalize(normalize(dominantDir) - viewDir);
            half nh = saturate(dot(normalWorld, halfDir));
            half perceptualRoughness = SmoothnessToPerceptualRoughness(smoothness);
            half roughness = PerceptualRoughnessToRoughness(perceptualRoughness);
            half spec = GGXTerm(nh, roughness);
            diffuseColor = spec * diffuseColor;
        #endif

        diffuseColor *= halfLambert / max(1e-4h, flatNormalHalfLambert);
    }
#elif defined(BAKERY_VERTEXLMSH)
    void BakeryVertexLMSH(inout float3 diffuseColor, inout float3 specularColor, float3 shL1x, float3 shL1y, float3 shL1z, float3 normalWorld, float3 viewDir, float smoothness)
    {
        float3 L0 = diffuseColor;
        float3 nL1x = shL1x;
        float3 nL1y = shL1y;
        float3 nL1z = shL1z;
        float3 L1x = nL1x * L0 * 2;
        float3 L1y = nL1y * L0 * 2;
        float3 L1z = nL1z * L0 * 2;

        float3 sh;
        sh.r = shEvaluateDiffuseL1Geomerics(L0.r, float3(L1x.r, L1y.r, L1z.r), normalWorld);
        sh.g = shEvaluateDiffuseL1Geomerics(L0.g, float3(L1x.g, L1y.g, L1z.g), normalWorld);
        sh.b = shEvaluateDiffuseL1Geomerics(L0.b, float3(L1x.b, L1y.b, L1z.b), normalWorld);

        diffuseColor = sh;

        #ifdef BAKERY_LMSPEC
            float3 dominantDir = float3(dot(nL1x, lumaConv), dot(nL1y, lumaConv), dot(nL1z, lumaConv));
            float focus = saturate(length(dominantDir));
            half3 halfDir = Unity_SafeNormalize(normalize(dominantDir) - viewDir);
            half nh = saturate(dot(normalWorld, halfDir));
            half perceptualRoughness = SmoothnessToPerceptualRoughness(smoothness );//* sqrt(focus));
            half roughness = PerceptualRoughnessToRoughness(perceptualRoughness);
            half spec = GGXTerm(nh, roughness);
            specularColor = spec * sh;
        #endif
    }
#endif
#endif

#ifdef BAKERY_BICUBIC
float BakeryBicubic_w0(float a)
{
    return (1.0f/6.0f)*(a*(a*(-a + 3.0f) - 3.0f) + 1.0f);
}

float BakeryBicubic_w1(float a)
{
    return (1.0f/6.0f)*(a*a*(3.0f*a - 6.0f) + 4.0f);
}

float BakeryBicubic_w2(float a)
{
    return (1.0f/6.0f)*(a*(a*(-3.0f*a + 3.0f) + 3.0f) + 1.0f);
}

float BakeryBicubic_w3(float a)
{
    return (1.0f/6.0f)*(a*a*a);
}

float BakeryBicubic_g0(float a)
{
    return BakeryBicubic_w0(a) + BakeryBicubic_w1(a);
}

float BakeryBicubic_g1(float a)
{
    return BakeryBicubic_w2(a) + BakeryBicubic_w3(a);
}

float BakeryBicubic_h0(float a)
{
    return -1.0f + BakeryBicubic_w1(a) / (BakeryBicubic_w0(a) + BakeryBicubic_w1(a)) + 0.5f;
}

float BakeryBicubic_h1(float a)
{
    return 1.0f + BakeryBicubic_w3(a) / (BakeryBicubic_w2(a) + BakeryBicubic_w3(a)) + 0.5f;
}
#endif

#if defined(BAKERY_RNM) || defined(BAKERY_SH)
sampler2D _RNM0, _RNM1, _RNM2;
float4 _RNM0_TexelSize;
#endif

// Decodes HDR textures
// handles dLDR, RGBM formats
inline half3 DecodeHDR (half4 data, half4 decodeInstructions)
{
    // Take into account texture alpha if decodeInstructions.w is true(the alpha value affects the RGB channels)
    half alpha = decodeInstructions.w * (data.a - 1.0) + 1.0;

    // If Linear mode is not supported we can skip exponent part
    #if defined(UNITY_COLORSPACE_GAMMA)
        return (decodeInstructions.x * alpha) * data.rgb;
    #else
    #   if defined(UNITY_USE_NATIVE_HDR)
            return decodeInstructions.x * data.rgb; // Multiplier for future HDRI relative to absolute conversion.
    #   else
            return (decodeInstructions.x * pow(alpha, decodeInstructions.y)) * data.rgb;
    #   endif
    #endif
}

// Decodes HDR textures
// handles dLDR, RGBM formats
// Called by DecodeLightmap when UNITY_NO_RGBM is not defined.
inline half3 DecodeLightmapRGBM (half4 data, half4 decodeInstructions)
{
    // If Linear mode is not supported we can skip exponent part
    #if defined(UNITY_COLORSPACE_GAMMA)
    # if defined(UNITY_FORCE_LINEAR_READ_FOR_RGBM)
        return (decodeInstructions.x * data.a) * sqrt(data.rgb);
    # else
        return (decodeInstructions.x * data.a) * data.rgb;
    # endif
    #else
        return (decodeInstructions.x * pow(data.a, decodeInstructions.y)) * data.rgb;
    #endif
}

// Decodes doubleLDR encoded lightmaps.
inline half3 DecodeLightmapDoubleLDR( float4 color, half4 decodeInstructions)
{
    // decodeInstructions.x contains 2.0 when gamma color space is used or pow(2.0, 2.2) = 4.59 when linear color space is used on mobile platforms
    return decodeInstructions.x * color.rgb;
}

inline half3 DecodeLightmap( float4 color)
{
#ifdef UNITY_LIGHTMAP_FULL_HDR
    bool useRGBMLightmap = false;
    float4 decodeInstructions = float4(0.0, 0.0, 0.0, 0.0); // Never used but needed for the interface since it supports gamma lightmaps
#else
    bool useRGBMLightmap = true;
    #if defined(UNITY_LIGHTMAP_RGBM_ENCODING)
        float4 decodeInstructions = float4(34.493242, 2.2, 0.0, 0.0); // range^2.2 = 5^2.2, gamma = 2.2
    #else
        float4 decodeInstructions = float4(2.0, 2.2, 0.0, 0.0); // range = 2.0^2.2 = 4.59
    #endif
#endif

#if defined(UNITY_LIGHTMAP_DLDR_ENCODING)
    return DecodeLightmapDoubleLDR(color, decodeInstructions);
#elif defined(UNITY_LIGHTMAP_RGBM_ENCODING)
    return DecodeLightmapRGBM(color, decodeInstructions);
#else //defined(UNITY_LIGHTMAP_FULL_HDR)
    return color.rgb;
#endif
}

#ifdef BAKERY_BICUBIC
    // Bicubic
    float4 BakeryTex2D(sampler2D tex, float2 uv, float4 texelSize)
    {
        float x = uv.x * texelSize.z;
        float y = uv.y * texelSize.z;

        x -= 0.5f;
        y -= 0.5f;

        float px = floor(x);
        float py = floor(y);

        float fx = x - px;
        float fy = y - py;

        float g0x = BakeryBicubic_g0(fx);
        float g1x = BakeryBicubic_g1(fx);
        float h0x = BakeryBicubic_h0(fx);
        float h1x = BakeryBicubic_h1(fx);
        float h0y = BakeryBicubic_h0(fy);
        float h1y = BakeryBicubic_h1(fy);

        return     BakeryBicubic_g0(fy) * ( g0x * tex2D(tex, (float2(px + h0x, py + h0y) * texelSize.x))   +
                              g1x * tex2D(tex, (float2(px + h1x, py + h0y) * texelSize.x))) +

                   BakeryBicubic_g1(fy) * ( g0x * tex2D(tex, (float2(px + h0x, py + h1y) * texelSize.x))   +
                              g1x * tex2D(tex, (float2(px + h1x, py + h1y) * texelSize.x)));
    }
    float4 BakeryTex2D(Texture2D tex, SamplerState s, float2 uv, float4 texelSize)
    {
        float x = uv.x * texelSize.z;
        float y = uv.y * texelSize.z;

        x -= 0.5f;
        y -= 0.5f;

        float px = floor(x);
        float py = floor(y);

        float fx = x - px;
        float fy = y - py;

        float g0x = BakeryBicubic_g0(fx);
        float g1x = BakeryBicubic_g1(fx);
        float h0x = BakeryBicubic_h0(fx);
        float h1x = BakeryBicubic_h1(fx);
        float h0y = BakeryBicubic_h0(fy);
        float h1y = BakeryBicubic_h1(fy);

        return     BakeryBicubic_g0(fy) * ( g0x * tex.Sample(s, (float2(px + h0x, py + h0y) * texelSize.x))   +
                              g1x * tex.Sample(s, (float2(px + h1x, py + h0y) * texelSize.x))) +

                   BakeryBicubic_g1(fy) * ( g0x * tex.Sample(s, (float2(px + h0x, py + h1y) * texelSize.x))   +
                              g1x * tex.Sample(s, (float2(px + h1x, py + h1y) * texelSize.x)));
    }
#else
    // Bilinear
    float4 BakeryTex2D(sampler2D tex, float2 uv, float4 texelSize)
    {
        return tex2D(tex, uv);
    }
    float4 BakeryTex2D(Texture2D tex, SamplerState s, float2 uv, float4 texelSize)
    {
        return tex.Sample(s, uv);
    }
#endif

#ifdef DIRLIGHTMAP_COMBINED
#ifdef BAKERY_LMSPEC
float BakeryDirectionalLightmapSpecular(float2 lmUV, float3 normalWorld, float3 viewDir, float smoothness)
{
    float3 dominantDir = UNITY_SAMPLE_TEX2D_SAMPLER(unity_LightmapInd, unity_Lightmap, lmUV).xyz * 2 - 1;
    half3 halfDir = Unity_SafeNormalize(normalize(dominantDir) - viewDir);
    half nh = saturate(dot(normalWorld, halfDir));
    half perceptualRoughness = SmoothnessToPerceptualRoughness(smoothness);
    half roughness = PerceptualRoughnessToRoughness(perceptualRoughness);
    half spec = GGXTerm(nh, roughness);
    return spec;
}
#endif
#endif

#ifdef BAKERY_RNM
void BakeryRNM(inout float3 diffuseColor, inout float3 specularColor, float2 lmUV, float3 normalMap, float smoothness, float3 viewDirT)
{
    const float3 rnmBasis0 = float3(0.816496580927726f, 0, 0.5773502691896258f);
    const float3 rnmBasis1 = float3(-0.4082482904638631f, 0.7071067811865475f, 0.5773502691896258f);
    const float3 rnmBasis2 = float3(-0.4082482904638631f, -0.7071067811865475f, 0.5773502691896258f);

    float3 rnm0 = c(BakeryTex2D(_RNM0, lmUV, _RNM0_TexelSize));
    float3 rnm1 = DecodeLightmap(BakeryTex2D(_RNM1, lmUV, _RNM0_TexelSize));
    float3 rnm2 = DecodeLightmap(BakeryTex2D(_RNM2, lmUV, _RNM0_TexelSize));

    #ifdef BAKERY_SSBUMP
        diffuseColor = normalMap.x * rnm0
                     + normalMap.z * rnm1
                     + normalMap.y * rnm2;
    #else
        diffuseColor = saturate(dot(rnmBasis0, normalMap)) * rnm0
                     + saturate(dot(rnmBasis1, normalMap)) * rnm1
                     + saturate(dot(rnmBasis2, normalMap)) * rnm2;
    #endif

    #ifdef BAKERY_LMSPEC
        float3 dominantDirT = rnmBasis0 * dot(rnm0, lumaConv) +
                              rnmBasis1 * dot(rnm1, lumaConv) +
                              rnmBasis2 * dot(rnm2, lumaConv);

        float3 dominantDirTN = NormalizePerPixelNormal(dominantDirT);
        float3 specColor = saturate(dot(rnmBasis0, dominantDirTN)) * rnm0 +
                           saturate(dot(rnmBasis1, dominantDirTN)) * rnm1 +
                           saturate(dot(rnmBasis2, dominantDirTN)) * rnm2;

        half3 halfDir = Unity_SafeNormalize(dominantDirTN - viewDirT);
        half nh = saturate(dot(normalMap, halfDir));
        half perceptualRoughness = SmoothnessToPerceptualRoughness(smoothness);
        half roughness = PerceptualRoughnessToRoughness(perceptualRoughness);
        half spec = GGXTerm(nh, roughness);
        specularColor = spec * specColor;
    #endif
}
#endif

#ifdef BAKERY_SH
void BakerySH(inout float3 diffuseColor, inout float3 specularColor, float2 lmUV, float3 normalWorld, float3 viewDir, float smoothness)
{
    float3 L0 = DecodeLightmap(BakeryTex2D(unity_Lightmap, samplerunity_Lightmap, lmUV, _RNM0_TexelSize));
    float3 nL1x = BakeryTex2D(_RNM0, lmUV, _RNM0_TexelSize) * 2 - 1;
    float3 nL1y = BakeryTex2D(_RNM1, lmUV, _RNM0_TexelSize) * 2 - 1;
    float3 nL1z = BakeryTex2D(_RNM2, lmUV, _RNM0_TexelSize) * 2 - 1;
    float3 L1x = nL1x * L0 * 2;
    float3 L1y = nL1y * L0 * 2;
    float3 L1z = nL1z * L0 * 2;

    float3 sh;
    //sh.r = shEvaluateDiffuseL1Geomerics(L0.r, float3(L1x.r, L1y.r, L1z.r), normalWorld);
    //sh.g = shEvaluateDiffuseL1Geomerics(L0.g, float3(L1x.g, L1y.g, L1z.g), normalWorld);
    //sh.b = shEvaluateDiffuseL1Geomerics(L0.b, float3(L1x.b, L1y.b, L1z.b), normalWorld);

    float lumaL0 = dot(L0, lumaConv);
    float lumaL1x = dot(L1x, lumaConv);
    float lumaL1y = dot(L1y, lumaConv);
    float lumaL1z = dot(L1z, lumaConv);
    float lumaSH = shEvaluateDiffuseL1Geomerics(lumaL0, float3(lumaL1x, lumaL1y, lumaL1z), normalWorld);

    sh = L0 + normalWorld.x * L1x + normalWorld.y * L1y + normalWorld.z * L1z;
    float regularLumaSH = dot(sh, lumaConv);
    //sh *= regularLumaSH < 0.001 ? 1 : (lumaSH / regularLumaSH);
    sh *= lerp(1, lumaSH / regularLumaSH, saturate(regularLumaSH*16));

    diffuseColor = max(sh,0.0);

    #ifdef BAKERY_LMSPEC
        float3 dominantDir = float3(dot(nL1x, lumaConv), dot(nL1y, lumaConv), dot(nL1z, lumaConv));
        float focus = saturate(length(dominantDir));
        half3 halfDir = Unity_SafeNormalize(normalize(dominantDir) - viewDir);
        half nh = saturate(dot(normalWorld, halfDir));
        half perceptualRoughness = SmoothnessToPerceptualRoughness(smoothness );//* sqrt(focus));
        half roughness = PerceptualRoughnessToRoughness(perceptualRoughness);
        half spec = GGXTerm(nh, roughness);

        //sh.r = shEvaluateDiffuseL1Geomerics(L0.r, float3(L1x.r, L1y.r, L1z.r), dominantDir);
        //sh.g = shEvaluateDiffuseL1Geomerics(L0.g, float3(L1x.g, L1y.g, L1z.g), dominantDir);
        //sh.b = shEvaluateDiffuseL1Geomerics(L0.b, float3(L1x.b, L1y.b, L1z.b), dominantDir);

        specularColor = spec * sh;
    #endif
}
#endif

#endif
