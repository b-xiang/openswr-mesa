/****************************************************************************
* Copyright (C) 2014-2015 Intel Corporation.   All Rights Reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a
* copy of this software and associated documentation files (the "Software"),
* to deal in the Software without restriction, including without limitation
* the rights to use, copy, modify, merge, publish, distribute, sublicense,
* and/or sell copies of the Software, and to permit persons to whom the
* Software is furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice (including the next
* paragraph) shall be included in all copies or substantial portions of the
* Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
* THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
* IN THE SOFTWARE.
*
* @file multisample.h
*
******************************************************************************/

#pragma once

#include "context.h"
#include "format_traits.h"

INLINE
uint32_t GetNumSamples(SWR_MULTISAMPLE_COUNT sampleCount)
{
    static const uint32_t sampleCountLUT[SWR_MULTISAMPLE_TYPE_MAX] {1, 2, 4, 8, 16};
    assert(sampleCount < SWR_MULTISAMPLE_TYPE_MAX);
    return sampleCountLUT[sampleCount];
}

INLINE
SWR_MULTISAMPLE_COUNT GetSampleCount(uint32_t numSamples)
{
    switch(numSamples)
    {
    case 1: return SWR_MULTISAMPLE_1X;
    case 2: return SWR_MULTISAMPLE_2X;
    case 4: return SWR_MULTISAMPLE_4X;
    case 8: return SWR_MULTISAMPLE_8X;
    case 16: return SWR_MULTISAMPLE_16X;
    default: assert(0); return SWR_MULTISAMPLE_1X;
    }
}

// hardcoded offsets based on Direct3d standard multisample positions
// 8 x 8 pixel grid ranging from (0, 0) to (15, 15), with (0, 0) = UL pixel corner
// coords are 0.8 fixed point offsets from (0, 0)
template<SWR_MULTISAMPLE_COUNT sampleCount>
struct MultisampleTraits
{
    INLINE static __m128i vXi(uint32_t sampleNum) = delete;
    INLINE static __m128i vYi(uint32_t sampleNum) = delete;
    INLINE static simdscalar vX(uint32_t sampleNum) = delete;
    INLINE static simdscalar vY(uint32_t sampleNum) = delete;
    INLINE static __m128i TileSampleOffsetsX() = delete;
    INLINE static __m128i TileSampleOffsetsY() = delete;
    INLINE static uint32_t RasterTileColorOffset(uint32_t sampleNum) = delete;
    INLINE static uint32_t RasterTileDepthOffset(uint32_t sampleNum) = delete;
    INLINE static uint32_t RasterTileStencilOffset(uint32_t sampleNum) = delete;

    static const uint32_t numSamples = 0;
    static const uint32_t sampleMask = 0;
};

template<>
struct MultisampleTraits<SWR_MULTISAMPLE_1X>
{
    INLINE static __m128i vXi(uint32_t sampleNum)
    {
        static const __m128i X = _mm_set1_epi32(samplePosX);
        return X;
    }

    INLINE static __m128i vYi(uint32_t sampleNum)
    {
        static const __m128i Y = _mm_set1_epi32(samplePosY);
        return Y;
    }

    INLINE static simdscalar vX(uint32_t sampleNum)
    {
        static const simdscalar X = _simd_set1_ps(0.5f);
        return X;
    }

    INLINE static simdscalar vY(uint32_t sampleNum)
    {
        static const simdscalar Y = _simd_set1_ps(0.5f);
        return Y;
    }

    INLINE static __m128i TileSampleOffsetsX()
    {
        static const uint32_t bboxLeftEdge = 0x80;
        static const uint32_t bboxRightEdge = 0x80;
                                                            // BR,            BL,           UR,            UL
        static const __m128i tileSampleOffsetX = _mm_set_epi32(bboxRightEdge, bboxLeftEdge, bboxRightEdge, bboxLeftEdge);
        return tileSampleOffsetX;
    }

    INLINE static __m128i TileSampleOffsetsY()
    {
        static const uint32_t bboxTopEdge = 0x80;
        static const uint32_t bboxBottomEdge = 0x80;
                                                            // BR,             BL,             UR,          UL
        static const __m128i tileSampleOffsetY = _mm_set_epi32(bboxBottomEdge, bboxBottomEdge, bboxTopEdge, bboxTopEdge);
        return tileSampleOffsetY;
    }

    INLINE static uint32_t RasterTileColorOffset(uint32_t sampleNum)
    {
        return 0;
    }

    INLINE static uint32_t RasterTileDepthOffset(uint32_t sampleNum)
    {
        return 0;
    }

    INLINE static uint32_t RasterTileStencilOffset(uint32_t sampleNum)
    {
        return 0;
    }

    static const uint32_t samplePosX {0x80};
    static const uint32_t samplePosY {0x80};
    static const uint32_t numSamples = 1;
    static const uint32_t sampleMask = 1;
};

template<>
struct MultisampleTraits<SWR_MULTISAMPLE_2X>
{
    INLINE static __m128i vXi(uint32_t sampleNum)
    {
        SWR_ASSERT(sampleNum < numSamples);
        static const __m128i X[numSamples] {_mm_set1_epi32(samplePosX[0]), _mm_set1_epi32(samplePosX[1])};
        return X[sampleNum];
    }

    INLINE static __m128i vYi(uint32_t sampleNum)
    {
        SWR_ASSERT(sampleNum < numSamples);
        static const __m128i Y[numSamples] {_mm_set1_epi32(samplePosY[0]), _mm_set1_epi32(samplePosY[1])};
        return Y[sampleNum];
    }

    INLINE static simdscalar vX(uint32_t sampleNum)
    {
        static const simdscalar X[numSamples] {_simd_set1_ps(0.75f), _simd_set1_ps(0.25f)};
        assert(sampleNum < numSamples);
        return X[sampleNum];
    }

    INLINE static simdscalar vY(uint32_t sampleNum)
    {
        static const simdscalar Y[numSamples] {_simd_set1_ps(0.75f), _simd_set1_ps(0.25f)};
        assert(sampleNum < numSamples);
        return Y[sampleNum];
    }

    INLINE static __m128i TileSampleOffsetsX()
    {
        static const uint32_t bboxLeftEdge = 0x40;
        static const uint32_t bboxRightEdge = 0xC0;
                                                            // BR,            BL,           UR,            UL
        static const __m128i tileSampleOffsetX = _mm_set_epi32(bboxRightEdge, bboxLeftEdge, bboxRightEdge, bboxLeftEdge);
        return tileSampleOffsetX;
    }

    INLINE static __m128i TileSampleOffsetsY()
    {
        static const uint32_t bboxTopEdge = 0x40;
        static const uint32_t bboxBottomEdge = 0xC0;
                                                            // BR,             BL,             UR,          UL
        static const __m128i tileSampleOffsetY = _mm_set_epi32(bboxBottomEdge, bboxBottomEdge, bboxTopEdge, bboxTopEdge);
        return tileSampleOffsetY;
    }

    INLINE static uint32_t RasterTileColorOffset(uint32_t sampleNum)
    {
        static const uint32_t RasterTileColorOffsets[numSamples]
        { 0,
          (KNOB_TILE_X_DIM * KNOB_TILE_Y_DIM * FormatTraits<KNOB_COLOR_HOT_TILE_FORMAT>::bpp / 8)
        };
        assert(sampleNum < numSamples);
        return RasterTileColorOffsets[sampleNum];
    }

    INLINE static uint32_t RasterTileDepthOffset(uint32_t sampleNum)
    {
        static const uint32_t RasterTileDepthOffsets[numSamples]
        { 0,
          (KNOB_TILE_X_DIM * KNOB_TILE_Y_DIM * FormatTraits<KNOB_DEPTH_HOT_TILE_FORMAT>::bpp / 8)
        };
        assert(sampleNum < numSamples);
        return RasterTileDepthOffsets[sampleNum];
    }

    INLINE static uint32_t RasterTileStencilOffset(uint32_t sampleNum)
    {
        static const uint32_t RasterTileStencilOffsets[numSamples]
        { 0,
          (KNOB_TILE_X_DIM * KNOB_TILE_Y_DIM * FormatTraits<KNOB_STENCIL_HOT_TILE_FORMAT>::bpp / 8)
        };
        assert(sampleNum < numSamples);
        return RasterTileStencilOffsets[sampleNum];
    }

    static const uint32_t samplePosX[2];
    static const uint32_t samplePosY[2];
    static const uint32_t numSamples = 2;
    static const uint32_t sampleMask = 0x3;
};

template<>
struct MultisampleTraits<SWR_MULTISAMPLE_4X>
{
    INLINE static __m128i vXi(uint32_t sampleNum)
    {
        static const __m128i X[numSamples]
        {_mm_set1_epi32(samplePosX[0]), _mm_set1_epi32(samplePosX[1]), _mm_set1_epi32(samplePosX[2]), _mm_set1_epi32(samplePosX[3])};
        SWR_ASSERT(sampleNum < numSamples);
        return X[sampleNum];
    }

    INLINE static __m128i vYi(uint32_t sampleNum)
    {
        static const __m128i Y[numSamples]
        {_mm_set1_epi32(samplePosY[0]), _mm_set1_epi32(samplePosY[1]), _mm_set1_epi32(samplePosY[2]), _mm_set1_epi32(samplePosY[3])};
        SWR_ASSERT(sampleNum < numSamples);
        return Y[sampleNum];
    }

    INLINE static simdscalar vX(uint32_t sampleNum)
    {
        static const simdscalar X[numSamples] 
        {_simd_set1_ps(0.375f), _simd_set1_ps(0.875), _simd_set1_ps(0.125), _simd_set1_ps(0.625)};
        assert(sampleNum < numSamples);
        return X[sampleNum];
    }

    INLINE static simdscalar vY(uint32_t sampleNum)
    {
        static const simdscalar Y[numSamples]
        {_simd_set1_ps(0.125), _simd_set1_ps(0.375f), _simd_set1_ps(0.625), _simd_set1_ps(0.875)};
        assert(sampleNum < numSamples);
        return Y[sampleNum];
    }

    INLINE static __m128i TileSampleOffsetsX()
    {
        static const uint32_t bboxLeftEdge = 0x20;
        static const uint32_t bboxRightEdge = 0xE0;
                                                            // BR,            BL,           UR,            UL
        static const __m128i tileSampleOffsetX = _mm_set_epi32(bboxRightEdge, bboxLeftEdge, bboxRightEdge, bboxLeftEdge);
        return tileSampleOffsetX;
    }

    INLINE static __m128i TileSampleOffsetsY()
    {
        static const uint32_t bboxTopEdge = 0x20;
        static const uint32_t bboxBottomEdge = 0xE0;
                                                            // BR,             BL,             UR,          UL
        static const __m128i tileSampleOffsetY = _mm_set_epi32(bboxBottomEdge, bboxBottomEdge, bboxTopEdge, bboxTopEdge);
        return tileSampleOffsetY;
    }

    INLINE static uint32_t RasterTileColorOffset(uint32_t sampleNum)
    {
        static const uint32_t RasterTileColorOffsets[numSamples]
        { 0,
          (KNOB_TILE_X_DIM * KNOB_TILE_Y_DIM * FormatTraits<KNOB_COLOR_HOT_TILE_FORMAT>::bpp / 8),
          (KNOB_TILE_X_DIM * KNOB_TILE_Y_DIM * FormatTraits<KNOB_COLOR_HOT_TILE_FORMAT>::bpp / 8) * 2,
          (KNOB_TILE_X_DIM * KNOB_TILE_Y_DIM * FormatTraits<KNOB_COLOR_HOT_TILE_FORMAT>::bpp / 8) * 3,
        };
        assert(sampleNum < numSamples);
        return RasterTileColorOffsets[sampleNum];
    }

    INLINE static uint32_t RasterTileDepthOffset(uint32_t sampleNum)
    {
        static const uint32_t RasterTileDepthOffsets[numSamples]
        { 0,
          (KNOB_TILE_X_DIM * KNOB_TILE_Y_DIM * FormatTraits<KNOB_DEPTH_HOT_TILE_FORMAT>::bpp / 8),
          (KNOB_TILE_X_DIM * KNOB_TILE_Y_DIM * FormatTraits<KNOB_DEPTH_HOT_TILE_FORMAT>::bpp / 8) * 2,
          (KNOB_TILE_X_DIM * KNOB_TILE_Y_DIM * FormatTraits<KNOB_DEPTH_HOT_TILE_FORMAT>::bpp / 8) * 3,
        };
        assert(sampleNum < numSamples);
        return RasterTileDepthOffsets[sampleNum];
    }

    INLINE static uint32_t RasterTileStencilOffset(uint32_t sampleNum)
    {
        static const uint32_t RasterTileStencilOffsets[numSamples]
        { 0,
          (KNOB_TILE_X_DIM * KNOB_TILE_Y_DIM * FormatTraits<KNOB_STENCIL_HOT_TILE_FORMAT>::bpp / 8),
          (KNOB_TILE_X_DIM * KNOB_TILE_Y_DIM * FormatTraits<KNOB_STENCIL_HOT_TILE_FORMAT>::bpp / 8) * 2,
          (KNOB_TILE_X_DIM * KNOB_TILE_Y_DIM * FormatTraits<KNOB_STENCIL_HOT_TILE_FORMAT>::bpp / 8) * 3,
        };
        assert(sampleNum < numSamples);
        return RasterTileStencilOffsets[sampleNum];
    }

    static const uint32_t samplePosX[4];
    static const uint32_t samplePosY[4];
    static const uint32_t numSamples = 4;
    static const uint32_t sampleMask = 0xF;
};

template<>
struct MultisampleTraits<SWR_MULTISAMPLE_8X>
{
    INLINE static __m128i vXi(uint32_t sampleNum)
    {
        static const __m128i X[numSamples]
        {_mm_set1_epi32(samplePosX[0]), _mm_set1_epi32(samplePosX[1]), _mm_set1_epi32(samplePosX[2]), _mm_set1_epi32(samplePosX[3]), 
         _mm_set1_epi32(samplePosX[4]), _mm_set1_epi32(samplePosX[5]), _mm_set1_epi32(samplePosX[6]), _mm_set1_epi32(samplePosX[7])};
        SWR_ASSERT(sampleNum < numSamples);
        return X[sampleNum];
    }

    INLINE static __m128i vYi(uint32_t sampleNum)
    {
        static const __m128i Y[numSamples]
        {_mm_set1_epi32(samplePosY[0]), _mm_set1_epi32(samplePosY[1]), _mm_set1_epi32(samplePosY[2]), _mm_set1_epi32(samplePosY[3]), 
         _mm_set1_epi32(samplePosY[4]), _mm_set1_epi32(samplePosY[5]), _mm_set1_epi32(samplePosY[6]), _mm_set1_epi32(samplePosY[7])};
        SWR_ASSERT(sampleNum < numSamples);
        return Y[sampleNum];
    }

    INLINE static simdscalar vX(uint32_t sampleNum)
    {
        static const simdscalar X[numSamples]
        {_simd_set1_ps(0.5625), _simd_set1_ps(0.4375), _simd_set1_ps(0.8125), _simd_set1_ps(0.3125),
         _simd_set1_ps(0.1875), _simd_set1_ps(0.0625), _simd_set1_ps(0.6875), _simd_set1_ps(0.9375)};
        assert(sampleNum < numSamples);
        return X[sampleNum];
    }

    INLINE static simdscalar vY(uint32_t sampleNum)
    {
        static const simdscalar Y[numSamples]
        {_simd_set1_ps(0.3125), _simd_set1_ps(0.6875), _simd_set1_ps(0.5625), _simd_set1_ps(0.1875),
         _simd_set1_ps(0.8125), _simd_set1_ps(0.4375), _simd_set1_ps(0.9375), _simd_set1_ps(0.0625)};
        assert(sampleNum < numSamples);
        return Y[sampleNum];
    }

    INLINE static __m128i TileSampleOffsetsX()
    {
        static const uint32_t bboxLeftEdge = 0x10;
        static const uint32_t bboxRightEdge = 0xF0;
                                                            // BR,            BL,           UR,            UL
        static const __m128i tileSampleOffsetX = _mm_set_epi32(bboxRightEdge, bboxLeftEdge, bboxRightEdge, bboxLeftEdge);
        return tileSampleOffsetX;
    }

    INLINE static __m128i TileSampleOffsetsY()
    {
        static const uint32_t bboxTopEdge = 0x10;
        static const uint32_t bboxBottomEdge = 0xF0;
                                                            // BR,             BL,             UR,          UL
        static const __m128i tileSampleOffsetY = _mm_set_epi32(bboxBottomEdge, bboxBottomEdge, bboxTopEdge, bboxTopEdge);
        return tileSampleOffsetY;
    }

    INLINE static uint32_t RasterTileColorOffset(uint32_t sampleNum)
    {
        static const uint32_t RasterTileColorOffsets[numSamples]
        { 0,
          (KNOB_TILE_X_DIM * KNOB_TILE_Y_DIM * FormatTraits<KNOB_COLOR_HOT_TILE_FORMAT>::bpp / 8),
          (KNOB_TILE_X_DIM * KNOB_TILE_Y_DIM * FormatTraits<KNOB_COLOR_HOT_TILE_FORMAT>::bpp / 8) * 2,
          (KNOB_TILE_X_DIM * KNOB_TILE_Y_DIM * FormatTraits<KNOB_COLOR_HOT_TILE_FORMAT>::bpp / 8) * 3,
          (KNOB_TILE_X_DIM * KNOB_TILE_Y_DIM * FormatTraits<KNOB_COLOR_HOT_TILE_FORMAT>::bpp / 8) * 4,
          (KNOB_TILE_X_DIM * KNOB_TILE_Y_DIM * FormatTraits<KNOB_COLOR_HOT_TILE_FORMAT>::bpp / 8) * 5,
          (KNOB_TILE_X_DIM * KNOB_TILE_Y_DIM * FormatTraits<KNOB_COLOR_HOT_TILE_FORMAT>::bpp / 8) * 6,
          (KNOB_TILE_X_DIM * KNOB_TILE_Y_DIM * FormatTraits<KNOB_COLOR_HOT_TILE_FORMAT>::bpp / 8) * 7,
        };
        assert(sampleNum < numSamples);
        return RasterTileColorOffsets[sampleNum];
    }

    INLINE static uint32_t RasterTileDepthOffset(uint32_t sampleNum)
    {
        static const uint32_t RasterTileDepthOffsets[numSamples]
        { 0,
          (KNOB_TILE_X_DIM * KNOB_TILE_Y_DIM * FormatTraits<KNOB_DEPTH_HOT_TILE_FORMAT>::bpp / 8),
          (KNOB_TILE_X_DIM * KNOB_TILE_Y_DIM * FormatTraits<KNOB_DEPTH_HOT_TILE_FORMAT>::bpp / 8) * 2,
          (KNOB_TILE_X_DIM * KNOB_TILE_Y_DIM * FormatTraits<KNOB_DEPTH_HOT_TILE_FORMAT>::bpp / 8) * 3,
          (KNOB_TILE_X_DIM * KNOB_TILE_Y_DIM * FormatTraits<KNOB_DEPTH_HOT_TILE_FORMAT>::bpp / 8) * 4,
          (KNOB_TILE_X_DIM * KNOB_TILE_Y_DIM * FormatTraits<KNOB_DEPTH_HOT_TILE_FORMAT>::bpp / 8) * 5,
          (KNOB_TILE_X_DIM * KNOB_TILE_Y_DIM * FormatTraits<KNOB_DEPTH_HOT_TILE_FORMAT>::bpp / 8) * 6,
          (KNOB_TILE_X_DIM * KNOB_TILE_Y_DIM * FormatTraits<KNOB_DEPTH_HOT_TILE_FORMAT>::bpp / 8) * 7,
        };
        assert(sampleNum < numSamples);
        return RasterTileDepthOffsets[sampleNum];
    }

    INLINE static uint32_t RasterTileStencilOffset(uint32_t sampleNum)
    {
        static const uint32_t RasterTileStencilOffsets[numSamples]
        { 0,
          (KNOB_TILE_X_DIM * KNOB_TILE_Y_DIM * FormatTraits<KNOB_STENCIL_HOT_TILE_FORMAT>::bpp / 8),
          (KNOB_TILE_X_DIM * KNOB_TILE_Y_DIM * FormatTraits<KNOB_STENCIL_HOT_TILE_FORMAT>::bpp / 8) * 2,
          (KNOB_TILE_X_DIM * KNOB_TILE_Y_DIM * FormatTraits<KNOB_STENCIL_HOT_TILE_FORMAT>::bpp / 8) * 3,
          (KNOB_TILE_X_DIM * KNOB_TILE_Y_DIM * FormatTraits<KNOB_STENCIL_HOT_TILE_FORMAT>::bpp / 8) * 4,
          (KNOB_TILE_X_DIM * KNOB_TILE_Y_DIM * FormatTraits<KNOB_STENCIL_HOT_TILE_FORMAT>::bpp / 8) * 5,
          (KNOB_TILE_X_DIM * KNOB_TILE_Y_DIM * FormatTraits<KNOB_STENCIL_HOT_TILE_FORMAT>::bpp / 8) * 6,
          (KNOB_TILE_X_DIM * KNOB_TILE_Y_DIM * FormatTraits<KNOB_STENCIL_HOT_TILE_FORMAT>::bpp / 8) * 7,
        };
        assert(sampleNum < numSamples);
        return RasterTileStencilOffsets[sampleNum];
    }

    static const uint32_t samplePosX[8];
    static const uint32_t samplePosY[8];
    static const uint32_t numSamples = 8;
    static const uint32_t sampleMask = 0xFF;
};

template<>
struct MultisampleTraits<SWR_MULTISAMPLE_16X>
{
    INLINE static __m128i vXi(uint32_t sampleNum)
    {
        static const __m128i X[numSamples]
        {_mm_set1_epi32(samplePosX[0]), _mm_set1_epi32(samplePosX[1]), _mm_set1_epi32(samplePosX[2]), _mm_set1_epi32(samplePosX[3]), 
         _mm_set1_epi32(samplePosX[4]), _mm_set1_epi32(samplePosX[5]), _mm_set1_epi32(samplePosX[6]), _mm_set1_epi32(samplePosX[7]), 
         _mm_set1_epi32(samplePosX[8]), _mm_set1_epi32(samplePosX[9]), _mm_set1_epi32(samplePosX[10]), _mm_set1_epi32(samplePosX[11]), 
         _mm_set1_epi32(samplePosX[12]), _mm_set1_epi32(samplePosX[13]), _mm_set1_epi32(samplePosX[14]), _mm_set1_epi32(samplePosX[15])};
        SWR_ASSERT(sampleNum < numSamples);
        return X[sampleNum];
    }

    INLINE static __m128i vYi(uint32_t sampleNum)
    {
        static const __m128i Y[numSamples]
        {_mm_set1_epi32(samplePosY[0]), _mm_set1_epi32(samplePosY[1]), _mm_set1_epi32(samplePosY[2]), _mm_set1_epi32(samplePosY[3]), 
         _mm_set1_epi32(samplePosY[4]), _mm_set1_epi32(samplePosY[5]), _mm_set1_epi32(samplePosY[6]), _mm_set1_epi32(samplePosY[7]), 
         _mm_set1_epi32(samplePosY[8]), _mm_set1_epi32(samplePosY[9]), _mm_set1_epi32(samplePosY[10]), _mm_set1_epi32(samplePosY[11]), 
         _mm_set1_epi32(samplePosY[12]), _mm_set1_epi32(samplePosY[13]), _mm_set1_epi32(samplePosY[14]), _mm_set1_epi32(samplePosY[15])};
        SWR_ASSERT(sampleNum < numSamples);
        return Y[sampleNum];
    }

    INLINE static simdscalar vX(uint32_t sampleNum)
    {
        static const simdscalar X[numSamples]
        {_simd_set1_ps(0.5625), _simd_set1_ps(0.4375), _simd_set1_ps(0.3125), _simd_set1_ps(0.7500),
         _simd_set1_ps(0.1875), _simd_set1_ps(0.6250), _simd_set1_ps(0.8125), _simd_set1_ps(0.6875),
         _simd_set1_ps(0.3750), _simd_set1_ps(0.5000), _simd_set1_ps(0.2500), _simd_set1_ps(0.1250),
         _simd_set1_ps(0.0000), _simd_set1_ps(0.9375), _simd_set1_ps(0.8750), _simd_set1_ps(0.0625)};
        assert(sampleNum < numSamples);
        return X[sampleNum];
    }

    INLINE static simdscalar vY(uint32_t sampleNum)
    {
        static const simdscalar Y[numSamples]
        {_simd_set1_ps(0.5625), _simd_set1_ps(0.3125), _simd_set1_ps(0.6250), _simd_set1_ps(0.4375),
         _simd_set1_ps(0.3750), _simd_set1_ps(0.8125), _simd_set1_ps(0.6875), _simd_set1_ps(0.1875),
         _simd_set1_ps(0.8750), _simd_set1_ps(0.0625), _simd_set1_ps(0.1250), _simd_set1_ps(0.7500),
         _simd_set1_ps(0.5000), _simd_set1_ps(0.2500), _simd_set1_ps(0.9375), _simd_set1_ps(0.0000)};
        assert(sampleNum < numSamples);
        return Y[sampleNum];
    }

    INLINE static __m128i TileSampleOffsetsX()
    {
        static const uint32_t bboxLeftEdge = 0x00;
        static const uint32_t bboxRightEdge = 0xF0;
                                                            // BR,            BL,           UR,            UL
        static const __m128i tileSampleOffsetX = _mm_set_epi32(bboxRightEdge, bboxLeftEdge, bboxRightEdge, bboxLeftEdge);
        return tileSampleOffsetX;
    }

    INLINE static __m128i TileSampleOffsetsY()
    {
        static const uint32_t bboxTopEdge = 0x00;
        static const uint32_t bboxBottomEdge = 0xF0;
                                                            // BR,             BL,             UR,          UL
        static const __m128i tileSampleOffsetY = _mm_set_epi32(bboxBottomEdge, bboxBottomEdge, bboxTopEdge, bboxTopEdge);
        return tileSampleOffsetY;
    }

    INLINE static uint32_t RasterTileColorOffset(uint32_t sampleNum)
    {
        static const uint32_t RasterTileColorOffsets[numSamples]
        { 0,
          (KNOB_TILE_X_DIM * KNOB_TILE_Y_DIM * FormatTraits<KNOB_COLOR_HOT_TILE_FORMAT>::bpp / 8),
          (KNOB_TILE_X_DIM * KNOB_TILE_Y_DIM * FormatTraits<KNOB_COLOR_HOT_TILE_FORMAT>::bpp / 8) * 2,
          (KNOB_TILE_X_DIM * KNOB_TILE_Y_DIM * FormatTraits<KNOB_COLOR_HOT_TILE_FORMAT>::bpp / 8) * 3,
          (KNOB_TILE_X_DIM * KNOB_TILE_Y_DIM * FormatTraits<KNOB_COLOR_HOT_TILE_FORMAT>::bpp / 8) * 4,
          (KNOB_TILE_X_DIM * KNOB_TILE_Y_DIM * FormatTraits<KNOB_COLOR_HOT_TILE_FORMAT>::bpp / 8) * 5,
          (KNOB_TILE_X_DIM * KNOB_TILE_Y_DIM * FormatTraits<KNOB_COLOR_HOT_TILE_FORMAT>::bpp / 8) * 6,
          (KNOB_TILE_X_DIM * KNOB_TILE_Y_DIM * FormatTraits<KNOB_COLOR_HOT_TILE_FORMAT>::bpp / 8) * 7,
          (KNOB_TILE_X_DIM * KNOB_TILE_Y_DIM * FormatTraits<KNOB_COLOR_HOT_TILE_FORMAT>::bpp / 8) * 8,
          (KNOB_TILE_X_DIM * KNOB_TILE_Y_DIM * FormatTraits<KNOB_COLOR_HOT_TILE_FORMAT>::bpp / 8) * 9,
          (KNOB_TILE_X_DIM * KNOB_TILE_Y_DIM * FormatTraits<KNOB_COLOR_HOT_TILE_FORMAT>::bpp / 8) * 10,
          (KNOB_TILE_X_DIM * KNOB_TILE_Y_DIM * FormatTraits<KNOB_COLOR_HOT_TILE_FORMAT>::bpp / 8) * 11,
          (KNOB_TILE_X_DIM * KNOB_TILE_Y_DIM * FormatTraits<KNOB_COLOR_HOT_TILE_FORMAT>::bpp / 8) * 12,
          (KNOB_TILE_X_DIM * KNOB_TILE_Y_DIM * FormatTraits<KNOB_COLOR_HOT_TILE_FORMAT>::bpp / 8) * 13,
          (KNOB_TILE_X_DIM * KNOB_TILE_Y_DIM * FormatTraits<KNOB_COLOR_HOT_TILE_FORMAT>::bpp / 8) * 14,
          (KNOB_TILE_X_DIM * KNOB_TILE_Y_DIM * FormatTraits<KNOB_COLOR_HOT_TILE_FORMAT>::bpp / 8) * 15,
        };
        assert(sampleNum < numSamples);
        return RasterTileColorOffsets[sampleNum];
    }

    INLINE static uint32_t RasterTileDepthOffset(uint32_t sampleNum)
    {
        static const uint32_t RasterTileDepthOffsets[numSamples]
        { 0,
          (KNOB_TILE_X_DIM * KNOB_TILE_Y_DIM * FormatTraits<KNOB_DEPTH_HOT_TILE_FORMAT>::bpp / 8),
          (KNOB_TILE_X_DIM * KNOB_TILE_Y_DIM * FormatTraits<KNOB_DEPTH_HOT_TILE_FORMAT>::bpp / 8) * 2,
          (KNOB_TILE_X_DIM * KNOB_TILE_Y_DIM * FormatTraits<KNOB_DEPTH_HOT_TILE_FORMAT>::bpp / 8) * 3,
          (KNOB_TILE_X_DIM * KNOB_TILE_Y_DIM * FormatTraits<KNOB_DEPTH_HOT_TILE_FORMAT>::bpp / 8) * 4,
          (KNOB_TILE_X_DIM * KNOB_TILE_Y_DIM * FormatTraits<KNOB_DEPTH_HOT_TILE_FORMAT>::bpp / 8) * 5,
          (KNOB_TILE_X_DIM * KNOB_TILE_Y_DIM * FormatTraits<KNOB_DEPTH_HOT_TILE_FORMAT>::bpp / 8) * 6,
          (KNOB_TILE_X_DIM * KNOB_TILE_Y_DIM * FormatTraits<KNOB_DEPTH_HOT_TILE_FORMAT>::bpp / 8) * 7,
          (KNOB_TILE_X_DIM * KNOB_TILE_Y_DIM * FormatTraits<KNOB_DEPTH_HOT_TILE_FORMAT>::bpp / 8) * 8,
          (KNOB_TILE_X_DIM * KNOB_TILE_Y_DIM * FormatTraits<KNOB_DEPTH_HOT_TILE_FORMAT>::bpp / 8) * 9,
          (KNOB_TILE_X_DIM * KNOB_TILE_Y_DIM * FormatTraits<KNOB_DEPTH_HOT_TILE_FORMAT>::bpp / 8) * 10,
          (KNOB_TILE_X_DIM * KNOB_TILE_Y_DIM * FormatTraits<KNOB_DEPTH_HOT_TILE_FORMAT>::bpp / 8) * 11,
          (KNOB_TILE_X_DIM * KNOB_TILE_Y_DIM * FormatTraits<KNOB_DEPTH_HOT_TILE_FORMAT>::bpp / 8) * 12,
          (KNOB_TILE_X_DIM * KNOB_TILE_Y_DIM * FormatTraits<KNOB_DEPTH_HOT_TILE_FORMAT>::bpp / 8) * 13,
          (KNOB_TILE_X_DIM * KNOB_TILE_Y_DIM * FormatTraits<KNOB_DEPTH_HOT_TILE_FORMAT>::bpp / 8) * 14,
          (KNOB_TILE_X_DIM * KNOB_TILE_Y_DIM * FormatTraits<KNOB_DEPTH_HOT_TILE_FORMAT>::bpp / 8) * 15,
        };
        assert(sampleNum < numSamples);
        return RasterTileDepthOffsets[sampleNum];
    }

    INLINE static uint32_t RasterTileStencilOffset(uint32_t sampleNum)
    {
        static const uint32_t RasterTileStencilOffsets[numSamples]
        { 0,
          (KNOB_TILE_X_DIM * KNOB_TILE_Y_DIM * FormatTraits<KNOB_STENCIL_HOT_TILE_FORMAT>::bpp / 8),
          (KNOB_TILE_X_DIM * KNOB_TILE_Y_DIM * FormatTraits<KNOB_STENCIL_HOT_TILE_FORMAT>::bpp / 8) * 2,
          (KNOB_TILE_X_DIM * KNOB_TILE_Y_DIM * FormatTraits<KNOB_STENCIL_HOT_TILE_FORMAT>::bpp / 8) * 3,
          (KNOB_TILE_X_DIM * KNOB_TILE_Y_DIM * FormatTraits<KNOB_STENCIL_HOT_TILE_FORMAT>::bpp / 8) * 4,
          (KNOB_TILE_X_DIM * KNOB_TILE_Y_DIM * FormatTraits<KNOB_STENCIL_HOT_TILE_FORMAT>::bpp / 8) * 5,
          (KNOB_TILE_X_DIM * KNOB_TILE_Y_DIM * FormatTraits<KNOB_STENCIL_HOT_TILE_FORMAT>::bpp / 8) * 6,
          (KNOB_TILE_X_DIM * KNOB_TILE_Y_DIM * FormatTraits<KNOB_STENCIL_HOT_TILE_FORMAT>::bpp / 8) * 7,
          (KNOB_TILE_X_DIM * KNOB_TILE_Y_DIM * FormatTraits<KNOB_STENCIL_HOT_TILE_FORMAT>::bpp / 8) * 8,
          (KNOB_TILE_X_DIM * KNOB_TILE_Y_DIM * FormatTraits<KNOB_STENCIL_HOT_TILE_FORMAT>::bpp / 8) * 9,
          (KNOB_TILE_X_DIM * KNOB_TILE_Y_DIM * FormatTraits<KNOB_STENCIL_HOT_TILE_FORMAT>::bpp / 8) * 10,
          (KNOB_TILE_X_DIM * KNOB_TILE_Y_DIM * FormatTraits<KNOB_STENCIL_HOT_TILE_FORMAT>::bpp / 8) * 11,
          (KNOB_TILE_X_DIM * KNOB_TILE_Y_DIM * FormatTraits<KNOB_STENCIL_HOT_TILE_FORMAT>::bpp / 8) * 12,
          (KNOB_TILE_X_DIM * KNOB_TILE_Y_DIM * FormatTraits<KNOB_STENCIL_HOT_TILE_FORMAT>::bpp / 8) * 13,
          (KNOB_TILE_X_DIM * KNOB_TILE_Y_DIM * FormatTraits<KNOB_STENCIL_HOT_TILE_FORMAT>::bpp / 8) * 14,
          (KNOB_TILE_X_DIM * KNOB_TILE_Y_DIM * FormatTraits<KNOB_STENCIL_HOT_TILE_FORMAT>::bpp / 8) * 15,
        };
        assert(sampleNum < numSamples);
        return RasterTileStencilOffsets[sampleNum];
    }

    static const uint32_t samplePosX[16];
    static const uint32_t samplePosY[16];
    static const uint32_t numSamples = 16;
    static const uint32_t sampleMask = 0xFFFF;
};
