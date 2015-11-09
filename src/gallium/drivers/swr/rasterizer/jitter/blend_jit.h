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
* @file blend_jit.h
*
* @brief Definition of the blend jitter
*
* Notes:
*
******************************************************************************/
#pragma once

#include "common/formats.h"
#include "core/context.h"
#include "core/state.h"

struct RENDER_TARGET_BLEND_COMPILE_STATE
{
    bool blendEnable;
    SWR_BLEND_FACTOR sourceAlphaBlendFactor;
    SWR_BLEND_FACTOR destAlphaBlendFactor;
    SWR_BLEND_FACTOR sourceBlendFactor;
    SWR_BLEND_FACTOR destBlendFactor;
    SWR_BLEND_OP colorBlendFunc;
    SWR_BLEND_OP alphaBlendFunc;
};

enum ALPHA_TEST_FORMAT
{
    ALPHA_TEST_UNORM8,
    ALPHA_TEST_FLOAT32
};

//////////////////////////////////////////////////////////////////////////
/// State required for blend jit
//////////////////////////////////////////////////////////////////////////
struct BLEND_COMPILE_STATE
{
    SWR_FORMAT format;          // format of render target being blended
    bool independentAlphaBlendEnable;
    RENDER_TARGET_BLEND_COMPILE_STATE blendState;

    bool alphaTestEnable;
    SWR_ZFUNCTION alphaTestFunction;
    ALPHA_TEST_FORMAT alphaTestFormat;

    bool operator==(const BLEND_COMPILE_STATE& other) const
    {
        return memcmp(this, &other, sizeof(BLEND_COMPILE_STATE)) == 0;
    }
};
