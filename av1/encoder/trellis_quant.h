/*
 * Copyright (c) 2024, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 3-Clause Clear License
 * and the Alliance for Open Media Patent License 1.0. If the BSD 3-Clause Clear
 * License was not distributed with this source code in the LICENSE file, you
 * can obtain it at aomedia.org/license/software-license/bsd-3-c-c/.  If the
 * Alliance for Open Media Patent License 1.0 was not distributed with this
 * source code in the PATENTS file, you can obtain it at
 * aomedia.org/license/patent-license/.
 */

#ifndef AOM_AV1_ENCODER_TRELLIS_QUANT_H_
#define AOM_AV1_ENCODER_TRELLIS_QUANT_H_

#include "config/aom_config.h"

#include "av1/common/av1_common_int.h"
#include "av1/common/blockd.h"
#include "av1/common/txb_common.h"
#include "av1/encoder/block.h"
#include "av1/encoder/encoder.h"

#ifdef __cplusplus
extern "C" {
#endif

int av1_dep_quant(const struct AV1_COMP *cpi, MACROBLOCK *x, int plane,
                  int block, TX_SIZE tx_size, TX_TYPE tx_type,
                  CctxType cctx_type, const TXB_CTX *const txb_ctx,
                  int *rate_cost, int sharpness);

#ifdef __cplusplus
}
#endif

#endif  // AOM_AV1_ENCODER_TRELLIS_QUANT_H_
