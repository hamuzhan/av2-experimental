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

#include "av1/encoder/trellis_quant.h"

#include "aom_ports/mem.h"
#include "av1/common/blockd.h"
#include "av1/common/idct.h"
#include "av1/common/pred_common.h"
#include "av1/common/reconintra.h"
#include "av1/common/scan.h"
#include "av1/encoder/bitstream.h"
#include "av1/encoder/cost.h"
#include "av1/encoder/encodeframe.h"
#include "av1/encoder/rdopt.h"
#include "av1/encoder/tokenize.h"

typedef struct _PQData {
  tran_low_t absLevel;
  int64_t deltaDist;
  tran_low_t dqc;
} PQData;

static AOM_INLINE void set_levels_buf(DECISION *decision, uint8_t *levels,
                                      const int16_t *scan, const int eob_minus1,
                                      const int scan_pos, const int bwl,
                                      const int sharpness) {
  if (decision->prevId == -2) {
    return;
  }
  // update current abs level
  levels[get_padded_idx(scan[scan_pos], bwl)] =
      AOMMIN(decision->absLevel, INT8_MAX);
  // check current node is a new start position? if so, set all previous
  // position to 0. prevId == -1 means a new start, prevId == -2 ?
  bool new_eob =
      decision->prevId < 0 && scan_pos + 1 <= eob_minus1 && sharpness == 0;
  if (new_eob) {
    for (int si = scan_pos + 1; si <= eob_minus1; si++) {
      levels[get_padded_idx(scan[si], bwl)] = 0;
    }
  }
}

static INLINE int get_low_range(int abs_qc, int lf) {
  int base_levels = lf ? 6 : 4;
  int parity = abs_qc & 1;
#if ((COEFF_BASE_RANGE & 1) == 1)
  int br_max = COEFF_BASE_RANGE + base_levels - 1 - parity;
  int low = AOMMIN(abs_qc, br_max);
  low -= base_levels - 1;
#else
  int abs2 = abs_qc & ~1;
  int low = AOMMIN(abs2, COEFF_BASE_RANGE + base_levels - 2) + parity;
  low -= base_levels - 1;
#endif
  return low;
}

static INLINE int get_high_range(int abs_qc, int lf) {
  int base_levels = lf ? 6 : 4;
  int low_range = get_low_range(abs_qc, lf);
  int high_range = (abs_qc - low_range - (base_levels - 1)) >> 1;
  return high_range;
}

static AOM_FORCE_INLINE int64_t get_coeff_dist(tran_low_t tcoeff,
                                               tran_low_t dqcoeff, int shift) {
  const int64_t diff = (tcoeff - dqcoeff) * (1 << shift);
  const int64_t error = diff * diff;
  return error;
}

static const int8_t eob_to_pos_small[33] = {
  0, 1, 2,                                        // 0-2
  3, 3,                                           // 3-4
  4, 4, 4, 4,                                     // 5-8
  5, 5, 5, 5, 5, 5, 5, 5,                         // 9-16
  6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6  // 17-32
};

static const int8_t eob_to_pos_large[17] = {
  6,                               // place holder
  7,                               // 33-64
  8,  8,                           // 65-128
  9,  9,  9,  9,                   // 129-256
  10, 10, 10, 10, 10, 10, 10, 10,  // 257-512
  11                               // 513-
};

static AOM_FORCE_INLINE int get_eob_pos_token(const int eob, int *const extra) {
  int t;

  if (eob < 33) {
    t = eob_to_pos_small[eob];
  } else {
    const int e = AOMMIN((eob - 1) >> 5, 16);
    t = eob_to_pos_large[e];
  }

  *extra = eob - av1_eob_group_start[t];

  return t;
}

static int get_eob_cost(int eob, const LV_MAP_EOB_COST *txb_eob_costs,
                        const LV_MAP_COEFF_COST *txb_costs
#if CONFIG_EOB_POS_LUMA
                        ,
                        const int is_inter
#endif  // CONFIG_EOB_POS_LUMA
) {
  int eob_cost = 0;
  int eob_extra;
  const int eob_pt = get_eob_pos_token(eob, &eob_extra);
  eob_cost += txb_eob_costs->eob_cost
#if CONFIG_EOB_POS_LUMA
                  [is_inter]
#endif  // CONFIG_EOB_POS_LUMA
                  [eob_pt - 1];
  const int eob_offset_bits = av1_eob_offset_bits[eob_pt];
  if (eob_offset_bits > 0) {
    const int eob_ctx = eob_pt - 3;
    const int eob_shift = eob_offset_bits - 1;
    const int bit = (eob_extra & (1 << eob_shift)) ? 1 : 0;
    eob_cost += txb_costs->eob_extra_cost[eob_ctx][bit];
    if (eob_offset_bits > 1) eob_cost += av1_cost_literal(eob_offset_bits - 1);
  }
  return eob_cost;
}

static const int golomb_bits_cost[32] = {
  0,       512,     512 * 3, 512 * 3, 512 * 5, 512 * 5, 512 * 5, 512 * 5,
  512 * 7, 512 * 7, 512 * 7, 512 * 7, 512 * 7, 512 * 7, 512 * 7, 512 * 7,
  512 * 9, 512 * 9, 512 * 9, 512 * 9, 512 * 9, 512 * 9, 512 * 9, 512 * 9,
  512 * 9, 512 * 9, 512 * 9, 512 * 9, 512 * 9, 512 * 9, 512 * 9, 512 * 9
};
static const int golomb_cost_diff[32] = {
  0,       512, 512 * 2, 0, 512 * 2, 0, 0, 0, 512 * 2, 0, 0, 0, 0, 0, 0, 0,
  512 * 2, 0,   0,       0, 0,       0, 0, 0, 0,       0, 0, 0, 0, 0, 0, 0
};

static AOM_FORCE_INLINE int get_golomb_cost(int abs_qc) {
#if NEWHR
  if (abs_qc >= NUM_BASE_LEVELS + COEFF_BASE_RANGE) {
    const int r = 1 + get_high_range(abs_qc, 0);
    const int length = get_msb(r) + 1;
    return av1_cost_literal(2 * length - 1);
  }
#else
  if (abs_qc >= 1 + NUM_BASE_LEVELS + COEFF_BASE_RANGE) {
    const int r = abs_qc - COEFF_BASE_RANGE - NUM_BASE_LEVELS;
    const int length = get_msb(r) + 1;
    return av1_cost_literal(2 * length - 1);
  }
#endif
  return 0;
}

// Golomb cost of coding bypass coded level values in the
// low-frequency region.
static AOM_FORCE_INLINE int get_golomb_cost_lf(int abs_qc) {
#if NEWHR
  if (abs_qc >= LF_NUM_BASE_LEVELS + COEFF_BASE_RANGE) {
    const int r = 1 + get_high_range(abs_qc, 1);
    const int length = get_msb(r) + 1;
    return av1_cost_literal(2 * length - 1);
  }
#else
  if (abs_qc >= 1 + LF_NUM_BASE_LEVELS + COEFF_BASE_RANGE) {
    const int r = abs_qc - COEFF_BASE_RANGE - LF_NUM_BASE_LEVELS;
    const int length = get_msb(r) + 1;
    return av1_cost_literal(2 * length - 1);
  }
#endif
  return 0;
}

static AOM_FORCE_INLINE int get_br_cost(tran_low_t level,
                                        const int *coeff_lps) {
  const int base_range = AOMMIN(level - 1 - NUM_BASE_LEVELS, COEFF_BASE_RANGE);
  return coeff_lps[base_range] + get_golomb_cost(level);
}

// Base range cost of coding level values in the
// low-frequency region, includes the bypass cost.
static AOM_FORCE_INLINE int get_br_lf_cost(tran_low_t level,
                                           const int *coeff_lps) {
  const int base_range =
      AOMMIN(level - 1 - LF_NUM_BASE_LEVELS, COEFF_BASE_RANGE);
  return coeff_lps[base_range] + get_golomb_cost_lf(level);
}

static INLINE int get_coeff_cost_eob(int ci, tran_low_t abs_qc, int sign,
                                     int coeff_ctx, int dc_sign_ctx,
                                     const LV_MAP_COEFF_COST *txb_costs,
                                     int bwl, TX_CLASS tx_class
#if CONFIG_CONTEXT_DERIVATION
                                     ,
                                     int32_t *tmp_sign
#endif  // CONFIG_CONTEXT_DERIVATION
                                     ,
                                     int plane) {
  int cost = 0;
  const int row = ci >> bwl;
  const int col = ci - (row << bwl);
  int limits = get_lf_limits(row, col, tx_class, plane);
#if CONFIG_LCCHROMA
  const int(*base_lf_eob_cost_ptr)[LF_BASE_SYMBOLS - 1] =
      plane > 0 ? txb_costs->base_lf_eob_cost_uv : txb_costs->base_lf_eob_cost;
  const int(*base_eob_cost_ptr)[3] =
      plane > 0 ? txb_costs->base_eob_cost_uv : txb_costs->base_eob_cost;

  cost += limits ? base_lf_eob_cost_ptr[coeff_ctx]
                                       [AOMMIN(abs_qc, LF_BASE_SYMBOLS - 1) - 1]
                 : base_eob_cost_ptr[coeff_ctx][AOMMIN(abs_qc, 3) - 1];
#else
  if (limits) {
    cost +=
        txb_costs->base_lf_eob_cost[coeff_ctx]
                                   [AOMMIN(abs_qc, LF_BASE_SYMBOLS - 1) - 1];
  } else {
    cost += txb_costs->base_eob_cost[coeff_ctx][AOMMIN(abs_qc, 3) - 1];
  }
#endif  // CONFIG_LCCHROMA
  if (abs_qc != 0) {
#if CONFIG_IMPROVEIDTX_CTXS
    const int dc_ph_group = 0;  // PH disabled
    const bool dc_2dtx = (ci == 0);
    const bool dc_hor = (col == 0) && tx_class == TX_CLASS_HORIZ;
    const bool dc_ver = (row == 0) && tx_class == TX_CLASS_VERT;
    if (dc_2dtx || dc_hor || dc_ver) {
      if (plane == AOM_PLANE_V)
        cost += txb_costs->v_dc_sign_cost[tmp_sign[ci]][dc_sign_ctx][sign];
      else
        cost += txb_costs->dc_sign_cost[dc_ph_group][dc_sign_ctx][sign];
    } else {
      if (plane == AOM_PLANE_V)
        cost += txb_costs->v_ac_sign_cost[tmp_sign[ci]][sign];
      else
        cost += av1_cost_literal(1);
    }
#else
    if (ci == 0) {
#if CONFIG_CONTEXT_DERIVATION
      if (plane == AOM_PLANE_V)
        cost += txb_costs->v_dc_sign_cost[tmp_sign[0]][dc_sign_ctx][sign];
      else
        cost += txb_costs->dc_sign_cost[dc_sign_ctx][sign];
#else
      cost += txb_costs->dc_sign_cost[dc_sign_ctx][sign];
#endif  // CONFIG_CONTEXT_DERIVATION
    } else {
#if CONFIG_CONTEXT_DERIVATION
      if (plane == AOM_PLANE_V)
        cost += txb_costs->v_ac_sign_cost[tmp_sign[ci]][sign];
      else
        cost += av1_cost_literal(1);
#else
      cost += av1_cost_literal(1);
#endif  // CONFIG_CONTEXT_DERIVATION
    }
#endif  // CONFIG_IMPROVEIDTX_CTXS
#if CONFIG_LCCHROMA
    if (plane > 0) {
      if (limits) {
        if (abs_qc > LF_NUM_BASE_LEVELS) {
          int br_ctx = get_br_ctx_lf_eob_chroma(ci, tx_class);
          cost += get_br_lf_cost(abs_qc, txb_costs->lps_lf_cost_uv[br_ctx]);
        }
      } else {
        if (abs_qc > NUM_BASE_LEVELS) {
          int br_ctx = 0; /* get_br_ctx_eob_chroma */
          cost += get_br_cost(abs_qc, txb_costs->lps_cost_uv[br_ctx]);
        }
      }
    } else {
      if (limits) {
        if (abs_qc > LF_NUM_BASE_LEVELS) {
          int br_ctx = get_br_ctx_lf_eob(ci, tx_class);
          cost += get_br_lf_cost(abs_qc, txb_costs->lps_lf_cost[br_ctx]);
        }
      } else {
        if (abs_qc > NUM_BASE_LEVELS) {
          int br_ctx = 0; /* get_br_ctx_eob */
          cost += get_br_cost(abs_qc, txb_costs->lps_cost[br_ctx]);
        }
      }
    }
#else
    if (limits) {
      if (abs_qc > LF_NUM_BASE_LEVELS) {
        int br_ctx = get_br_ctx_lf_eob(ci, tx_class);
        cost += get_br_lf_cost(abs_qc, txb_costs->lps_lf_cost[br_ctx]);
      }
    } else {
      if (abs_qc > NUM_BASE_LEVELS) {
        int br_ctx = 0; /* get_br_ctx_eob */
        cost += get_br_cost(abs_qc, txb_costs->lps_cost[br_ctx]);
      }
    }
#endif  // CONFIG_LCCHROMA
  }
  return cost;
}

static INLINE int get_coeff_cost_general(int is_last, int ci, tran_low_t abs_qc,
                                         int sign, int coeff_ctx,
                                         int dc_sign_ctx,
                                         const LV_MAP_COEFF_COST *txb_costs,
                                         int bwl, TX_CLASS tx_class,
                                         const uint8_t *levels
#if CONFIG_CONTEXT_DERIVATION
                                         ,
                                         int32_t *tmp_sign
#endif  // CONFIG_CONTEXT_DERIVATION
                                         ,
                                         int plane, int limits) {
  int cost = 0;
  if (is_last) {
#if CONFIG_LCCHROMA
    const int(*base_lf_eob_cost_ptr)[LF_BASE_SYMBOLS - 1] =
        plane > 0 ? txb_costs->base_lf_eob_cost_uv
                  : txb_costs->base_lf_eob_cost;
    const int(*base_eob_cost_ptr)[3] =
        plane > 0 ? txb_costs->base_eob_cost_uv : txb_costs->base_eob_cost;

    cost += limits
                ? base_lf_eob_cost_ptr[coeff_ctx]
                                      [AOMMIN(abs_qc, LF_BASE_SYMBOLS - 1) - 1]
                : base_eob_cost_ptr[coeff_ctx][AOMMIN(abs_qc, 3) - 1];

#else
    if (limits) {
      cost +=
          txb_costs->base_lf_eob_cost[coeff_ctx]
                                     [AOMMIN(abs_qc, LF_BASE_SYMBOLS - 1) - 1];
    } else {
      cost += txb_costs->base_eob_cost[coeff_ctx][AOMMIN(abs_qc, 3) - 1];
    }
#endif  // CONFIG_LCCHROMA
  } else {
#if CONFIG_LCCHROMA && CONFIG_DQ
    const int(*base_lf_cost_ptr)[DQ_CTXS][LF_BASE_SYMBOLS * 2] =
        plane > 0 ? txb_costs->base_lf_cost_uv : txb_costs->base_lf_cost;
    const int(*base_cost_ptr)[DQ_CTXS][8] =
        plane > 0 ? txb_costs->base_cost_uv : txb_costs->base_cost;
    cost += limits ? base_lf_cost_ptr[coeff_ctx][0]
                                     [AOMMIN(abs_qc, LF_BASE_SYMBOLS - 1)]
                   : base_cost_ptr[coeff_ctx][0][AOMMIN(abs_qc, 3)];
#elif CONFIG_LCCHROMA
    const int(*base_lf_cost_ptr)[LF_BASE_SYMBOLS * 2] =
        plane > 0 ? txb_costs->base_lf_cost_uv : txb_costs->base_lf_cost;
    const int(*base_cost_ptr)[8] =
        plane > 0 ? txb_costs->base_cost_uv : txb_costs->base_cost;
    cost +=
        limits
            ? base_lf_cost_ptr[coeff_ctx][AOMMIN(abs_qc, LF_BASE_SYMBOLS - 1)]
            : base_cost_ptr[coeff_ctx][AOMMIN(abs_qc, 3)];
#else
    if (limits) {
      cost +=
          txb_costs
              ->base_lf_cost[coeff_ctx][AOMMIN(abs_qc, LF_BASE_SYMBOLS - 1)];
    } else {
      cost += txb_costs->base_cost[coeff_ctx][AOMMIN(abs_qc, 3)];
    }
#endif  // CONFIG_LCCHROMA
  }
  if (abs_qc != 0) {
#if CONFIG_IMPROVEIDTX_CTXS
    const int dc_ph_group = 0;  // PH disabled
    const int row = ci >> bwl;
    const int col = ci - (row << bwl);
    const bool dc_2dtx = (ci == 0);
    const bool dc_hor = (col == 0) && tx_class == TX_CLASS_HORIZ;
    const bool dc_ver = (row == 0) && tx_class == TX_CLASS_VERT;
    if (dc_2dtx || dc_hor || dc_ver) {
      if (plane == AOM_PLANE_V)
        cost += txb_costs->v_dc_sign_cost[tmp_sign[ci]][dc_sign_ctx][sign];
      else
        cost += txb_costs->dc_sign_cost[dc_ph_group][dc_sign_ctx][sign];
    } else {
      if (plane == AOM_PLANE_V)
        cost += txb_costs->v_ac_sign_cost[tmp_sign[ci]][sign];
      else
        cost += av1_cost_literal(1);
    }
#else
    if (ci == 0) {
#if CONFIG_CONTEXT_DERIVATION
      if (plane == AOM_PLANE_V)
        cost += txb_costs->v_dc_sign_cost[tmp_sign[0]][dc_sign_ctx][sign];
      else
        cost += txb_costs->dc_sign_cost[dc_sign_ctx][sign];
#else
      cost += txb_costs->dc_sign_cost[dc_sign_ctx][sign];
#endif  // CONFIG_CONTEXT_DERIVATION
    } else {
#if CONFIG_CONTEXT_DERIVATION
      if (plane == AOM_PLANE_V)
        cost += txb_costs->v_ac_sign_cost[tmp_sign[ci]][sign];
      else
        cost += av1_cost_literal(1);
#else
      cost += av1_cost_literal(1);
#endif  // CONFIG_CONTEXT_DERIVATION
    }
#endif  // CONFIG_IMPROVEIDTX_CTXS
#if CONFIG_LCCHROMA
    if (plane > 0) {
      if (limits) {
        if (abs_qc > LF_NUM_BASE_LEVELS) {
          int br_ctx;
          if (is_last)
            br_ctx = get_br_ctx_lf_eob_chroma(ci, tx_class);
          else
            br_ctx = get_br_lf_ctx_chroma(levels, ci, bwl, tx_class);
          cost += get_br_lf_cost(abs_qc, txb_costs->lps_lf_cost_uv[br_ctx]);
        }
      } else {
        if (abs_qc > NUM_BASE_LEVELS) {
          int br_ctx;
          if (is_last)
            br_ctx = 0; /* get_br_ctx_eob_chroma */
          else
            br_ctx = get_br_ctx_chroma(levels, ci, bwl, tx_class);
          cost += get_br_cost(abs_qc, txb_costs->lps_cost_uv[br_ctx]);
        }
      }
    } else {
      if (limits) {
        if (abs_qc > LF_NUM_BASE_LEVELS) {
          int br_ctx;
          if (is_last)
            br_ctx = get_br_ctx_lf_eob(ci, tx_class);
          else
            br_ctx = get_br_lf_ctx(levels, ci, bwl, tx_class);
          cost += get_br_lf_cost(abs_qc, txb_costs->lps_lf_cost[br_ctx]);
        }
      } else {
        if (abs_qc > NUM_BASE_LEVELS) {
          int br_ctx;
          if (is_last)
            br_ctx = 0; /* get_br_ctx_eob */
          else
            br_ctx = get_br_ctx(levels, ci, bwl, tx_class);
          cost += get_br_cost(abs_qc, txb_costs->lps_cost[br_ctx]);
        }
      }
    }
#else
    if (limits) {
      if (abs_qc > LF_NUM_BASE_LEVELS) {
        int br_ctx;
        if (is_last)
          br_ctx = get_br_ctx_lf_eob(ci, tx_class);
        else
          br_ctx = get_br_lf_ctx(levels, ci, bwl, tx_class);
        cost += get_br_lf_cost(abs_qc, txb_costs->lps_lf_cost[br_ctx]);
      }
    } else {
      if (abs_qc > NUM_BASE_LEVELS) {
        int br_ctx;
        if (is_last)
          br_ctx = 0; /* get_br_ctx_eob */
        else
          br_ctx = get_br_ctx(levels, ci, bwl, tx_class);
        cost += get_br_cost(abs_qc, txb_costs->lps_cost[br_ctx]);
      }
    }
#endif  // CONFIG_LCCHROMA
  }
  return cost;
}

static INLINE int get_coeff_cost_general_q1(int is_last, int ci,
                                            tran_low_t abs_qc, int sign,
                                            int coeff_ctx, int dc_sign_ctx,
                                            const LV_MAP_COEFF_COST *txb_costs,
                                            int bwl, TX_CLASS tx_class,
                                            const uint8_t *levels
#if CONFIG_CONTEXT_DERIVATION
                                            ,
                                            int32_t *tmp_sign
#endif  // CONFIG_CONTEXT_DERIVATION
                                            ,
                                            int plane, int limits) {
  int cost = 0;
  if (is_last) {
    // quantizer 1 can not be used for last non-zero coeff.
    assert(0);
  } else {
#if CONFIG_LCCHROMA
    const int(*base_lf_cost_ptr)[DQ_CTXS][LF_BASE_SYMBOLS * 2] =
        plane > 0 ? txb_costs->base_lf_cost_uv : txb_costs->base_lf_cost;
    const int(*base_cost_ptr)[DQ_CTXS][8] =
        plane > 0 ? txb_costs->base_cost_uv : txb_costs->base_cost;
    cost += limits ? base_lf_cost_ptr[coeff_ctx][1]
                                     [AOMMIN(abs_qc, LF_BASE_SYMBOLS - 1)]
                   : base_cost_ptr[coeff_ctx][1][AOMMIN(abs_qc, 3)];
#else
    if (limits) {
      cost += txb_costs->base_lf_cost_tcq[coeff_ctx]
                                         [AOMMIN(abs_qc, LF_BASE_SYMBOLS - 1)];
    } else {
      cost += txb_costs->base_cost_tcq[coeff_ctx][AOMMIN(abs_qc, 3)];
    }
#endif  // CONFIG_LCCHROMA
  }
  if (abs_qc != 0) {
#if CONFIG_IMPROVEIDTX_CTXS
    const int dc_ph_group = 0;  // PH disabled
    const int row = ci >> bwl;
    const int col = ci - (row << bwl);
    const bool dc_2dtx = (ci == 0);
    const bool dc_hor = (col == 0) && tx_class == TX_CLASS_HORIZ;
    const bool dc_ver = (row == 0) && tx_class == TX_CLASS_VERT;
    if (dc_2dtx || dc_hor || dc_ver) {
      if (plane == AOM_PLANE_V)
        cost += txb_costs->v_dc_sign_cost[tmp_sign[ci]][dc_sign_ctx][sign];
      else
        cost += txb_costs->dc_sign_cost[dc_ph_group][dc_sign_ctx][sign];
    } else {
      if (plane == AOM_PLANE_V)
        cost += txb_costs->v_ac_sign_cost[tmp_sign[ci]][sign];
      else
        cost += av1_cost_literal(1);
    }
#else
    if (ci == 0) {
#if CONFIG_CONTEXT_DERIVATION
      if (plane == AOM_PLANE_V)
        cost += txb_costs->v_dc_sign_cost[tmp_sign[0]][dc_sign_ctx][sign];
      else
        cost += txb_costs->dc_sign_cost[dc_sign_ctx][sign];
#else
      cost += txb_costs->dc_sign_cost[dc_sign_ctx][sign];
#endif  // CONFIG_CONTEXT_DERIVATION
    } else {
#if CONFIG_CONTEXT_DERIVATION
      if (plane == AOM_PLANE_V)
        cost += txb_costs->v_ac_sign_cost[tmp_sign[ci]][sign];
      else
        cost += av1_cost_literal(1);
#else
      cost += av1_cost_literal(1);
#endif  // CONFIG_CONTEXT_DERIVATION
    }
#endif  // CONFIG_IMPROVEIDTX_CTXS
#if CONFIG_LCCHROMA
    if (plane > 0) {
      if (limits) {
        if (abs_qc > LF_NUM_BASE_LEVELS) {
          int br_ctx;
          if (is_last)
            br_ctx = get_br_ctx_lf_eob_chroma(ci, tx_class);
          else
            br_ctx = get_br_lf_ctx_chroma(levels, ci, bwl, tx_class);
          cost += get_br_lf_cost(abs_qc, txb_costs->lps_lf_cost_uv[br_ctx]);
        }
      } else {
        if (abs_qc > NUM_BASE_LEVELS) {
          int br_ctx;
          if (is_last)
            br_ctx = 0; /* get_br_ctx_eob_chroma */
          else
            br_ctx = get_br_ctx_chroma(levels, ci, bwl, tx_class);
          cost += get_br_cost(abs_qc, txb_costs->lps_cost_uv[br_ctx]);
        }
      }
    } else {
      if (limits) {
        if (abs_qc > LF_NUM_BASE_LEVELS) {
          int br_ctx;
          if (is_last)
            br_ctx = get_br_ctx_lf_eob(ci, tx_class);
          else
            br_ctx = get_br_lf_ctx(levels, ci, bwl, tx_class);
          cost += get_br_lf_cost(abs_qc, txb_costs->lps_lf_cost[br_ctx]);
        }
      } else {
        if (abs_qc > NUM_BASE_LEVELS) {
          int br_ctx;
          if (is_last)
            br_ctx = 0; /* get_br_ctx_eob */
          else
            br_ctx = get_br_ctx(levels, ci, bwl, tx_class);
          cost += get_br_cost(abs_qc, txb_costs->lps_cost[br_ctx]);
        }
      }
    }
#else
    if (limits) {
      if (abs_qc > LF_NUM_BASE_LEVELS) {
        int br_ctx;
        if (is_last)
          br_ctx = get_br_ctx_lf_eob(ci, tx_class);
        else
          br_ctx = get_br_lf_ctx(levels, ci, bwl, tx_class);
        cost += get_br_lf_cost(abs_qc, txb_costs->lps_lf_cost[br_ctx]);
      }
    } else {
      if (abs_qc > NUM_BASE_LEVELS) {
        int br_ctx;
        if (is_last)
          br_ctx = 0; /* get_br_ctx_eob */
        else
          br_ctx = get_br_ctx(levels, ci, bwl, tx_class);
        cost += get_br_cost(abs_qc, txb_costs->lps_cost[br_ctx]);
      }
    }
#endif  // CONFIG_LCCHROMA
  }
  return cost;
}

static void decide(int64_t distA, int64_t distB, int64_t distzero,
                   int64_t rdmult, int rateA, int rateB, int rate_zero,
                   PQData *pqDataA, PQData *pqDataB, int limits, int prev_state,
                   DECISION *decision_02, DECISION *decision_1) {
  int parityA = tcq_parity(pqDataA->absLevel, limits);
  int parityB = tcq_parity(pqDataB->absLevel, limits);
  int64_t costA = RDCOST(rdmult, rateA, distA);
  int64_t costB = RDCOST(rdmult, rateB, distB);
  int64_t cost_zero = RDCOST(rdmult, rate_zero, distzero);
  if (parityA) {
    if (costA < decision_1->rdCost) {
      decision_1->rdCost = costA;
      decision_1->dist = distA;
      decision_1->rate = rateA;
      decision_1->prevId = prev_state;
      decision_1->absLevel = pqDataA->absLevel;
      decision_1->dqc = pqDataA->dqc;
    }
  } else {
    if (costA < decision_02->rdCost) {
      decision_02->rdCost = costA;
      decision_02->dist = distA;
      decision_02->rate = rateA;
      decision_02->prevId = prev_state;
      decision_02->absLevel = pqDataA->absLevel;
      decision_02->dqc = pqDataA->dqc;
    }
  }

  if (parityB) {
    if (costB < decision_1->rdCost) {
      decision_1->rdCost = costB;
      decision_1->dist = distB;
      decision_1->rate = rateB;
      decision_1->prevId = prev_state;
      decision_1->absLevel = pqDataB->absLevel;
      decision_1->dqc = pqDataB->dqc;
    }
  } else {
    if (costB < decision_02->rdCost) {
      decision_02->rdCost = costB;
      decision_02->dist = distB;
      decision_02->rate = rateB;
      decision_02->prevId = prev_state;
      decision_02->absLevel = pqDataB->absLevel;
      decision_02->dqc = pqDataB->dqc;
    }
  }
  if (cost_zero < decision_02->rdCost) {
    decision_02->rdCost = cost_zero;
    decision_02->dist = distzero;
    decision_02->rate = rate_zero;
    decision_02->prevId = prev_state;
    decision_02->absLevel = 0;
    decision_02->dqc = 0;
  }
}

void pre_quant(tran_low_t tqc, PQData *pqData, const int32_t *quant_ptr,
               int dqv, int log_scale, int scan_pos) {
  // calculate qIdx
  const int shift = 16 - log_scale + QUANT_FP_BITS;
  tran_low_t add = -((3 << shift) >> 1);
  tran_low_t abs_tqc = abs(tqc);

  tran_low_t qIdx = (int)AOMMAX(
      1, AOMMIN(((1 << 16) - 1),
                ((int64_t)abs_tqc * quant_ptr[scan_pos != 0] + add) >> shift));

  const int64_t dist0 = get_coeff_dist(abs_tqc, 0, log_scale - 1);

  int Idx_a = qIdx & 3;

  tran_low_t dqca = (tran_low_t)ROUND_POWER_OF_TWO_64((tran_high_t)qIdx * dqv,
                                                      QUANT_TABLE_BITS) >>
                    log_scale;

  pqData[Idx_a].absLevel = (++qIdx) >> 1;
  pqData[Idx_a].deltaDist =
      get_coeff_dist(abs_tqc, dqca, log_scale - 1) - dist0;
  pqData[Idx_a].dqc = dqca;

  int Idx_b = qIdx & 3;

  tran_low_t dqcb = (tran_low_t)ROUND_POWER_OF_TWO_64((tran_high_t)qIdx * dqv,
                                                      QUANT_TABLE_BITS) >>
                    log_scale;

  pqData[Idx_b].absLevel = (++qIdx) >> 1;
  pqData[Idx_b].deltaDist =
      get_coeff_dist(abs_tqc, dqcb, log_scale - 1) - dist0;
  pqData[Idx_b].dqc = dqcb;

  int Idx_c = qIdx & 3;

  tran_low_t dqcc = (tran_low_t)ROUND_POWER_OF_TWO_64((tran_high_t)qIdx * dqv,
                                                      QUANT_TABLE_BITS) >>
                    log_scale;

  pqData[Idx_c].absLevel = (++qIdx) >> 1;
  pqData[Idx_c].deltaDist =
      get_coeff_dist(abs_tqc, dqcc, log_scale - 1) - dist0;
  pqData[Idx_c].dqc = dqcc;

  int Idx_d = qIdx & 3;

  tran_low_t dqcd = (tran_low_t)ROUND_POWER_OF_TWO_64((tran_high_t)qIdx * dqv,
                                                      QUANT_TABLE_BITS) >>
                    log_scale;

  pqData[Idx_d].absLevel = (++qIdx) >> 1;
  pqData[Idx_d].deltaDist =
      get_coeff_dist(abs_tqc, dqcd, log_scale - 1) - dist0;
  pqData[Idx_d].dqc = dqcd;
}

// This function gets the estimated bit cost for a 'secondary tx set'
static int get_sec_tx_set_cost(const MACROBLOCK *x, const MB_MODE_INFO *mbmi,
                               TX_TYPE tx_type) {
  uint8_t stx_set_flag = get_secondary_tx_set(tx_type);
  if (get_primary_tx_type(tx_type) == ADST_ADST) stx_set_flag -= IST_DIR_SIZE;
  assert(stx_set_flag < IST_DIR_SIZE);
  uint8_t intra_mode = get_intra_mode(mbmi, PLANE_TYPE_Y);
#if CONFIG_INTRA_TX_IST_PARSE
  return x->mode_costs.most_probable_stx_set_flag_cost
      [most_probable_stx_mapping[intra_mode][stx_set_flag]];
#else
  uint8_t stx_set_ctx = stx_transpose_mapping[intra_mode];
  assert(stx_set_ctx < IST_DIR_SIZE);
  return x->mode_costs.stx_set_flag_cost[stx_set_ctx][stx_set_flag];
#endif  // CONFIG_INTRA_TX_IST_PARSE
}

// TODO(angiebird): use this function whenever it's possible
static int get_tx_type_cost(const MACROBLOCK *x, const MACROBLOCKD *xd,
                            int plane, TX_SIZE tx_size, TX_TYPE tx_type,
                            int reduced_tx_set_used, int eob, int bob_code,
                            int is_fsc) {
  if (plane > 0) return 0;

  const TX_SIZE square_tx_size = txsize_sqr_map[tx_size];

  const MB_MODE_INFO *mbmi = xd->mi[0];
  if (mbmi->fsc_mode[xd->tree_type == CHROMA_PART] &&
      !is_inter_block(mbmi, xd->tree_type) && plane == PLANE_TYPE_Y) {
    return 0;
  }
  const int is_inter = is_inter_block(mbmi, xd->tree_type);
  if (get_ext_tx_types(tx_size, is_inter, reduced_tx_set_used) > 1 &&
      !xd->lossless[xd->mi[0]->segment_id]) {
    const int ext_tx_set =
        get_ext_tx_set(tx_size, is_inter, reduced_tx_set_used);
    if (is_inter) {
      if (ext_tx_set > 0) {
        const int esc_eob = is_fsc ? bob_code : eob;
        const int eob_tx_ctx =
            get_lp2tx_ctx(tx_size, get_txb_bwl(tx_size), esc_eob);
#if CONFIG_INTER_IST
        int tx_type_cost = 0;
        tx_type_cost =
            x->mode_costs
                .inter_tx_type_costs[ext_tx_set][eob_tx_ctx][square_tx_size]
                                    [get_primary_tx_type(tx_type)];
        if (block_signals_sec_tx_type(xd, tx_size, tx_type, eob) &&
            xd->enable_ist) {
          tx_type_cost +=
              x->mode_costs.stx_flag_cost[is_inter][square_tx_size]
                                         [get_secondary_tx_type(tx_type)];
        }
        return tx_type_cost;
#else
        return x->mode_costs.inter_tx_type_costs[ext_tx_set][eob_tx_ctx]
                                                [square_tx_size][tx_type];
#endif  // CONFIG_INTER_IST
      }
    } else {
      if (ext_tx_set > 0) {
        PREDICTION_MODE intra_dir;
        if (mbmi->filter_intra_mode_info.use_filter_intra)
          intra_dir = fimode_to_intradir[mbmi->filter_intra_mode_info
                                             .filter_intra_mode];
        else
          intra_dir = get_intra_mode(mbmi, AOM_PLANE_Y);
        int tx_type_cost = 0;
        if (eob != 1) {
#if CONFIG_INTRA_TX_IST_PARSE
          const TxSetType tx_set_type =
              av1_get_ext_tx_set_type(tx_size, is_inter, reduced_tx_set_used);
          const int size_info = av1_size_class[tx_size];
          const int tx_type_idx = av1_tx_type_to_idx(
              get_primary_tx_type(tx_type), tx_set_type, intra_dir, size_info);
          tx_type_cost +=
              x->mode_costs
                  .intra_tx_type_costs[ext_tx_set][square_tx_size][tx_type_idx];
#else
          TX_TYPE primary_tx_type = get_primary_tx_type(tx_type);
          tx_type_cost +=
              x->mode_costs.intra_tx_type_costs[ext_tx_set][square_tx_size]
                                               [intra_dir][primary_tx_type];
#endif  // CONFIG_INTRA_TX_IST_PARSE
        } else {
          return tx_type_cost;
        }
        if (block_signals_sec_tx_type(xd, tx_size, tx_type, eob) &&
            xd->enable_ist) {
#if CONFIG_INTER_IST
          tx_type_cost +=
              x->mode_costs.stx_flag_cost[is_inter][square_tx_size]
                                         [get_secondary_tx_type(tx_type)];
#else
          tx_type_cost +=
              x->mode_costs.stx_flag_cost[square_tx_size]
                                         [get_secondary_tx_type(tx_type)];
#endif  // CONFIG_INTER_IST
#if CONFIG_IST_SET_FLAG
          if (get_secondary_tx_type(tx_type) > 0)
            tx_type_cost += get_sec_tx_set_cost(x, mbmi, tx_type);
#endif  // CONFIG_IST_SET_FLAG
        }
        return tx_type_cost;
      }
    }
#if CONFIG_INTER_IST
  } else if (!xd->lossless[xd->mi[0]->segment_id]) {
#else
  } else if (!is_inter && !xd->lossless[xd->mi[0]->segment_id]) {
#endif  // CONFIG_INTER_IST
    if (block_signals_sec_tx_type(xd, tx_size, tx_type, eob) &&
        xd->enable_ist) {
#if CONFIG_INTER_IST
      int tx_type_cost =
          x->mode_costs.stx_flag_cost[is_inter][square_tx_size]
                                     [get_secondary_tx_type(tx_type)];
#else
      int tx_type_cost =
          x->mode_costs
              .stx_flag_cost[square_tx_size][get_secondary_tx_type(tx_type)];
#endif  // CONFIG_INTER_IST
#if CONFIG_IST_SET_FLAG
#if CONFIG_INTER_IST
      if (get_secondary_tx_type(tx_type) > 0 && !is_inter)
        tx_type_cost += get_sec_tx_set_cost(x, mbmi, tx_type);
#else
      if (get_secondary_tx_type(tx_type) > 0)
        tx_type_cost += get_sec_tx_set_cost(x, mbmi, tx_type);
#endif  // CONFIG_INTER_IST
#endif  // CONFIG_IST_SET_FLAG
      return tx_type_cost;
    }
  }
  return 0;
}

static AOM_FORCE_INLINE int get_dqv(const int32_t *dequant, int coeff_idx,
                                    const qm_val_t *iqmatrix) {
  int dqv = dequant[!!coeff_idx];
  if (iqmatrix != NULL)
    dqv =
        ((iqmatrix[coeff_idx] * dqv) + (1 << (AOM_QM_BITS - 1))) >> AOM_QM_BITS;
  return dqv;
}

int av1_dep_quant(const struct AV1_COMP *cpi, MACROBLOCK *x, int plane,
                  int block, TX_SIZE tx_size, TX_TYPE tx_type,
                  CctxType cctx_type, const TXB_CTX *const txb_ctx,
                  int *rate_cost, int sharpness) {
  MACROBLOCKD *xd = &x->e_mbd;
  const struct macroblock_plane *p = &x->plane[plane];

  const SCAN_ORDER *scan_order =
      get_scan(tx_size, get_primary_tx_type(tx_type));

  const int16_t *scan = scan_order->scan;
  const int shift = av1_get_tx_scale(tx_size);
  int eob = p->eobs[block];

  const int32_t *dequant = p->dequant_QTX;
  const int32_t *quant = p->quant_fp_QTX;  // quant_QTX

  const qm_val_t *iqmatrix =
      av1_get_iqmatrix(&cpi->common.quant_params, xd, plane, tx_size, tx_type);
  const int block_offset = BLOCK_OFFSET(block);
  tran_low_t *qcoeff = p->qcoeff + block_offset;
  tran_low_t *dqcoeff = p->dqcoeff + block_offset;
  const tran_low_t *tcoeff = p->coeff + block_offset;
  const CoeffCosts *coeff_costs = &x->coeff_costs;

  // This function is not called if eob = 0.
  assert(eob > 0);

  const AV1_COMMON *cm = &cpi->common;
  const PLANE_TYPE plane_type = get_plane_type(plane);
  const TX_SIZE txs_ctx = get_txsize_entropy_ctx(tx_size);

  const TX_CLASS tx_class = tx_type_to_class[get_primary_tx_type(tx_type)];

  const MB_MODE_INFO *mbmi = xd->mi[0];
  const int bwl = get_txb_bwl(tx_size);
  const int width = get_txb_wide(tx_size);
  const int height = get_txb_high(tx_size);
  assert(width == (1 << bwl));

  const int is_inter = is_inter_block(mbmi, xd->tree_type);
  const int bob_code = p->bobs[block];
  const int is_fsc = (xd->mi[0]->fsc_mode[xd->tree_type == CHROMA_PART] &&
                      plane == PLANE_TYPE_Y) ||
                     use_inter_fsc(&cpi->common, plane, tx_type, is_inter);
  const LV_MAP_COEFF_COST *txb_costs =
      &coeff_costs->coeff_costs[txs_ctx][plane_type];
  const int eob_multi_size = txsize_log2_minus4[tx_size];
  const LV_MAP_EOB_COST *txb_eob_costs =
      &coeff_costs->eob_costs[eob_multi_size][plane_type];

  const int rshift =
      (sharpness +
       (cpi->oxcf.q_cfg.aq_mode == VARIANCE_AQ && mbmi->segment_id < 4
            ? 7 - mbmi->segment_id
            : 2) +
       (cpi->oxcf.q_cfg.aq_mode != VARIANCE_AQ &&
                cpi->oxcf.q_cfg.deltaq_mode == DELTA_Q_PERCEPTUAL &&
                cm->delta_q_info.delta_q_present_flag && x->sb_energy_level < 0
            ? (3 - x->sb_energy_level)
            : 0));
  int64_t rdmult = (((int64_t)x->rdmult * (plane_rd_mult[is_inter][plane_type]
                                           << (2 * (xd->bd - 8)))) +
                    2) >>
                   rshift;

  // getting context from previous level buf, updating levels on current level
  // buf. initialization all value by 0, since we update every position.
  int bufsize = (width + 4) * (height + 4) + TX_PAD_END;
  int mem_tcq_sz = sizeof(uint8_t) * bufsize * 2 * TOTALSTATES;
  uint8_t *mem_tcq = (uint8_t *)malloc(mem_tcq_sz);
  if (!mem_tcq) {
    exit(1);
  }
  if (eob > 1) {
    memset(mem_tcq, 0, mem_tcq_sz);
  }

  int si = eob - 1;
  // populate trellis
  assert(si < MAX_TRELLIS);
  DECISION trellis[MAX_TRELLIS][TOTALSTATES];

  int first_test_pos = si;
  for (int scan_pos = first_test_pos; scan_pos >= 0; scan_pos--) {
    uint8_t *levels[TOTALSTATES];
    uint8_t *prev_levels[TOTALSTATES];
    for (int i = 0; i < TOTALSTATES; i++) {
      if (scan_pos & 1) {
        levels[i] = &mem_tcq[(TOTALSTATES + i) * bufsize];
        prev_levels[i] = &mem_tcq[i * bufsize];
      } else {
        levels[i] = &mem_tcq[i * bufsize];
        prev_levels[i] = &mem_tcq[(TOTALSTATES + i) * bufsize];
      }
    }

    int blk_pos = scan[scan_pos];
    DECISION *decision = trellis[scan_pos];

    PQData pqData[4];
    int tempdqv = get_dqv(dequant, scan[scan_pos], iqmatrix);
    // preQuant tcoeff qcoeff dqcoeff shift dequant, scan[si], iqmatrix

    pre_quant(tcoeff[blk_pos], pqData, quant, tempdqv, shift + 1, scan_pos);
    // pre_quant(tcoeff[blk_pos], pqData, quant, tempdqv, shift, scan_pos);

    // init state
    for (int state = 0; state < TOTALSTATES; state++) {
      decision[state].rdCost = INT64_MAX >> 1;
      decision[state].dist = INT64_MAX >> 10;
      decision[state].rate = INT32_MAX >> 10;
      decision[state].absLevel = -1;
      decision[state].prevId = -2;
      decision[state].dqc = INT32_MAX;
    }
    const int coeff_sign = tcoeff[blk_pos] < 0;

    const int row = blk_pos >> bwl;
    const int col = blk_pos - (row << bwl);
    int limits = get_lf_limits(row, col, tx_class, plane);

    // calculate rate distortion
    bool is_first_pos = (scan_pos == first_test_pos);
    if (is_first_pos) {
      // try to quantize first coeff to nzcoeff
      int coeff_ctx = get_lower_levels_ctx_eob(bwl, height, si);
#if CONFIG_EOB_POS_LUMA
      const int eob_rate =
          get_eob_cost(si + 1, txb_eob_costs, txb_costs, is_inter);
#else
      const int eob_rate = get_eob_cost(si + 1, txb_eob_costs, txb_costs);
#endif  // CONFIG_EOB_POS_LUMA

      int rate_Q0_a =
          get_coeff_cost_eob(blk_pos, pqData[0].absLevel, (qcoeff[blk_pos] < 0),
                             coeff_ctx, txb_ctx->dc_sign_ctx, txb_costs, bwl,
                             tx_class
#if CONFIG_CONTEXT_DERIVATION
                             ,
                             xd->tmp_sign
#endif  // CONFIG_CONTEXT_DERIVATION
                             ,
                             plane) +
          eob_rate;
      int rate_Q0_b =
          get_coeff_cost_eob(blk_pos, pqData[2].absLevel, (qcoeff[blk_pos] < 0),
                             coeff_ctx, txb_ctx->dc_sign_ctx, txb_costs, bwl,
                             tx_class
#if CONFIG_CONTEXT_DERIVATION
                             ,
                             xd->tmp_sign
#endif  // CONFIG_CONTEXT_DERIVATION
                             ,
                             plane) +
          eob_rate;
#if MORESTATES
      decide(pqData[0].deltaDist, pqData[2].deltaDist, INT32_MAX >> 1, rdmult,
             rate_Q0_a, rate_Q0_b, INT32_MAX >> 1, &pqData[0], &pqData[2],
             limits, -1, &decision[0], &decision[4]);
#else
      decide(pqData[0].deltaDist, pqData[2].deltaDist, INT32_MAX >> 1, rdmult,
             rate_Q0_a, rate_Q0_b, INT32_MAX >> 1, &pqData[0], &pqData[2],
             limits, -1, &decision[0], &decision[2]);
#endif
    } else {
      int coeff_ctx0 = 0;
      int coeff_ctx1 = 0;
      int coeff_ctx2 = 0;
      int coeff_ctx3 = 0;
#if MORESTATES
      int coeff_ctx4 = 0;
      int coeff_ctx5 = 0;
      int coeff_ctx6 = 0;
      int coeff_ctx7 = 0;
#endif
      if (limits) {
        coeff_ctx0 =
            get_lower_levels_lf_ctx(prev_levels[0], blk_pos, bwl, tx_class);
        coeff_ctx1 =
            get_lower_levels_lf_ctx(prev_levels[1], blk_pos, bwl, tx_class);
        coeff_ctx2 =
            get_lower_levels_lf_ctx(prev_levels[2], blk_pos, bwl, tx_class);
        coeff_ctx3 =
            get_lower_levels_lf_ctx(prev_levels[3], blk_pos, bwl, tx_class);
#if MORESTATES
        coeff_ctx4 =
            get_lower_levels_lf_ctx(prev_levels[4], blk_pos, bwl, tx_class);
        coeff_ctx5 =
            get_lower_levels_lf_ctx(prev_levels[5], blk_pos, bwl, tx_class);
        coeff_ctx6 =
            get_lower_levels_lf_ctx(prev_levels[6], blk_pos, bwl, tx_class);
        coeff_ctx7 =
            get_lower_levels_lf_ctx(prev_levels[7], blk_pos, bwl, tx_class);

#endif
      } else {
        coeff_ctx0 = get_lower_levels_ctx(prev_levels[0], blk_pos, bwl, tx_class
#if CONFIG_CHROMA_TX_COEFF_CODING
                                          ,
                                          plane
#endif  // CONFIG_CHROMA_TX_COEFF_CODING
        );
        coeff_ctx1 = get_lower_levels_ctx(prev_levels[1], blk_pos, bwl, tx_class
#if CONFIG_CHROMA_TX_COEFF_CODING
                                          ,
                                          plane
#endif  // CONFIG_CHROMA_TX_COEFF_CODING
        );
        coeff_ctx2 = get_lower_levels_ctx(prev_levels[2], blk_pos, bwl, tx_class
#if CONFIG_CHROMA_TX_COEFF_CODING
                                          ,
                                          plane
#endif  // CONFIG_CHROMA_TX_COEFF_CODING
        );
        coeff_ctx3 = get_lower_levels_ctx(prev_levels[3], blk_pos, bwl, tx_class
#if CONFIG_CHROMA_TX_COEFF_CODING
                                          ,
                                          plane
#endif  // CONFIG_CHROMA_TX_COEFF_CODING
        );
#if MORESTATES
        coeff_ctx4 = get_lower_levels_ctx(prev_levels[4], blk_pos, bwl, tx_class
#if CONFIG_CHROMA_TX_COEFF_CODING
                                          ,
                                          plane
#endif  // CONFIG_CHROMA_TX_COEFF_CODING
        );
        coeff_ctx5 = get_lower_levels_ctx(prev_levels[5], blk_pos, bwl, tx_class
#if CONFIG_CHROMA_TX_COEFF_CODING
                                          ,
                                          plane
#endif  // CONFIG_CHROMA_TX_COEFF_CODING
        );
        coeff_ctx6 = get_lower_levels_ctx(prev_levels[6], blk_pos, bwl, tx_class
#if CONFIG_CHROMA_TX_COEFF_CODING
                                          ,
                                          plane
#endif  // CONFIG_CHROMA_TX_COEFF_CODING
        );
        coeff_ctx7 = get_lower_levels_ctx(prev_levels[7], blk_pos, bwl, tx_class
#if CONFIG_CHROMA_TX_COEFF_CODING
                                          ,
                                          plane
#endif  // CONFIG_CHROMA_TX_COEFF_CODING
        );
#endif
      }
      // calculate RDcost

      int rate_zero_0 = 0;
      int rate_zero_1 = 0;
      int rate_zero_2 = 0;
      int rate_zero_3 = 0;
#if MORESTATES
      int rate_zero_4 = 0;
      int rate_zero_5 = 0;
      int rate_zero_6 = 0;
      int rate_zero_7 = 0;

#endif
      if (limits) {
#if CONFIG_LCCHROMA
        const int(*base_lf_cost_ptr)[DQ_CTXS][LF_BASE_SYMBOLS * 2] =
            plane > 0 ? txb_costs->base_lf_cost_uv : txb_costs->base_lf_cost;
        rate_zero_0 = base_lf_cost_ptr[coeff_ctx0][0][0];
        rate_zero_1 = base_lf_cost_ptr[coeff_ctx1][0][0];
        rate_zero_2 = base_lf_cost_ptr[coeff_ctx2][1][0];
        rate_zero_3 = base_lf_cost_ptr[coeff_ctx3][1][0];
#if MORESTATES
        rate_zero_4 = base_lf_cost_ptr[coeff_ctx4][0][0];
        rate_zero_5 = base_lf_cost_ptr[coeff_ctx5][0][0];
        rate_zero_6 = base_lf_cost_ptr[coeff_ctx6][1][0];
        rate_zero_7 = base_lf_cost_ptr[coeff_ctx7][1][0];
#endif
#else
        rate_zero_0 = txb_costs->base_lf_cost[coeff_ctx0][0];
        rate_zero_1 = txb_costs->base_lf_cost[coeff_ctx1][0];
        rate_zero_2 = txb_costs->base_lf_cost_tcq[coeff_ctx2][0];
        rate_zero_3 = txb_costs->base_lf_cost_tcq[coeff_ctx3][0];
#if MORESTATES
        rate_zero_4 = txb_costs->base_lf_cost[coeff_ctx4][0];
        rate_zero_5 = txb_costs->base_lf_cost[coeff_ctx5][0];
        rate_zero_6 = txb_costs->base_lf_cost_tcq[coeff_ctx6][0];
        rate_zero_7 = txb_costs->base_lf_cost_tcq[coeff_ctx7][0];
#endif
#endif

      } else {
#if CONFIG_LCCHROMA
        const int(*base_cost_ptr)[DQ_CTXS][8] =
            plane > 0 ? txb_costs->base_cost_uv : txb_costs->base_cost;
        rate_zero_0 = base_cost_ptr[coeff_ctx0][0][0];
        rate_zero_1 = base_cost_ptr[coeff_ctx1][0][0];
        rate_zero_2 = base_cost_ptr[coeff_ctx2][1][0];
        rate_zero_3 = base_cost_ptr[coeff_ctx3][1][0];
#if MORESTATES
        rate_zero_4 = base_cost_ptr[coeff_ctx4][0][0];
        rate_zero_5 = base_cost_ptr[coeff_ctx5][0][0];
        rate_zero_6 = base_cost_ptr[coeff_ctx6][1][0];
        rate_zero_7 = base_cost_ptr[coeff_ctx7][1][0];
#endif
#else
        rate_zero_0 = txb_costs->base_cost[coeff_ctx0][0];
        rate_zero_1 = txb_costs->base_cost[coeff_ctx1][0];
        rate_zero_2 = txb_costs->base_cost_tcq[coeff_ctx2][0];
        rate_zero_3 = txb_costs->base_cost_tcq[coeff_ctx3][0];
#if MORESTATES
        rate_zero_4 = txb_costs->base_cost[coeff_ctx4][0];
        rate_zero_5 = txb_costs->base_cost[coeff_ctx5][0];
        rate_zero_6 = txb_costs->base_cost_tcq[coeff_ctx6][0];
        rate_zero_7 = txb_costs->base_cost_tcq[coeff_ctx7][0];
#endif
#endif
      }

      DECISION *prd = trellis[scan_pos + 1];
      int rate_Q0_a_prd0 = get_coeff_cost_general(
          0, blk_pos, pqData[0].absLevel, coeff_sign, coeff_ctx0,
          txb_ctx->dc_sign_ctx, txb_costs, bwl, tx_class, prev_levels[0]
#if CONFIG_CONTEXT_DERIVATION
          ,
          xd->tmp_sign
#endif
          ,
          plane, limits);
      int rate_Q0_a_prd1 = get_coeff_cost_general(
          0, blk_pos, pqData[0].absLevel, coeff_sign, coeff_ctx1,
          txb_ctx->dc_sign_ctx, txb_costs, bwl, tx_class, prev_levels[1]
#if CONFIG_CONTEXT_DERIVATION
          ,
          xd->tmp_sign
#endif
          ,
          plane, limits);
      int rate_Q0_b_prd0 = get_coeff_cost_general(
          0, blk_pos, pqData[2].absLevel, coeff_sign, coeff_ctx0,
          txb_ctx->dc_sign_ctx, txb_costs, bwl, tx_class, prev_levels[0]
#if CONFIG_CONTEXT_DERIVATION
          ,
          xd->tmp_sign
#endif
          ,
          plane, limits);
      int rate_Q0_b_prd1 = get_coeff_cost_general(
          0, blk_pos, pqData[2].absLevel, coeff_sign, coeff_ctx1,
          txb_ctx->dc_sign_ctx, txb_costs, bwl, tx_class, prev_levels[1]
#if CONFIG_CONTEXT_DERIVATION
          ,
          xd->tmp_sign
#endif
          ,
          plane, limits);
      int rate_Q1_a_prd2 = get_coeff_cost_general_q1(
          0, blk_pos, pqData[1].absLevel, coeff_sign, coeff_ctx2,
          txb_ctx->dc_sign_ctx, txb_costs, bwl, tx_class, prev_levels[2]
#if CONFIG_CONTEXT_DERIVATION
          ,
          xd->tmp_sign
#endif
          ,
          plane, limits);
      int rate_Q1_a_prd3 = get_coeff_cost_general_q1(
          0, blk_pos, pqData[1].absLevel, coeff_sign, coeff_ctx3,
          txb_ctx->dc_sign_ctx, txb_costs, bwl, tx_class, prev_levels[3]
#if CONFIG_CONTEXT_DERIVATION
          ,
          xd->tmp_sign
#endif
          ,
          plane, limits);
      int rate_Q1_b_prd2 = get_coeff_cost_general_q1(
          0, blk_pos, pqData[3].absLevel, coeff_sign, coeff_ctx2,
          txb_ctx->dc_sign_ctx, txb_costs, bwl, tx_class, prev_levels[2]
#if CONFIG_CONTEXT_DERIVATION
          ,
          xd->tmp_sign
#endif
          ,
          plane, limits);
      int rate_Q1_b_prd3 = get_coeff_cost_general_q1(
          0, blk_pos, pqData[3].absLevel, coeff_sign, coeff_ctx3,
          txb_ctx->dc_sign_ctx, txb_costs, bwl, tx_class, prev_levels[3]
#if CONFIG_CONTEXT_DERIVATION
          ,
          xd->tmp_sign
#endif
          ,
          plane, limits);
#if MORESTATES
      int rate_Q0_a_prd4 = get_coeff_cost_general(
          0, blk_pos, pqData[0].absLevel, coeff_sign, coeff_ctx4,
          txb_ctx->dc_sign_ctx, txb_costs, bwl, tx_class, prev_levels[4]
#if CONFIG_CONTEXT_DERIVATION
          ,
          xd->tmp_sign
#endif
          ,
          plane, limits);
      int rate_Q0_a_prd5 = get_coeff_cost_general(
          0, blk_pos, pqData[0].absLevel, coeff_sign, coeff_ctx5,
          txb_ctx->dc_sign_ctx, txb_costs, bwl, tx_class, prev_levels[5]
#if CONFIG_CONTEXT_DERIVATION
          ,
          xd->tmp_sign
#endif
          ,
          plane, limits);
      int rate_Q0_b_prd4 = get_coeff_cost_general(
          0, blk_pos, pqData[2].absLevel, coeff_sign, coeff_ctx4,
          txb_ctx->dc_sign_ctx, txb_costs, bwl, tx_class, prev_levels[4]
#if CONFIG_CONTEXT_DERIVATION
          ,
          xd->tmp_sign
#endif
          ,
          plane, limits);
      int rate_Q0_b_prd5 = get_coeff_cost_general(
          0, blk_pos, pqData[2].absLevel, coeff_sign, coeff_ctx5,
          txb_ctx->dc_sign_ctx, txb_costs, bwl, tx_class, prev_levels[5]
#if CONFIG_CONTEXT_DERIVATION
          ,
          xd->tmp_sign
#endif
          ,
          plane, limits);
      int rate_Q1_a_prd6 = get_coeff_cost_general_q1(
          0, blk_pos, pqData[1].absLevel, coeff_sign, coeff_ctx6,
          txb_ctx->dc_sign_ctx, txb_costs, bwl, tx_class, prev_levels[6]
#if CONFIG_CONTEXT_DERIVATION
          ,
          xd->tmp_sign
#endif
          ,
          plane, limits);
      int rate_Q1_a_prd7 = get_coeff_cost_general_q1(
          0, blk_pos, pqData[1].absLevel, coeff_sign, coeff_ctx7,
          txb_ctx->dc_sign_ctx, txb_costs, bwl, tx_class, prev_levels[7]
#if CONFIG_CONTEXT_DERIVATION
          ,
          xd->tmp_sign
#endif
          ,
          plane, limits);
      int rate_Q1_b_prd6 = get_coeff_cost_general_q1(
          0, blk_pos, pqData[3].absLevel, coeff_sign, coeff_ctx6,
          txb_ctx->dc_sign_ctx, txb_costs, bwl, tx_class, prev_levels[6]
#if CONFIG_CONTEXT_DERIVATION
          ,
          xd->tmp_sign
#endif
          ,
          plane, limits);
      int rate_Q1_b_prd7 = get_coeff_cost_general_q1(
          0, blk_pos, pqData[3].absLevel, coeff_sign, coeff_ctx7,
          txb_ctx->dc_sign_ctx, txb_costs, bwl, tx_class, prev_levels[7]
#if CONFIG_CONTEXT_DERIVATION
          ,
          xd->tmp_sign
#endif
          ,
          plane, limits);
#endif
      rate_Q0_a_prd0 += prd[0].rate;
      rate_Q0_a_prd1 += prd[1].rate;
      rate_Q0_b_prd0 += prd[0].rate;
      rate_Q0_b_prd1 += prd[1].rate;
      rate_Q1_a_prd2 += prd[2].rate;
      rate_Q1_a_prd3 += prd[3].rate;
      rate_Q1_b_prd2 += prd[2].rate;
      rate_Q1_b_prd3 += prd[3].rate;
#if MORESTATES
      rate_Q0_a_prd4 += prd[4].rate;
      rate_Q0_a_prd5 += prd[5].rate;
      rate_Q0_b_prd4 += prd[4].rate;
      rate_Q0_b_prd5 += prd[5].rate;
      rate_Q1_a_prd6 += prd[6].rate;
      rate_Q1_a_prd7 += prd[7].rate;
      rate_Q1_b_prd6 += prd[6].rate;
      rate_Q1_b_prd7 += prd[7].rate;

#endif
      {
        int64_t dist_Q0_a_prd0 = pqData[0].deltaDist + prd[0].dist;
        int64_t dist_Q0_a_prd1 = pqData[0].deltaDist + prd[1].dist;
        int64_t dist_Q0_b_prd0 = pqData[2].deltaDist + prd[0].dist;
        int64_t dist_Q0_b_prd1 = pqData[2].deltaDist + prd[1].dist;
        int64_t dist_Q1_a_prd2 = pqData[1].deltaDist + prd[2].dist;
        int64_t dist_Q1_a_prd3 = pqData[1].deltaDist + prd[3].dist;
        int64_t dist_Q1_b_prd2 = pqData[3].deltaDist + prd[2].dist;
        int64_t dist_Q1_b_prd3 = pqData[3].deltaDist + prd[3].dist;
#if MORESTATES
        int64_t dist_Q0_a_prd4 = pqData[0].deltaDist + prd[4].dist;
        int64_t dist_Q0_a_prd5 = pqData[0].deltaDist + prd[5].dist;
        int64_t dist_Q0_b_prd4 = pqData[2].deltaDist + prd[4].dist;
        int64_t dist_Q0_b_prd5 = pqData[2].deltaDist + prd[5].dist;
        int64_t dist_Q1_a_prd6 = pqData[1].deltaDist + prd[6].dist;
        int64_t dist_Q1_a_prd7 = pqData[1].deltaDist + prd[7].dist;
        int64_t dist_Q1_b_prd6 = pqData[3].deltaDist + prd[6].dist;
        int64_t dist_Q1_b_prd7 = pqData[3].deltaDist + prd[7].dist;

#endif
        // todo: Q0 can skip the sig_flag or skip some another flag. This is not
        // included in the calculation of RDcost now.
#if MORESTATES
        // pre_state is 0
        decide(dist_Q0_a_prd0, dist_Q0_b_prd0, prd[0].dist, rdmult,
               rate_Q0_a_prd0, rate_Q0_b_prd0, rate_zero_0 + prd[0].rate,
               &pqData[0], &pqData[2], limits, 0, &decision[0], &decision[4]);
        // pre_state is 1
        decide(dist_Q0_a_prd1, dist_Q0_b_prd1, prd[1].dist, rdmult,
               rate_Q0_a_prd1, rate_Q0_b_prd1, rate_zero_1 + prd[1].rate,
               &pqData[0], &pqData[2], limits, 1, &decision[4], &decision[0]);
        // pre_state is 2
        decide(dist_Q1_a_prd2, dist_Q1_b_prd2, prd[2].dist, rdmult,
               rate_Q1_a_prd2, rate_Q1_b_prd2, rate_zero_2 + prd[2].rate,
               &pqData[1], &pqData[3], limits, 2, &decision[1], &decision[5]);
        // pre_state is 3
        decide(dist_Q1_a_prd3, dist_Q1_b_prd3, prd[3].dist, rdmult,
               rate_Q1_a_prd3, rate_Q1_b_prd3, rate_zero_3 + prd[3].rate,
               &pqData[1], &pqData[3], limits, 3, &decision[5], &decision[1]);
        // pre_state is 4
        decide(dist_Q0_a_prd4, dist_Q0_b_prd4, prd[4].dist, rdmult,
               rate_Q0_a_prd4, rate_Q0_b_prd4, rate_zero_4 + prd[4].rate,
               &pqData[0], &pqData[2], limits, 4, &decision[6], &decision[2]);
        // pre_state is 5
        decide(dist_Q0_a_prd5, dist_Q0_b_prd5, prd[5].dist, rdmult,
               rate_Q0_a_prd5, rate_Q0_b_prd5, rate_zero_5 + prd[5].rate,
               &pqData[0], &pqData[2], limits, 5, &decision[2], &decision[6]);
        // pre_state is 6
        decide(dist_Q1_a_prd6, dist_Q1_b_prd6, prd[6].dist, rdmult,
               rate_Q1_a_prd6, rate_Q1_b_prd6, rate_zero_6 + prd[6].rate,
               &pqData[1], &pqData[3], limits, 6, &decision[7], &decision[3]);
        // pre_state is 7
        decide(dist_Q1_a_prd7, dist_Q1_b_prd7, prd[7].dist, rdmult,
               rate_Q1_a_prd7, rate_Q1_b_prd7, rate_zero_7 + prd[7].rate,
               &pqData[1], &pqData[3], limits, 7, &decision[3], &decision[7]);

#else
        // pre_state is 0
        decide(dist_Q0_a_prd0, dist_Q0_b_prd0, prd[0].dist, rdmult,
               rate_Q0_a_prd0, rate_Q0_b_prd0, rate_zero_0 + prd[0].rate,
               &pqData[0], &pqData[2], limits, 0, &decision[0], &decision[2]);
        // pre_state is 1
        decide(dist_Q0_a_prd1, dist_Q0_b_prd1, prd[1].dist, rdmult,
               rate_Q0_a_prd1, rate_Q0_b_prd1, rate_zero_1 + prd[1].rate,
               &pqData[0], &pqData[2], limits, 1, &decision[2], &decision[0]);
        // pre_state is 2
        decide(dist_Q1_a_prd2, dist_Q1_b_prd2, prd[2].dist, rdmult,
               rate_Q1_a_prd2, rate_Q1_b_prd2, rate_zero_2 + prd[2].rate,
               &pqData[1], &pqData[3], limits, 2, &decision[1], &decision[3]);
        // pre_state is 3
        decide(dist_Q1_a_prd3, dist_Q1_b_prd3, prd[3].dist, rdmult,
               rate_Q1_a_prd3, rate_Q1_b_prd3, rate_zero_3 + prd[3].rate,
               &pqData[1], &pqData[3], limits, 3, &decision[3], &decision[1]);
#endif
        // assume current state is 0, current coeff is new eob.
        // input: scan_pos,pqData[0],pqData[2], decison[0] and decision[2]
        //  update eob if better use current position as eob

        if (sharpness == 0) {
#if CONFIG_EOB_POS_LUMA
          const int new_eob_rate =
              get_eob_cost(scan_pos + 1, txb_eob_costs, txb_costs, is_inter);
#else
          const int new_eob_rate =
              get_eob_cost(scan_pos + 1, txb_eob_costs, txb_costs);
#endif  // CONFIG_EOB_POS_LUMA
          int new_eob_ctx = get_lower_levels_ctx_eob(bwl, height, scan_pos);
          int rate_Q0_a =
              get_coeff_cost_eob(blk_pos, pqData[0].absLevel,
                                 (qcoeff[blk_pos] < 0), new_eob_ctx,
                                 txb_ctx->dc_sign_ctx, txb_costs, bwl, tx_class
#if CONFIG_CONTEXT_DERIVATION
                                 ,
                                 xd->tmp_sign
#endif  // CONFIG_CONTEXT_DERIVATION
                                 ,
                                 plane) +
              new_eob_rate;
          int rate_Q0_b =
              get_coeff_cost_eob(blk_pos, pqData[2].absLevel,
                                 (qcoeff[blk_pos] < 0), new_eob_ctx,
                                 txb_ctx->dc_sign_ctx, txb_costs, bwl, tx_class
#if CONFIG_CONTEXT_DERIVATION
                                 ,
                                 xd->tmp_sign
#endif  // CONFIG_CONTEXT_DERIVATION
                                 ,
                                 plane) +
              new_eob_rate;
#if MORESTATES
          decide(pqData[0].deltaDist, pqData[2].deltaDist, INT32_MAX >> 1,
                 rdmult, rate_Q0_a, rate_Q0_b, INT32_MAX >> 1, &pqData[0],
                 &pqData[2], limits, -1, &decision[0], &decision[4]);
#else
          decide(pqData[0].deltaDist, pqData[2].deltaDist, INT32_MAX >> 1,
                 rdmult, rate_Q0_a, rate_Q0_b, INT32_MAX >> 1, &pqData[0],
                 &pqData[2], limits, -1, &decision[0], &decision[2]);
#endif
        }
      }
    }

    // copy corresponding context from previous level buffer
    for (int state = 0; state < TOTALSTATES && scan_pos != si; state++) {
      if (decision[state].prevId >= 0)
        memcpy(levels[state], prev_levels[decision[state].prevId],
               sizeof(uint8_t) * bufsize);
    }
    // update levels_buf
    for (int state = 0; state < TOTALSTATES && scan_pos != 0; state++) {
      set_levels_buf(&decision[state], levels[state], scan, si, scan_pos, bwl,
                     sharpness);
    }
  }

  free(mem_tcq);

  // find best path
  int64_t min_path_cost = INT64_MAX;
  int min_rate = INT32_MAX;
  int64_t min_dist = INT32_MAX;
  DECISION decision;
  decision.prevId = -2;
  for (int state = 0; state < TOTALSTATES; state++) {
    if (trellis[0][state].rdCost < min_path_cost) {
      decision.prevId = state;
      min_path_cost = trellis[0][state].rdCost;
      min_rate = trellis[0][state].rate;
      min_dist = trellis[0][state].dist;
    }
  }
  // backward scannig  dqc,tqc,qc,level
  int scan_pos = 0;
  // printf("quantization... \n");
  for (; decision.prevId >= 0; scan_pos++) {
    decision = trellis[scan_pos][decision.prevId];
    // printf("%d ", decision.absLevel);
    int blk_pos = scan[scan_pos];
    qcoeff[blk_pos] =
        (tcoeff[blk_pos] < 0 ? -decision.absLevel : decision.absLevel);
    dqcoeff[blk_pos] = (tcoeff[blk_pos] < 0 ? -decision.dqc : decision.dqc);
  }

  eob = scan_pos;
  for (; scan_pos <= first_test_pos; scan_pos++) {
    int blk_pos = scan[scan_pos];
    qcoeff[blk_pos] = 0;
    dqcoeff[blk_pos] = 0;
  }

#if CONFIG_CONTEXT_DERIVATION
  int txb_skip_ctx = txb_ctx->txb_skip_ctx;
  int non_skip_cost = 0;
  int skip_cost = 0;
  if (plane == AOM_PLANE_V) {
    txb_skip_ctx +=
        (x->plane[AOM_PLANE_U].eobs[block] ? V_TXB_SKIP_CONTEXT_OFFSET : 0);
    non_skip_cost = txb_costs->v_txb_skip_cost[txb_skip_ctx][0];
    skip_cost = txb_costs->v_txb_skip_cost[txb_skip_ctx][1];
  } else {
#if CONFIG_TX_SKIP_FLAG_MODE_DEP_CTX
    const int pred_mode_ctx =
        (is_inter || mbmi->fsc_mode[xd->tree_type == CHROMA_PART]) ? 1 : 0;
    non_skip_cost = txb_costs->txb_skip_cost[pred_mode_ctx][txb_skip_ctx][0];
    skip_cost = txb_costs->txb_skip_cost[pred_mode_ctx][txb_skip_ctx][1];
#else
    non_skip_cost = txb_costs->txb_skip_cost[txb_skip_ctx][0];
    skip_cost = txb_costs->txb_skip_cost[txb_skip_ctx][1];
#endif  // CONFIG_TX_SKIP_FLAG_MODE_DEP_CTX
  }
#else
  const int non_skip_cost = txb_costs->txb_skip_cost[txb_ctx->txb_skip_ctx][0];
  const int skip_cost = txb_costs->txb_skip_cost[txb_ctx->txb_skip_ctx][1];
#endif  // CONFIG_CONTEXT_DERIVATION

  int accu_rate = 0;
  set_bob(x, plane, block, tx_size, tx_type);

  if (eob == 0)
    assert(0);  // in current implementation, this could not happen.
  else {
    const int tx_type_cost = get_tx_type_cost(x, xd, plane, tx_size, tx_type,
                                              cm->features.reduced_tx_set_used,
                                              eob, bob_code, is_fsc);
    accu_rate += non_skip_cost + tx_type_cost + min_rate;
    // skip block
    if (RDCOST(rdmult, accu_rate, min_dist) > RDCOST(rdmult, skip_cost, 0) &&
        sharpness == 0) {
      for (int scan_idx = 0; scan_idx <= first_test_pos; scan_idx++) {
        int blk_idx = scan[scan_idx];
        qcoeff[blk_idx] = 0;
        dqcoeff[blk_idx] = 0;
      }
      accu_rate = skip_cost;
      eob = 0;
    }
  }

  p->eobs[block] = eob;
  p->txb_entropy_ctx[block] =
      av1_get_txb_entropy_context(qcoeff, scan_order, p->eobs[block]);

  accu_rate += get_cctx_type_cost(cm, x, xd, plane, tx_size, block, cctx_type);

  *rate_cost = accu_rate;
  return eob;
}
