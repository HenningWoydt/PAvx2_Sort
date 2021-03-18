#ifndef QUICKSORT_PAVX2_H
#define QUICKSORT_PAVX2_H

#include <omp.h>
#include <immintrin.h>
#include <cmath>
#include <vector>
#include <deque>
#include <algorithm>

namespace PAvx2 {
    namespace _internal {

#define LOAD_VECTOR(arr) _mm256_loadu_si256(reinterpret_cast<__m256i *>(arr))
#define STORE_VECTOR(arr, vec) _mm256_storeu_si256(reinterpret_cast<__m256i *>(arr), vec)
#define CONST_VECTOR(c) _mm256_set1_epi32(c)

/* vectorized sorting networks
 * from Blacher https://github.com/simd-sorting/fast-and-robust/blob/master/avx2_sort_demo/avx2sort.h
************************************/

#define COEX(a, b){auto vec_tmp = a; a = _mm256_min_epi32(a, b); b = _mm256_max_epi32(vec_tmp, b);}

        /* shuffle 2 vectors, instruction for int is missing,
        * therefore shuffle with float */
#define SHUFFLE_2_VECS(a, b, mask) reinterpret_cast<__m256i>(_mm256_shuffle_ps(reinterpret_cast<__m256>(a), reinterpret_cast<__m256>(b), mask));

        /* optimized sorting network for two vectors, that is 16 ints */
        inline void sort_16(__m256i &v1, __m256i &v2) {
            COEX(v1, v2);                                  /* step 1 */

            v2 = _mm256_shuffle_epi32(v2, _MM_SHUFFLE(2, 3, 0, 1)); /* step 2 */
            COEX(v1, v2);

            auto tmp = v1;                                          /* step  3 */
            v1 = SHUFFLE_2_VECS(v1, v2, 0b10001000);
            v2 = SHUFFLE_2_VECS(tmp, v2, 0b11011101);
            COEX(v1, v2);

            v2 = _mm256_shuffle_epi32(v2, _MM_SHUFFLE(0, 1, 2, 3)); /* step  4 */
            COEX(v1, v2);

            tmp = v1;                                               /* step  5 */
            v1 = SHUFFLE_2_VECS(v1, v2, 0b01000100);
            v2 = SHUFFLE_2_VECS(tmp, v2, 0b11101110);
            COEX(v1, v2);

            tmp = v1;                                               /* step  6 */
            v1 = SHUFFLE_2_VECS(v1, v2, 0b11011000);
            v2 = SHUFFLE_2_VECS(tmp, v2, 0b10001101);
            COEX(v1, v2);

            v2 = _mm256_permutevar8x32_epi32(v2, _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0));
            COEX(v1, v2);                                           /* step  7 */

            tmp = v1;                                               /* step  8 */
            v1 = SHUFFLE_2_VECS(v1, v2, 0b11011000);
            v2 = SHUFFLE_2_VECS(tmp, v2, 0b10001101);
            COEX(v1, v2);

            tmp = v1;                                               /* step  9 */
            v1 = SHUFFLE_2_VECS(v1, v2, 0b11011000);
            v2 = SHUFFLE_2_VECS(tmp, v2, 0b10001101);
            COEX(v1, v2);

            /* permute to make it easier to restore order */
            v1 = _mm256_permutevar8x32_epi32(v1, _mm256_setr_epi32(0, 4, 1, 5, 6, 2, 7, 3));
            v2 = _mm256_permutevar8x32_epi32(v2, _mm256_setr_epi32(0, 4, 1, 5, 6, 2, 7, 3));

            tmp = v1;                                              /* step  10 */
            v1 = SHUFFLE_2_VECS(v1, v2, 0b10001000);
            v2 = SHUFFLE_2_VECS(tmp, v2, 0b11011101);
            COEX(v1, v2);

            /* restore order */
            auto b2 = _mm256_shuffle_epi32(v2, 0b10110001);
            auto b1 = _mm256_shuffle_epi32(v1, 0b10110001);
            v1 = _mm256_blend_epi32(v1, b2, 0b10101010);
            v2 = _mm256_blend_epi32(b1, v2, 0b10101010);
        }

#define ASC(a, b, c, d, e, f, g, h)                                    \
            (((h < 7) << 7) | ((g < 6) << 6) | ((f < 5) << 5) | ((e < 4) << 4) | \
            ((d < 3) << 3) | ((c < 2) << 2) | ((b < 1) << 1) | (a < 0))

#define COEX_PERMUTE(vec, a, b, c, d, e, f, g, h, MASK){               \
            __m256i permute_mask = _mm256_setr_epi32(a, b, c, d, e, f, g, h);  \
            __m256i permuted = _mm256_permutevar8x32_epi32(vec, permute_mask); \
            __m256i min = _mm256_min_epi32(permuted, vec);                     \
            __m256i max = _mm256_max_epi32(permuted, vec);                     \
            constexpr int blend_mask = MASK(a, b, c, d, e, f, g, h);           \
            vec = _mm256_blend_epi32(min, max, blend_mask);}

#define COEX_SHUFFLE(vec, a, b, c, d, e, f, g, h, MASK){               \
            constexpr int shuffle_mask = _MM_SHUFFLE(d, c, b, a);              \
            __m256i shuffled = _mm256_shuffle_epi32(vec, shuffle_mask);        \
            __m256i min = _mm256_min_epi32(shuffled, vec);                     \
            __m256i max = _mm256_max_epi32(shuffled, vec);                     \
            constexpr int blend_mask = MASK(a, b, c, d, e, f, g, h);           \
            vec = _mm256_blend_epi32(min, max, blend_mask);}

#define REVERSE_VEC(vec){                                              \
            vec = _mm256_permutevar8x32_epi32(                                 \
            vec, _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0));}

        /* sorting network for 8 int with compare-exchange macros
        * (used for pv selection in median of the medians) */
#define SORT_8(vec){                                                   \
            COEX_SHUFFLE(vec, 1, 0, 3, 2, 5, 4, 7, 6, ASC);                           \
            COEX_SHUFFLE(vec, 2, 3, 0, 1, 6, 7, 4, 5, ASC);                           \
            COEX_SHUFFLE(vec, 0, 2, 1, 3, 4, 6, 5, 7, ASC);                           \
            COEX_PERMUTE(vec, 7, 6, 5, 4, 3, 2, 1, 0, ASC);                           \
            COEX_SHUFFLE(vec, 2, 3, 0, 1, 6, 7, 4, 5, ASC);}

        /* merge N vectors with bitonic merge, N % 2 == 0 and N > 0
        * s = 2 means that two vectors are already sorted */
        inline void bitonic_merge_16(__m256i *vecs, int N, int s = 2) {
            for (int t = s * 2; t < 2 * N; t *= 2) {
                for (int l = 0; l < N; l += t) {
                    for (int j = std::max(l + t - N, 0); j < t / 2; j += 2) {
                        REVERSE_VEC(vecs[l + t - 1 - j]);
                        REVERSE_VEC(vecs[l + t - 2 - j]);
                        COEX(vecs[l + j], vecs[l + t - 1 - j]);
                        COEX(vecs[l + j + 1], vecs[l + t - 2 - j]);
                    }
                }
                for (int m = t / 2; m > 4; m /= 2) {
                    for (int k = 0; k < N - m / 2; k += m) {
                        int bound = std::min((k + m / 2), N - (m / 2));
                        for (int j = k; j < bound; j += 2) {
                            COEX(vecs[j], vecs[m / 2 + j]);
                            COEX(vecs[j + 1], vecs[m / 2 + j + 1]);
                        }
                    }
                }
                for (int j = 0; j < N - 2; j += 4) {
                    COEX(vecs[j], vecs[j + 2]);
                    COEX(vecs[j + 1], vecs[j + 3]);
                }
                for (int j = 0; j < N; j += 2) {
                    COEX(vecs[j], vecs[j + 1]);
                }
                for (int i = 0; i < N; i += 2) {
                    COEX_PERMUTE(vecs[i], 4, 5, 6, 7, 0, 1, 2, 3, ASC);
                    COEX_PERMUTE(vecs[i + 1], 4, 5, 6, 7, 0, 1, 2, 3, ASC);
                    auto tmp = vecs[i];
                    vecs[i] = _mm256_unpacklo_epi32(vecs[i], vecs[i + 1]);
                    vecs[i + 1] = _mm256_unpackhi_epi32(tmp, vecs[i + 1]);
                    COEX(vecs[i], vecs[i + 1]);
                    tmp = vecs[i];
                    vecs[i] = _mm256_unpacklo_epi32(vecs[i], vecs[i + 1]);
                    vecs[i + 1] = _mm256_unpackhi_epi32(tmp, vecs[i + 1]);
                    COEX(vecs[i], vecs[i + 1]);
                    tmp = vecs[i];
                    vecs[i] = _mm256_unpacklo_epi32(vecs[i], vecs[i + 1]);
                    vecs[i + 1] = _mm256_unpackhi_epi32(tmp, vecs[i + 1]);
                }
            }
        }

        inline void bitonic_merge_128(__m256i *vecs, int N, int s = 16) {
            int remainder16 = N - N % 16;
            int remainder8 = N - N % 8;
            for (int t = s * 2; t < 2 * N; t *= 2) {
                for (int l = 0; l < N; l += t) {
                    for (int j = std::max(l + t - N, 0); j < t / 2; j += 2) {
                        REVERSE_VEC(vecs[l + t - 1 - j]);
                        REVERSE_VEC(vecs[l + t - 2 - j]);
                        COEX(vecs[l + j], vecs[l + t - 1 - j]);
                        COEX(vecs[l + j + 1], vecs[l + t - 2 - j]);
                    }
                }
                for (int m = t / 2; m > 16; m /= 2) {
                    for (int k = 0; k < N - m / 2; k += m) {
                        int bound = std::min((k + m / 2), N - (m / 2));
                        for (int j = k; j < bound; j += 2) {
                            COEX(vecs[j], vecs[m / 2 + j]);
                            COEX(vecs[j + 1], vecs[m / 2 + j + 1]);
                        }
                    }
                }
                for (int j = 0; j < remainder16; j += 16) {
                    COEX(vecs[j], vecs[j + 8]);
                    COEX(vecs[j + 1], vecs[j + 9]);
                    COEX(vecs[j + 2], vecs[j + 10]);
                    COEX(vecs[j + 3], vecs[j + 11]);
                    COEX(vecs[j + 4], vecs[j + 12]);
                    COEX(vecs[j + 5], vecs[j + 13]);
                    COEX(vecs[j + 6], vecs[j + 14]);
                    COEX(vecs[j + 7], vecs[j + 15]);
                }
                for (int j = remainder16 + 8; j < N; j += 1) {
                    COEX(vecs[j - 8], vecs[j]);
                }
                for (int j = 0; j < remainder8; j += 8) {
                    COEX(vecs[j], vecs[j + 4]);
                    COEX(vecs[j + 1], vecs[j + 5]);
                    COEX(vecs[j + 2], vecs[j + 6]);
                    COEX(vecs[j + 3], vecs[j + 7]);
                }
                for (int j = remainder8 + 4; j < N; j += 1) {
                    COEX(vecs[j - 4], vecs[j]);
                }
                for (int j = 0; j < N - 2; j += 4) {
                    COEX(vecs[j], vecs[j + 2]);
                    COEX(vecs[j + 1], vecs[j + 3]);
                }
                for (int j = 0; j < N; j += 2) {
                    COEX(vecs[j], vecs[j + 1]);
                }
                for (int i = 0; i < N; i += 2) {
                    COEX_PERMUTE(vecs[i], 4, 5, 6, 7, 0, 1, 2, 3, ASC);
                    COEX_PERMUTE(vecs[i + 1], 4, 5, 6, 7, 0, 1, 2, 3, ASC);
                    auto tmp = vecs[i];
                    vecs[i] = _mm256_unpacklo_epi32(vecs[i], vecs[i + 1]);
                    vecs[i + 1] = _mm256_unpackhi_epi32(tmp, vecs[i + 1]);
                    COEX(vecs[i], vecs[i + 1]);
                    tmp = vecs[i];
                    vecs[i] = _mm256_unpacklo_epi32(vecs[i], vecs[i + 1]);
                    vecs[i + 1] = _mm256_unpackhi_epi32(tmp, vecs[i + 1]);
                    COEX(vecs[i], vecs[i + 1]);
                    tmp = vecs[i];
                    vecs[i] = _mm256_unpacklo_epi32(vecs[i], vecs[i + 1]);
                    vecs[i + 1] = _mm256_unpackhi_epi32(tmp, vecs[i + 1]);
                }
            }
        }

        /* sort 8 columns each containing 16 int, with 60 modules */
        inline void sort_16_int_vertical(__m256i *vecs) {
            COEX(vecs[0], vecs[1]);
            COEX(vecs[2], vecs[3]);  /* step 1 */
            COEX(vecs[4], vecs[5]);
            COEX(vecs[6], vecs[7]);
            COEX(vecs[8], vecs[9]);
            COEX(vecs[10], vecs[11])
            COEX(vecs[12], vecs[13]);
            COEX(vecs[14], vecs[15])
            COEX(vecs[0], vecs[2]);
            COEX(vecs[1], vecs[3]);  /* step 2 */
            COEX(vecs[4], vecs[6]);
            COEX(vecs[5], vecs[7]);
            COEX(vecs[8], vecs[10]);
            COEX(vecs[9], vecs[11]);
            COEX(vecs[12], vecs[14]);
            COEX(vecs[13], vecs[15]);
            COEX(vecs[0], vecs[4]);
            COEX(vecs[1], vecs[5]);  /* step 3 */
            COEX(vecs[2], vecs[6]);
            COEX(vecs[3], vecs[7]);
            COEX(vecs[8], vecs[12]);
            COEX(vecs[9], vecs[13]);
            COEX(vecs[10], vecs[14]);
            COEX(vecs[11], vecs[15]);
            COEX(vecs[0], vecs[8]);
            COEX(vecs[1], vecs[9])   /* step 4 */
            COEX(vecs[2], vecs[10]);
            COEX(vecs[3], vecs[11])
            COEX(vecs[4], vecs[12]);
            COEX(vecs[5], vecs[13])
            COEX(vecs[6], vecs[14]);
            COEX(vecs[7], vecs[15])
            COEX(vecs[5], vecs[10]);
            COEX(vecs[6], vecs[9]); /* step 5 */
            COEX(vecs[3], vecs[12]);
            COEX(vecs[7], vecs[11]);
            COEX(vecs[13], vecs[14]);
            COEX(vecs[4], vecs[8]);
            COEX(vecs[1], vecs[2]);
            COEX(vecs[1], vecs[4]);
            COEX(vecs[7], vecs[13]); /* step 6 */
            COEX(vecs[2], vecs[8]);
            COEX(vecs[11], vecs[14]);
            COEX(vecs[2], vecs[4]);
            COEX(vecs[5], vecs[6]);  /* step 7 */
            COEX(vecs[9], vecs[10]);
            COEX(vecs[11], vecs[13]);
            COEX(vecs[3], vecs[8]);
            COEX(vecs[7], vecs[12]);
            COEX(vecs[3], vecs[5]);
            COEX(vecs[6], vecs[8]);  /* step 8 */
            COEX(vecs[7], vecs[9]);
            COEX(vecs[10], vecs[12]);
            COEX(vecs[3], vecs[4]);
            COEX(vecs[5], vecs[6]);  /* step 9 */
            COEX(vecs[7], vecs[8]);
            COEX(vecs[9], vecs[10]);
            COEX(vecs[11], vecs[12]);
            COEX(vecs[6], vecs[7]);
            COEX(vecs[8], vecs[9]); /* step 10 */}

        /* auto generated code for merging 8 columns, each column contains 16 elements,
        * without transposition */
        void inline merge_8_columns_with_16_elements(__m256i *vecs) {
            vecs[8] = _mm256_shuffle_epi32(vecs[8], _MM_SHUFFLE(2, 3, 0, 1));
            COEX(vecs[7], vecs[8]);
            vecs[9] = _mm256_shuffle_epi32(vecs[9], _MM_SHUFFLE(2, 3, 0, 1));
            COEX(vecs[6], vecs[9]);
            vecs[10] = _mm256_shuffle_epi32(vecs[10], _MM_SHUFFLE(2, 3, 0, 1));
            COEX(vecs[5], vecs[10]);
            vecs[11] = _mm256_shuffle_epi32(vecs[11], _MM_SHUFFLE(2, 3, 0, 1));
            COEX(vecs[4], vecs[11]);
            vecs[12] = _mm256_shuffle_epi32(vecs[12], _MM_SHUFFLE(2, 3, 0, 1));
            COEX(vecs[3], vecs[12]);
            vecs[13] = _mm256_shuffle_epi32(vecs[13], _MM_SHUFFLE(2, 3, 0, 1));
            COEX(vecs[2], vecs[13]);
            vecs[14] = _mm256_shuffle_epi32(vecs[14], _MM_SHUFFLE(2, 3, 0, 1));
            COEX(vecs[1], vecs[14]);
            vecs[15] = _mm256_shuffle_epi32(vecs[15], _MM_SHUFFLE(2, 3, 0, 1));
            COEX(vecs[0], vecs[15]);
            vecs[4] = _mm256_shuffle_epi32(vecs[4], _MM_SHUFFLE(2, 3, 0, 1));
            COEX(vecs[3], vecs[4]);
            vecs[5] = _mm256_shuffle_epi32(vecs[5], _MM_SHUFFLE(2, 3, 0, 1));
            COEX(vecs[2], vecs[5]);
            vecs[6] = _mm256_shuffle_epi32(vecs[6], _MM_SHUFFLE(2, 3, 0, 1));
            COEX(vecs[1], vecs[6]);
            vecs[7] = _mm256_shuffle_epi32(vecs[7], _MM_SHUFFLE(2, 3, 0, 1));
            COEX(vecs[0], vecs[7]);
            vecs[12] = _mm256_shuffle_epi32(vecs[12], _MM_SHUFFLE(2, 3, 0, 1));
            COEX(vecs[11], vecs[12]);
            vecs[13] = _mm256_shuffle_epi32(vecs[13], _MM_SHUFFLE(2, 3, 0, 1));
            COEX(vecs[10], vecs[13]);
            vecs[14] = _mm256_shuffle_epi32(vecs[14], _MM_SHUFFLE(2, 3, 0, 1));
            COEX(vecs[9], vecs[14]);
            vecs[15] = _mm256_shuffle_epi32(vecs[15], _MM_SHUFFLE(2, 3, 0, 1));
            COEX(vecs[8], vecs[15]);
            vecs[2] = _mm256_shuffle_epi32(vecs[2], _MM_SHUFFLE(2, 3, 0, 1));
            COEX(vecs[1], vecs[2]);
            vecs[3] = _mm256_shuffle_epi32(vecs[3], _MM_SHUFFLE(2, 3, 0, 1));
            COEX(vecs[0], vecs[3]);
            vecs[6] = _mm256_shuffle_epi32(vecs[6], _MM_SHUFFLE(2, 3, 0, 1));
            COEX(vecs[5], vecs[6]);
            vecs[7] = _mm256_shuffle_epi32(vecs[7], _MM_SHUFFLE(2, 3, 0, 1));
            COEX(vecs[4], vecs[7]);
            vecs[10] = _mm256_shuffle_epi32(vecs[10], _MM_SHUFFLE(2, 3, 0, 1));
            COEX(vecs[9], vecs[10]);
            vecs[11] = _mm256_shuffle_epi32(vecs[11], _MM_SHUFFLE(2, 3, 0, 1));
            COEX(vecs[8], vecs[11]);
            vecs[14] = _mm256_shuffle_epi32(vecs[14], _MM_SHUFFLE(2, 3, 0, 1));
            COEX(vecs[13], vecs[14]);
            vecs[15] = _mm256_shuffle_epi32(vecs[15], _MM_SHUFFLE(2, 3, 0, 1));
            COEX(vecs[12], vecs[15]);
            vecs[1] = _mm256_shuffle_epi32(vecs[1], _MM_SHUFFLE(2, 3, 0, 1));
            COEX(vecs[0], vecs[1]);
            vecs[3] = _mm256_shuffle_epi32(vecs[3], _MM_SHUFFLE(2, 3, 0, 1));
            COEX(vecs[2], vecs[3]);
            vecs[5] = _mm256_shuffle_epi32(vecs[5], _MM_SHUFFLE(2, 3, 0, 1));
            COEX(vecs[4], vecs[5]);
            vecs[7] = _mm256_shuffle_epi32(vecs[7], _MM_SHUFFLE(2, 3, 0, 1));
            COEX(vecs[6], vecs[7]);
            vecs[9] = _mm256_shuffle_epi32(vecs[9], _MM_SHUFFLE(2, 3, 0, 1));
            COEX(vecs[8], vecs[9]);
            vecs[11] = _mm256_shuffle_epi32(vecs[11], _MM_SHUFFLE(2, 3, 0, 1));
            COEX(vecs[10], vecs[11]);
            vecs[13] = _mm256_shuffle_epi32(vecs[13], _MM_SHUFFLE(2, 3, 0, 1));
            COEX(vecs[12], vecs[13]);
            vecs[15] = _mm256_shuffle_epi32(vecs[15], _MM_SHUFFLE(2, 3, 0, 1));
            COEX(vecs[14], vecs[15]);
            COEX_SHUFFLE(vecs[0], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
            COEX_SHUFFLE(vecs[1], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
            COEX_SHUFFLE(vecs[2], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
            COEX_SHUFFLE(vecs[3], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
            COEX_SHUFFLE(vecs[4], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
            COEX_SHUFFLE(vecs[5], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
            COEX_SHUFFLE(vecs[6], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
            COEX_SHUFFLE(vecs[7], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
            COEX_SHUFFLE(vecs[8], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
            COEX_SHUFFLE(vecs[9], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
            COEX_SHUFFLE(vecs[10], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
            COEX_SHUFFLE(vecs[11], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
            COEX_SHUFFLE(vecs[12], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
            COEX_SHUFFLE(vecs[13], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
            COEX_SHUFFLE(vecs[14], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
            COEX_SHUFFLE(vecs[15], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
            vecs[8] = _mm256_shuffle_epi32(vecs[8], _MM_SHUFFLE(0, 1, 2, 3));
            COEX(vecs[7], vecs[8]);
            vecs[9] = _mm256_shuffle_epi32(vecs[9], _MM_SHUFFLE(0, 1, 2, 3));
            COEX(vecs[6], vecs[9]);
            vecs[10] = _mm256_shuffle_epi32(vecs[10], _MM_SHUFFLE(0, 1, 2, 3));
            COEX(vecs[5], vecs[10]);
            vecs[11] = _mm256_shuffle_epi32(vecs[11], _MM_SHUFFLE(0, 1, 2, 3));
            COEX(vecs[4], vecs[11]);
            vecs[12] = _mm256_shuffle_epi32(vecs[12], _MM_SHUFFLE(0, 1, 2, 3));
            COEX(vecs[3], vecs[12]);
            vecs[13] = _mm256_shuffle_epi32(vecs[13], _MM_SHUFFLE(0, 1, 2, 3));
            COEX(vecs[2], vecs[13]);
            vecs[14] = _mm256_shuffle_epi32(vecs[14], _MM_SHUFFLE(0, 1, 2, 3));
            COEX(vecs[1], vecs[14]);
            vecs[15] = _mm256_shuffle_epi32(vecs[15], _MM_SHUFFLE(0, 1, 2, 3));
            COEX(vecs[0], vecs[15]);
            vecs[4] = _mm256_shuffle_epi32(vecs[4], _MM_SHUFFLE(0, 1, 2, 3));
            COEX(vecs[3], vecs[4]);
            vecs[5] = _mm256_shuffle_epi32(vecs[5], _MM_SHUFFLE(0, 1, 2, 3));
            COEX(vecs[2], vecs[5]);
            vecs[6] = _mm256_shuffle_epi32(vecs[6], _MM_SHUFFLE(0, 1, 2, 3));
            COEX(vecs[1], vecs[6]);
            vecs[7] = _mm256_shuffle_epi32(vecs[7], _MM_SHUFFLE(0, 1, 2, 3));
            COEX(vecs[0], vecs[7]);
            vecs[12] = _mm256_shuffle_epi32(vecs[12], _MM_SHUFFLE(0, 1, 2, 3));
            COEX(vecs[11], vecs[12]);
            vecs[13] = _mm256_shuffle_epi32(vecs[13], _MM_SHUFFLE(0, 1, 2, 3));
            COEX(vecs[10], vecs[13]);
            vecs[14] = _mm256_shuffle_epi32(vecs[14], _MM_SHUFFLE(0, 1, 2, 3));
            COEX(vecs[9], vecs[14]);
            vecs[15] = _mm256_shuffle_epi32(vecs[15], _MM_SHUFFLE(0, 1, 2, 3));
            COEX(vecs[8], vecs[15]);
            vecs[2] = _mm256_shuffle_epi32(vecs[2], _MM_SHUFFLE(0, 1, 2, 3));
            COEX(vecs[1], vecs[2]);
            vecs[3] = _mm256_shuffle_epi32(vecs[3], _MM_SHUFFLE(0, 1, 2, 3));
            COEX(vecs[0], vecs[3]);
            vecs[6] = _mm256_shuffle_epi32(vecs[6], _MM_SHUFFLE(0, 1, 2, 3));
            COEX(vecs[5], vecs[6]);
            vecs[7] = _mm256_shuffle_epi32(vecs[7], _MM_SHUFFLE(0, 1, 2, 3));
            COEX(vecs[4], vecs[7]);
            vecs[10] = _mm256_shuffle_epi32(vecs[10], _MM_SHUFFLE(0, 1, 2, 3));
            COEX(vecs[9], vecs[10]);
            vecs[11] = _mm256_shuffle_epi32(vecs[11], _MM_SHUFFLE(0, 1, 2, 3));
            COEX(vecs[8], vecs[11]);
            vecs[14] = _mm256_shuffle_epi32(vecs[14], _MM_SHUFFLE(0, 1, 2, 3));
            COEX(vecs[13], vecs[14]);
            vecs[15] = _mm256_shuffle_epi32(vecs[15], _MM_SHUFFLE(0, 1, 2, 3));
            COEX(vecs[12], vecs[15]);
            vecs[1] = _mm256_shuffle_epi32(vecs[1], _MM_SHUFFLE(0, 1, 2, 3));
            COEX(vecs[0], vecs[1]);
            vecs[3] = _mm256_shuffle_epi32(vecs[3], _MM_SHUFFLE(0, 1, 2, 3));
            COEX(vecs[2], vecs[3]);
            vecs[5] = _mm256_shuffle_epi32(vecs[5], _MM_SHUFFLE(0, 1, 2, 3));
            COEX(vecs[4], vecs[5]);
            vecs[7] = _mm256_shuffle_epi32(vecs[7], _MM_SHUFFLE(0, 1, 2, 3));
            COEX(vecs[6], vecs[7]);
            vecs[9] = _mm256_shuffle_epi32(vecs[9], _MM_SHUFFLE(0, 1, 2, 3));
            COEX(vecs[8], vecs[9]);
            vecs[11] = _mm256_shuffle_epi32(vecs[11], _MM_SHUFFLE(0, 1, 2, 3));
            COEX(vecs[10], vecs[11]);
            vecs[13] = _mm256_shuffle_epi32(vecs[13], _MM_SHUFFLE(0, 1, 2, 3));
            COEX(vecs[12], vecs[13]);
            vecs[15] = _mm256_shuffle_epi32(vecs[15], _MM_SHUFFLE(0, 1, 2, 3));
            COEX(vecs[14], vecs[15]);
            COEX_SHUFFLE(vecs[0], 3, 2, 1, 0, 7, 6, 5, 4, ASC);
            COEX_SHUFFLE(vecs[0], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
            COEX_SHUFFLE(vecs[1], 3, 2, 1, 0, 7, 6, 5, 4, ASC);
            COEX_SHUFFLE(vecs[1], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
            COEX_SHUFFLE(vecs[2], 3, 2, 1, 0, 7, 6, 5, 4, ASC);
            COEX_SHUFFLE(vecs[2], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
            COEX_SHUFFLE(vecs[3], 3, 2, 1, 0, 7, 6, 5, 4, ASC);
            COEX_SHUFFLE(vecs[3], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
            COEX_SHUFFLE(vecs[4], 3, 2, 1, 0, 7, 6, 5, 4, ASC);
            COEX_SHUFFLE(vecs[4], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
            COEX_SHUFFLE(vecs[5], 3, 2, 1, 0, 7, 6, 5, 4, ASC);
            COEX_SHUFFLE(vecs[5], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
            COEX_SHUFFLE(vecs[6], 3, 2, 1, 0, 7, 6, 5, 4, ASC);
            COEX_SHUFFLE(vecs[6], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
            COEX_SHUFFLE(vecs[7], 3, 2, 1, 0, 7, 6, 5, 4, ASC);
            COEX_SHUFFLE(vecs[7], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
            COEX_SHUFFLE(vecs[8], 3, 2, 1, 0, 7, 6, 5, 4, ASC);
            COEX_SHUFFLE(vecs[8], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
            COEX_SHUFFLE(vecs[9], 3, 2, 1, 0, 7, 6, 5, 4, ASC);
            COEX_SHUFFLE(vecs[9], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
            COEX_SHUFFLE(vecs[10], 3, 2, 1, 0, 7, 6, 5, 4, ASC);
            COEX_SHUFFLE(vecs[10], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
            COEX_SHUFFLE(vecs[11], 3, 2, 1, 0, 7, 6, 5, 4, ASC);
            COEX_SHUFFLE(vecs[11], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
            COEX_SHUFFLE(vecs[12], 3, 2, 1, 0, 7, 6, 5, 4, ASC);
            COEX_SHUFFLE(vecs[12], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
            COEX_SHUFFLE(vecs[13], 3, 2, 1, 0, 7, 6, 5, 4, ASC);
            COEX_SHUFFLE(vecs[13], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
            COEX_SHUFFLE(vecs[14], 3, 2, 1, 0, 7, 6, 5, 4, ASC);
            COEX_SHUFFLE(vecs[14], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
            COEX_SHUFFLE(vecs[15], 3, 2, 1, 0, 7, 6, 5, 4, ASC);
            COEX_SHUFFLE(vecs[15], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
            REVERSE_VEC(vecs[8]);
            COEX(vecs[7], vecs[8]);
            REVERSE_VEC(vecs[9]);
            COEX(vecs[6], vecs[9]);
            REVERSE_VEC(vecs[10]);
            COEX(vecs[5], vecs[10]);
            REVERSE_VEC(vecs[11]);
            COEX(vecs[4], vecs[11]);
            REVERSE_VEC(vecs[12]);
            COEX(vecs[3], vecs[12]);
            REVERSE_VEC(vecs[13]);
            COEX(vecs[2], vecs[13]);
            REVERSE_VEC(vecs[14]);
            COEX(vecs[1], vecs[14]);
            REVERSE_VEC(vecs[15]);
            COEX(vecs[0], vecs[15]);
            REVERSE_VEC(vecs[4]);
            COEX(vecs[3], vecs[4]);
            REVERSE_VEC(vecs[5]);
            COEX(vecs[2], vecs[5]);
            REVERSE_VEC(vecs[6]);
            COEX(vecs[1], vecs[6]);
            REVERSE_VEC(vecs[7]);
            COEX(vecs[0], vecs[7]);
            REVERSE_VEC(vecs[12]);
            COEX(vecs[11], vecs[12]);
            REVERSE_VEC(vecs[13]);
            COEX(vecs[10], vecs[13]);
            REVERSE_VEC(vecs[14]);
            COEX(vecs[9], vecs[14]);
            REVERSE_VEC(vecs[15]);
            COEX(vecs[8], vecs[15]);
            REVERSE_VEC(vecs[2]);
            COEX(vecs[1], vecs[2]);
            REVERSE_VEC(vecs[3]);
            COEX(vecs[0], vecs[3]);
            REVERSE_VEC(vecs[6]);
            COEX(vecs[5], vecs[6]);
            REVERSE_VEC(vecs[7]);
            COEX(vecs[4], vecs[7]);
            REVERSE_VEC(vecs[10]);
            COEX(vecs[9], vecs[10]);
            REVERSE_VEC(vecs[11]);
            COEX(vecs[8], vecs[11]);
            REVERSE_VEC(vecs[14]);
            COEX(vecs[13], vecs[14]);
            REVERSE_VEC(vecs[15]);
            COEX(vecs[12], vecs[15]);
            REVERSE_VEC(vecs[1]);
            COEX(vecs[0], vecs[1]);
            REVERSE_VEC(vecs[3]);
            COEX(vecs[2], vecs[3]);
            REVERSE_VEC(vecs[5]);
            COEX(vecs[4], vecs[5]);
            REVERSE_VEC(vecs[7]);
            COEX(vecs[6], vecs[7]);
            REVERSE_VEC(vecs[9]);
            COEX(vecs[8], vecs[9]);
            REVERSE_VEC(vecs[11]);
            COEX(vecs[10], vecs[11]);
            REVERSE_VEC(vecs[13]);
            COEX(vecs[12], vecs[13]);
            REVERSE_VEC(vecs[15]);
            COEX(vecs[14], vecs[15]);
            COEX_PERMUTE(vecs[0], 7, 6, 5, 4, 3, 2, 1, 0, ASC);
            COEX_SHUFFLE(vecs[0], 2, 3, 0, 1, 6, 7, 4, 5, ASC);
            COEX_SHUFFLE(vecs[0], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
            COEX_PERMUTE(vecs[1], 7, 6, 5, 4, 3, 2, 1, 0, ASC);
            COEX_SHUFFLE(vecs[1], 2, 3, 0, 1, 6, 7, 4, 5, ASC);
            COEX_SHUFFLE(vecs[1], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
            COEX_PERMUTE(vecs[2], 7, 6, 5, 4, 3, 2, 1, 0, ASC);
            COEX_SHUFFLE(vecs[2], 2, 3, 0, 1, 6, 7, 4, 5, ASC);
            COEX_SHUFFLE(vecs[2], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
            COEX_PERMUTE(vecs[3], 7, 6, 5, 4, 3, 2, 1, 0, ASC);
            COEX_SHUFFLE(vecs[3], 2, 3, 0, 1, 6, 7, 4, 5, ASC);
            COEX_SHUFFLE(vecs[3], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
            COEX_PERMUTE(vecs[4], 7, 6, 5, 4, 3, 2, 1, 0, ASC);
            COEX_SHUFFLE(vecs[4], 2, 3, 0, 1, 6, 7, 4, 5, ASC);
            COEX_SHUFFLE(vecs[4], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
            COEX_PERMUTE(vecs[5], 7, 6, 5, 4, 3, 2, 1, 0, ASC);
            COEX_SHUFFLE(vecs[5], 2, 3, 0, 1, 6, 7, 4, 5, ASC);
            COEX_SHUFFLE(vecs[5], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
            COEX_PERMUTE(vecs[6], 7, 6, 5, 4, 3, 2, 1, 0, ASC);
            COEX_SHUFFLE(vecs[6], 2, 3, 0, 1, 6, 7, 4, 5, ASC);
            COEX_SHUFFLE(vecs[6], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
            COEX_PERMUTE(vecs[7], 7, 6, 5, 4, 3, 2, 1, 0, ASC);
            COEX_SHUFFLE(vecs[7], 2, 3, 0, 1, 6, 7, 4, 5, ASC);
            COEX_SHUFFLE(vecs[7], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
            COEX_PERMUTE(vecs[8], 7, 6, 5, 4, 3, 2, 1, 0, ASC);
            COEX_SHUFFLE(vecs[8], 2, 3, 0, 1, 6, 7, 4, 5, ASC);
            COEX_SHUFFLE(vecs[8], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
            COEX_PERMUTE(vecs[9], 7, 6, 5, 4, 3, 2, 1, 0, ASC);
            COEX_SHUFFLE(vecs[9], 2, 3, 0, 1, 6, 7, 4, 5, ASC);
            COEX_SHUFFLE(vecs[9], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
            COEX_PERMUTE(vecs[10], 7, 6, 5, 4, 3, 2, 1, 0, ASC);
            COEX_SHUFFLE(vecs[10], 2, 3, 0, 1, 6, 7, 4, 5, ASC);
            COEX_SHUFFLE(vecs[10], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
            COEX_PERMUTE(vecs[11], 7, 6, 5, 4, 3, 2, 1, 0, ASC);
            COEX_SHUFFLE(vecs[11], 2, 3, 0, 1, 6, 7, 4, 5, ASC);
            COEX_SHUFFLE(vecs[11], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
            COEX_PERMUTE(vecs[12], 7, 6, 5, 4, 3, 2, 1, 0, ASC);
            COEX_SHUFFLE(vecs[12], 2, 3, 0, 1, 6, 7, 4, 5, ASC);
            COEX_SHUFFLE(vecs[12], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
            COEX_PERMUTE(vecs[13], 7, 6, 5, 4, 3, 2, 1, 0, ASC);
            COEX_SHUFFLE(vecs[13], 2, 3, 0, 1, 6, 7, 4, 5, ASC);
            COEX_SHUFFLE(vecs[13], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
            COEX_PERMUTE(vecs[14], 7, 6, 5, 4, 3, 2, 1, 0, ASC);
            COEX_SHUFFLE(vecs[14], 2, 3, 0, 1, 6, 7, 4, 5, ASC);
            COEX_SHUFFLE(vecs[14], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
            COEX_PERMUTE(vecs[15], 7, 6, 5, 4, 3, 2, 1, 0, ASC);
            COEX_SHUFFLE(vecs[15], 2, 3, 0, 1, 6, 7, 4, 5, ASC);
            COEX_SHUFFLE(vecs[15], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
        }

        inline void sort_int_sorting_network(int *arr, int *buff, int n) {
            if (n < 2) return;
            __m256i *buffer = reinterpret_cast<__m256i *>(buff);

            auto remainder = int(n % 8 ? n % 8 : 8);
            int idx_max_pad = n - remainder;
            auto mask = _mm256_add_epi32(_mm256_set1_epi32(-remainder),
                                         _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7));
            auto max_pad_vec = _mm256_blendv_epi8(_mm256_set1_epi32(INT32_MAX),
                                                  _mm256_maskload_epi32(arr + idx_max_pad, mask), mask);

            for (int i = 0; i < idx_max_pad / 8; ++i) {
                buffer[i] = LOAD_VECTOR(arr + i * 8);
            }
            buffer[idx_max_pad / 8] = max_pad_vec;
            buffer[idx_max_pad / 8 + 1] = _mm256_set1_epi32(INT32_MAX);

            int N = ((idx_max_pad % 16 == 0) * 8 + idx_max_pad + 8) / 8;

            for (int j = 0; j < N - N % 16; j += 16) {
                sort_16_int_vertical(buffer + j);
                merge_8_columns_with_16_elements(buffer + j);
            }
            for (int i = N - N % 16; i < N; i += 2) {
                sort_16(buffer[i], buffer[i + 1]);
            }
            bitonic_merge_16(buffer + N - N % 16, N % 16, 2);
            bitonic_merge_128(buffer, N, 16);
            for (int i = 0; i < idx_max_pad / 8; i += 1) {
                STORE_VECTOR(arr + i * 8, buffer[i]);
            }
            _mm256_maskstore_epi32(arr + idx_max_pad, mask, buffer[idx_max_pad / 8]);
        }
/* end of sorting networks
*********************************************/


/*** vectorized helping functions
 * from Blacher https://github.com/simd-sorting/fast-and-robust/blob/master/avx2_sort_demo/avx2sort.h
**************************************/

/* auto generated permutations masks for quicksort
 * from Blacher https://github.com/simd-sorting/fast-and-robust/blob/master/avx2_sort_demo/avx2sort.h */
        __m256i permutation_masks[256] = {_mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7),
                                          _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 0),
                                          _mm256_setr_epi32(0, 2, 3, 4, 5, 6, 7, 1),
                                          _mm256_setr_epi32(2, 3, 4, 5, 6, 7, 0, 1),
                                          _mm256_setr_epi32(0, 1, 3, 4, 5, 6, 7, 2),
                                          _mm256_setr_epi32(1, 3, 4, 5, 6, 7, 0, 2),
                                          _mm256_setr_epi32(0, 3, 4, 5, 6, 7, 1, 2),
                                          _mm256_setr_epi32(3, 4, 5, 6, 7, 0, 1, 2),
                                          _mm256_setr_epi32(0, 1, 2, 4, 5, 6, 7, 3),
                                          _mm256_setr_epi32(1, 2, 4, 5, 6, 7, 0, 3),
                                          _mm256_setr_epi32(0, 2, 4, 5, 6, 7, 1, 3),
                                          _mm256_setr_epi32(2, 4, 5, 6, 7, 0, 1, 3),
                                          _mm256_setr_epi32(0, 1, 4, 5, 6, 7, 2, 3),
                                          _mm256_setr_epi32(1, 4, 5, 6, 7, 0, 2, 3),
                                          _mm256_setr_epi32(0, 4, 5, 6, 7, 1, 2, 3),
                                          _mm256_setr_epi32(4, 5, 6, 7, 0, 1, 2, 3),
                                          _mm256_setr_epi32(0, 1, 2, 3, 5, 6, 7, 4),
                                          _mm256_setr_epi32(1, 2, 3, 5, 6, 7, 0, 4),
                                          _mm256_setr_epi32(0, 2, 3, 5, 6, 7, 1, 4),
                                          _mm256_setr_epi32(2, 3, 5, 6, 7, 0, 1, 4),
                                          _mm256_setr_epi32(0, 1, 3, 5, 6, 7, 2, 4),
                                          _mm256_setr_epi32(1, 3, 5, 6, 7, 0, 2, 4),
                                          _mm256_setr_epi32(0, 3, 5, 6, 7, 1, 2, 4),
                                          _mm256_setr_epi32(3, 5, 6, 7, 0, 1, 2, 4),
                                          _mm256_setr_epi32(0, 1, 2, 5, 6, 7, 3, 4),
                                          _mm256_setr_epi32(1, 2, 5, 6, 7, 0, 3, 4),
                                          _mm256_setr_epi32(0, 2, 5, 6, 7, 1, 3, 4),
                                          _mm256_setr_epi32(2, 5, 6, 7, 0, 1, 3, 4),
                                          _mm256_setr_epi32(0, 1, 5, 6, 7, 2, 3, 4),
                                          _mm256_setr_epi32(1, 5, 6, 7, 0, 2, 3, 4),
                                          _mm256_setr_epi32(0, 5, 6, 7, 1, 2, 3, 4),
                                          _mm256_setr_epi32(5, 6, 7, 0, 1, 2, 3, 4),
                                          _mm256_setr_epi32(0, 1, 2, 3, 4, 6, 7, 5),
                                          _mm256_setr_epi32(1, 2, 3, 4, 6, 7, 0, 5),
                                          _mm256_setr_epi32(0, 2, 3, 4, 6, 7, 1, 5),
                                          _mm256_setr_epi32(2, 3, 4, 6, 7, 0, 1, 5),
                                          _mm256_setr_epi32(0, 1, 3, 4, 6, 7, 2, 5),
                                          _mm256_setr_epi32(1, 3, 4, 6, 7, 0, 2, 5),
                                          _mm256_setr_epi32(0, 3, 4, 6, 7, 1, 2, 5),
                                          _mm256_setr_epi32(3, 4, 6, 7, 0, 1, 2, 5),
                                          _mm256_setr_epi32(0, 1, 2, 4, 6, 7, 3, 5),
                                          _mm256_setr_epi32(1, 2, 4, 6, 7, 0, 3, 5),
                                          _mm256_setr_epi32(0, 2, 4, 6, 7, 1, 3, 5),
                                          _mm256_setr_epi32(2, 4, 6, 7, 0, 1, 3, 5),
                                          _mm256_setr_epi32(0, 1, 4, 6, 7, 2, 3, 5),
                                          _mm256_setr_epi32(1, 4, 6, 7, 0, 2, 3, 5),
                                          _mm256_setr_epi32(0, 4, 6, 7, 1, 2, 3, 5),
                                          _mm256_setr_epi32(4, 6, 7, 0, 1, 2, 3, 5),
                                          _mm256_setr_epi32(0, 1, 2, 3, 6, 7, 4, 5),
                                          _mm256_setr_epi32(1, 2, 3, 6, 7, 0, 4, 5),
                                          _mm256_setr_epi32(0, 2, 3, 6, 7, 1, 4, 5),
                                          _mm256_setr_epi32(2, 3, 6, 7, 0, 1, 4, 5),
                                          _mm256_setr_epi32(0, 1, 3, 6, 7, 2, 4, 5),
                                          _mm256_setr_epi32(1, 3, 6, 7, 0, 2, 4, 5),
                                          _mm256_setr_epi32(0, 3, 6, 7, 1, 2, 4, 5),
                                          _mm256_setr_epi32(3, 6, 7, 0, 1, 2, 4, 5),
                                          _mm256_setr_epi32(0, 1, 2, 6, 7, 3, 4, 5),
                                          _mm256_setr_epi32(1, 2, 6, 7, 0, 3, 4, 5),
                                          _mm256_setr_epi32(0, 2, 6, 7, 1, 3, 4, 5),
                                          _mm256_setr_epi32(2, 6, 7, 0, 1, 3, 4, 5),
                                          _mm256_setr_epi32(0, 1, 6, 7, 2, 3, 4, 5),
                                          _mm256_setr_epi32(1, 6, 7, 0, 2, 3, 4, 5),
                                          _mm256_setr_epi32(0, 6, 7, 1, 2, 3, 4, 5),
                                          _mm256_setr_epi32(6, 7, 0, 1, 2, 3, 4, 5),
                                          _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 7, 6),
                                          _mm256_setr_epi32(1, 2, 3, 4, 5, 7, 0, 6),
                                          _mm256_setr_epi32(0, 2, 3, 4, 5, 7, 1, 6),
                                          _mm256_setr_epi32(2, 3, 4, 5, 7, 0, 1, 6),
                                          _mm256_setr_epi32(0, 1, 3, 4, 5, 7, 2, 6),
                                          _mm256_setr_epi32(1, 3, 4, 5, 7, 0, 2, 6),
                                          _mm256_setr_epi32(0, 3, 4, 5, 7, 1, 2, 6),
                                          _mm256_setr_epi32(3, 4, 5, 7, 0, 1, 2, 6),
                                          _mm256_setr_epi32(0, 1, 2, 4, 5, 7, 3, 6),
                                          _mm256_setr_epi32(1, 2, 4, 5, 7, 0, 3, 6),
                                          _mm256_setr_epi32(0, 2, 4, 5, 7, 1, 3, 6),
                                          _mm256_setr_epi32(2, 4, 5, 7, 0, 1, 3, 6),
                                          _mm256_setr_epi32(0, 1, 4, 5, 7, 2, 3, 6),
                                          _mm256_setr_epi32(1, 4, 5, 7, 0, 2, 3, 6),
                                          _mm256_setr_epi32(0, 4, 5, 7, 1, 2, 3, 6),
                                          _mm256_setr_epi32(4, 5, 7, 0, 1, 2, 3, 6),
                                          _mm256_setr_epi32(0, 1, 2, 3, 5, 7, 4, 6),
                                          _mm256_setr_epi32(1, 2, 3, 5, 7, 0, 4, 6),
                                          _mm256_setr_epi32(0, 2, 3, 5, 7, 1, 4, 6),
                                          _mm256_setr_epi32(2, 3, 5, 7, 0, 1, 4, 6),
                                          _mm256_setr_epi32(0, 1, 3, 5, 7, 2, 4, 6),
                                          _mm256_setr_epi32(1, 3, 5, 7, 0, 2, 4, 6),
                                          _mm256_setr_epi32(0, 3, 5, 7, 1, 2, 4, 6),
                                          _mm256_setr_epi32(3, 5, 7, 0, 1, 2, 4, 6),
                                          _mm256_setr_epi32(0, 1, 2, 5, 7, 3, 4, 6),
                                          _mm256_setr_epi32(1, 2, 5, 7, 0, 3, 4, 6),
                                          _mm256_setr_epi32(0, 2, 5, 7, 1, 3, 4, 6),
                                          _mm256_setr_epi32(2, 5, 7, 0, 1, 3, 4, 6),
                                          _mm256_setr_epi32(0, 1, 5, 7, 2, 3, 4, 6),
                                          _mm256_setr_epi32(1, 5, 7, 0, 2, 3, 4, 6),
                                          _mm256_setr_epi32(0, 5, 7, 1, 2, 3, 4, 6),
                                          _mm256_setr_epi32(5, 7, 0, 1, 2, 3, 4, 6),
                                          _mm256_setr_epi32(0, 1, 2, 3, 4, 7, 5, 6),
                                          _mm256_setr_epi32(1, 2, 3, 4, 7, 0, 5, 6),
                                          _mm256_setr_epi32(0, 2, 3, 4, 7, 1, 5, 6),
                                          _mm256_setr_epi32(2, 3, 4, 7, 0, 1, 5, 6),
                                          _mm256_setr_epi32(0, 1, 3, 4, 7, 2, 5, 6),
                                          _mm256_setr_epi32(1, 3, 4, 7, 0, 2, 5, 6),
                                          _mm256_setr_epi32(0, 3, 4, 7, 1, 2, 5, 6),
                                          _mm256_setr_epi32(3, 4, 7, 0, 1, 2, 5, 6),
                                          _mm256_setr_epi32(0, 1, 2, 4, 7, 3, 5, 6),
                                          _mm256_setr_epi32(1, 2, 4, 7, 0, 3, 5, 6),
                                          _mm256_setr_epi32(0, 2, 4, 7, 1, 3, 5, 6),
                                          _mm256_setr_epi32(2, 4, 7, 0, 1, 3, 5, 6),
                                          _mm256_setr_epi32(0, 1, 4, 7, 2, 3, 5, 6),
                                          _mm256_setr_epi32(1, 4, 7, 0, 2, 3, 5, 6),
                                          _mm256_setr_epi32(0, 4, 7, 1, 2, 3, 5, 6),
                                          _mm256_setr_epi32(4, 7, 0, 1, 2, 3, 5, 6),
                                          _mm256_setr_epi32(0, 1, 2, 3, 7, 4, 5, 6),
                                          _mm256_setr_epi32(1, 2, 3, 7, 0, 4, 5, 6),
                                          _mm256_setr_epi32(0, 2, 3, 7, 1, 4, 5, 6),
                                          _mm256_setr_epi32(2, 3, 7, 0, 1, 4, 5, 6),
                                          _mm256_setr_epi32(0, 1, 3, 7, 2, 4, 5, 6),
                                          _mm256_setr_epi32(1, 3, 7, 0, 2, 4, 5, 6),
                                          _mm256_setr_epi32(0, 3, 7, 1, 2, 4, 5, 6),
                                          _mm256_setr_epi32(3, 7, 0, 1, 2, 4, 5, 6),
                                          _mm256_setr_epi32(0, 1, 2, 7, 3, 4, 5, 6),
                                          _mm256_setr_epi32(1, 2, 7, 0, 3, 4, 5, 6),
                                          _mm256_setr_epi32(0, 2, 7, 1, 3, 4, 5, 6),
                                          _mm256_setr_epi32(2, 7, 0, 1, 3, 4, 5, 6),
                                          _mm256_setr_epi32(0, 1, 7, 2, 3, 4, 5, 6),
                                          _mm256_setr_epi32(1, 7, 0, 2, 3, 4, 5, 6),
                                          _mm256_setr_epi32(0, 7, 1, 2, 3, 4, 5, 6),
                                          _mm256_setr_epi32(7, 0, 1, 2, 3, 4, 5, 6),
                                          _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7),
                                          _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 0, 7),
                                          _mm256_setr_epi32(0, 2, 3, 4, 5, 6, 1, 7),
                                          _mm256_setr_epi32(2, 3, 4, 5, 6, 0, 1, 7),
                                          _mm256_setr_epi32(0, 1, 3, 4, 5, 6, 2, 7),
                                          _mm256_setr_epi32(1, 3, 4, 5, 6, 0, 2, 7),
                                          _mm256_setr_epi32(0, 3, 4, 5, 6, 1, 2, 7),
                                          _mm256_setr_epi32(3, 4, 5, 6, 0, 1, 2, 7),
                                          _mm256_setr_epi32(0, 1, 2, 4, 5, 6, 3, 7),
                                          _mm256_setr_epi32(1, 2, 4, 5, 6, 0, 3, 7),
                                          _mm256_setr_epi32(0, 2, 4, 5, 6, 1, 3, 7),
                                          _mm256_setr_epi32(2, 4, 5, 6, 0, 1, 3, 7),
                                          _mm256_setr_epi32(0, 1, 4, 5, 6, 2, 3, 7),
                                          _mm256_setr_epi32(1, 4, 5, 6, 0, 2, 3, 7),
                                          _mm256_setr_epi32(0, 4, 5, 6, 1, 2, 3, 7),
                                          _mm256_setr_epi32(4, 5, 6, 0, 1, 2, 3, 7),
                                          _mm256_setr_epi32(0, 1, 2, 3, 5, 6, 4, 7),
                                          _mm256_setr_epi32(1, 2, 3, 5, 6, 0, 4, 7),
                                          _mm256_setr_epi32(0, 2, 3, 5, 6, 1, 4, 7),
                                          _mm256_setr_epi32(2, 3, 5, 6, 0, 1, 4, 7),
                                          _mm256_setr_epi32(0, 1, 3, 5, 6, 2, 4, 7),
                                          _mm256_setr_epi32(1, 3, 5, 6, 0, 2, 4, 7),
                                          _mm256_setr_epi32(0, 3, 5, 6, 1, 2, 4, 7),
                                          _mm256_setr_epi32(3, 5, 6, 0, 1, 2, 4, 7),
                                          _mm256_setr_epi32(0, 1, 2, 5, 6, 3, 4, 7),
                                          _mm256_setr_epi32(1, 2, 5, 6, 0, 3, 4, 7),
                                          _mm256_setr_epi32(0, 2, 5, 6, 1, 3, 4, 7),
                                          _mm256_setr_epi32(2, 5, 6, 0, 1, 3, 4, 7),
                                          _mm256_setr_epi32(0, 1, 5, 6, 2, 3, 4, 7),
                                          _mm256_setr_epi32(1, 5, 6, 0, 2, 3, 4, 7),
                                          _mm256_setr_epi32(0, 5, 6, 1, 2, 3, 4, 7),
                                          _mm256_setr_epi32(5, 6, 0, 1, 2, 3, 4, 7),
                                          _mm256_setr_epi32(0, 1, 2, 3, 4, 6, 5, 7),
                                          _mm256_setr_epi32(1, 2, 3, 4, 6, 0, 5, 7),
                                          _mm256_setr_epi32(0, 2, 3, 4, 6, 1, 5, 7),
                                          _mm256_setr_epi32(2, 3, 4, 6, 0, 1, 5, 7),
                                          _mm256_setr_epi32(0, 1, 3, 4, 6, 2, 5, 7),
                                          _mm256_setr_epi32(1, 3, 4, 6, 0, 2, 5, 7),
                                          _mm256_setr_epi32(0, 3, 4, 6, 1, 2, 5, 7),
                                          _mm256_setr_epi32(3, 4, 6, 0, 1, 2, 5, 7),
                                          _mm256_setr_epi32(0, 1, 2, 4, 6, 3, 5, 7),
                                          _mm256_setr_epi32(1, 2, 4, 6, 0, 3, 5, 7),
                                          _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7),
                                          _mm256_setr_epi32(2, 4, 6, 0, 1, 3, 5, 7),
                                          _mm256_setr_epi32(0, 1, 4, 6, 2, 3, 5, 7),
                                          _mm256_setr_epi32(1, 4, 6, 0, 2, 3, 5, 7),
                                          _mm256_setr_epi32(0, 4, 6, 1, 2, 3, 5, 7),
                                          _mm256_setr_epi32(4, 6, 0, 1, 2, 3, 5, 7),
                                          _mm256_setr_epi32(0, 1, 2, 3, 6, 4, 5, 7),
                                          _mm256_setr_epi32(1, 2, 3, 6, 0, 4, 5, 7),
                                          _mm256_setr_epi32(0, 2, 3, 6, 1, 4, 5, 7),
                                          _mm256_setr_epi32(2, 3, 6, 0, 1, 4, 5, 7),
                                          _mm256_setr_epi32(0, 1, 3, 6, 2, 4, 5, 7),
                                          _mm256_setr_epi32(1, 3, 6, 0, 2, 4, 5, 7),
                                          _mm256_setr_epi32(0, 3, 6, 1, 2, 4, 5, 7),
                                          _mm256_setr_epi32(3, 6, 0, 1, 2, 4, 5, 7),
                                          _mm256_setr_epi32(0, 1, 2, 6, 3, 4, 5, 7),
                                          _mm256_setr_epi32(1, 2, 6, 0, 3, 4, 5, 7),
                                          _mm256_setr_epi32(0, 2, 6, 1, 3, 4, 5, 7),
                                          _mm256_setr_epi32(2, 6, 0, 1, 3, 4, 5, 7),
                                          _mm256_setr_epi32(0, 1, 6, 2, 3, 4, 5, 7),
                                          _mm256_setr_epi32(1, 6, 0, 2, 3, 4, 5, 7),
                                          _mm256_setr_epi32(0, 6, 1, 2, 3, 4, 5, 7),
                                          _mm256_setr_epi32(6, 0, 1, 2, 3, 4, 5, 7),
                                          _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7),
                                          _mm256_setr_epi32(1, 2, 3, 4, 5, 0, 6, 7),
                                          _mm256_setr_epi32(0, 2, 3, 4, 5, 1, 6, 7),
                                          _mm256_setr_epi32(2, 3, 4, 5, 0, 1, 6, 7),
                                          _mm256_setr_epi32(0, 1, 3, 4, 5, 2, 6, 7),
                                          _mm256_setr_epi32(1, 3, 4, 5, 0, 2, 6, 7),
                                          _mm256_setr_epi32(0, 3, 4, 5, 1, 2, 6, 7),
                                          _mm256_setr_epi32(3, 4, 5, 0, 1, 2, 6, 7),
                                          _mm256_setr_epi32(0, 1, 2, 4, 5, 3, 6, 7),
                                          _mm256_setr_epi32(1, 2, 4, 5, 0, 3, 6, 7),
                                          _mm256_setr_epi32(0, 2, 4, 5, 1, 3, 6, 7),
                                          _mm256_setr_epi32(2, 4, 5, 0, 1, 3, 6, 7),
                                          _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7),
                                          _mm256_setr_epi32(1, 4, 5, 0, 2, 3, 6, 7),
                                          _mm256_setr_epi32(0, 4, 5, 1, 2, 3, 6, 7),
                                          _mm256_setr_epi32(4, 5, 0, 1, 2, 3, 6, 7),
                                          _mm256_setr_epi32(0, 1, 2, 3, 5, 4, 6, 7),
                                          _mm256_setr_epi32(1, 2, 3, 5, 0, 4, 6, 7),
                                          _mm256_setr_epi32(0, 2, 3, 5, 1, 4, 6, 7),
                                          _mm256_setr_epi32(2, 3, 5, 0, 1, 4, 6, 7),
                                          _mm256_setr_epi32(0, 1, 3, 5, 2, 4, 6, 7),
                                          _mm256_setr_epi32(1, 3, 5, 0, 2, 4, 6, 7),
                                          _mm256_setr_epi32(0, 3, 5, 1, 2, 4, 6, 7),
                                          _mm256_setr_epi32(3, 5, 0, 1, 2, 4, 6, 7),
                                          _mm256_setr_epi32(0, 1, 2, 5, 3, 4, 6, 7),
                                          _mm256_setr_epi32(1, 2, 5, 0, 3, 4, 6, 7),
                                          _mm256_setr_epi32(0, 2, 5, 1, 3, 4, 6, 7),
                                          _mm256_setr_epi32(2, 5, 0, 1, 3, 4, 6, 7),
                                          _mm256_setr_epi32(0, 1, 5, 2, 3, 4, 6, 7),
                                          _mm256_setr_epi32(1, 5, 0, 2, 3, 4, 6, 7),
                                          _mm256_setr_epi32(0, 5, 1, 2, 3, 4, 6, 7),
                                          _mm256_setr_epi32(5, 0, 1, 2, 3, 4, 6, 7),
                                          _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7),
                                          _mm256_setr_epi32(1, 2, 3, 4, 0, 5, 6, 7),
                                          _mm256_setr_epi32(0, 2, 3, 4, 1, 5, 6, 7),
                                          _mm256_setr_epi32(2, 3, 4, 0, 1, 5, 6, 7),
                                          _mm256_setr_epi32(0, 1, 3, 4, 2, 5, 6, 7),
                                          _mm256_setr_epi32(1, 3, 4, 0, 2, 5, 6, 7),
                                          _mm256_setr_epi32(0, 3, 4, 1, 2, 5, 6, 7),
                                          _mm256_setr_epi32(3, 4, 0, 1, 2, 5, 6, 7),
                                          _mm256_setr_epi32(0, 1, 2, 4, 3, 5, 6, 7),
                                          _mm256_setr_epi32(1, 2, 4, 0, 3, 5, 6, 7),
                                          _mm256_setr_epi32(0, 2, 4, 1, 3, 5, 6, 7),
                                          _mm256_setr_epi32(2, 4, 0, 1, 3, 5, 6, 7),
                                          _mm256_setr_epi32(0, 1, 4, 2, 3, 5, 6, 7),
                                          _mm256_setr_epi32(1, 4, 0, 2, 3, 5, 6, 7),
                                          _mm256_setr_epi32(0, 4, 1, 2, 3, 5, 6, 7),
                                          _mm256_setr_epi32(4, 0, 1, 2, 3, 5, 6, 7),
                                          _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7),
                                          _mm256_setr_epi32(1, 2, 3, 0, 4, 5, 6, 7),
                                          _mm256_setr_epi32(0, 2, 3, 1, 4, 5, 6, 7),
                                          _mm256_setr_epi32(2, 3, 0, 1, 4, 5, 6, 7),
                                          _mm256_setr_epi32(0, 1, 3, 2, 4, 5, 6, 7),
                                          _mm256_setr_epi32(1, 3, 0, 2, 4, 5, 6, 7),
                                          _mm256_setr_epi32(0, 3, 1, 2, 4, 5, 6, 7),
                                          _mm256_setr_epi32(3, 0, 1, 2, 4, 5, 6, 7),
                                          _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7),
                                          _mm256_setr_epi32(1, 2, 0, 3, 4, 5, 6, 7),
                                          _mm256_setr_epi32(0, 2, 1, 3, 4, 5, 6, 7),
                                          _mm256_setr_epi32(2, 0, 1, 3, 4, 5, 6, 7),
                                          _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7),
                                          _mm256_setr_epi32(1, 0, 2, 3, 4, 5, 6, 7),
                                          _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7),
                                          _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7)};

/* partition a single vector, return how many values are greater than pv,
* update s_key and largest values in s_vec and b_vec respectively
* from Blacher https://github.com/simd-sorting/fast-and-robust/blob/master/avx2_sort_demo/avx2sort.h */
inline int partition_vec(__m256i &vec, __m256i &pv_vec, __m256i &s_vec,
                         __m256i &b_vec) {

    /* which elements are larger than the pv */
    __m256i compared = _mm256_cmpgt_epi32(vec, pv_vec);
    /* update the s_key and largest values of the array */
    s_vec = _mm256_min_epi32(vec, s_vec);
    b_vec = _mm256_max_epi32(vec, b_vec);
    /* extract the most significant bit from each integer of the vector */
    int mm = _mm256_movemask_ps(reinterpret_cast<__m256>(compared));
    /* how many ones, each 1 stands for an element greater than pv */
    int num_b_keys = _mm_popcnt_u32((mm));
    /* permute elements larger than pv to the right, and,
     * smaller than or equal to the pv, to the left */
    vec = _mm256_permutevar8x32_epi32(vec, permutation_masks[mm]);
    /* return how many elements are greater than pv */
    return num_b_keys;
}

/* from Blacher https://github.com/simd-sorting/fast-and-robust/blob/master/avx2_sort_demo/avx2sort.h */
inline int calc_min(__m256i vec) { /* minimum of 8 int */
    auto perm_mask = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    vec = _mm256_min_epi32(vec, _mm256_permutevar8x32_epi32(vec, perm_mask));
    vec = _mm256_min_epi32(vec, _mm256_shuffle_epi32(vec, 0b10110001));
    vec = _mm256_min_epi32(vec, _mm256_shuffle_epi32(vec, 0b01001110));
    return _mm256_extract_epi32(vec, 0);
}

inline int calc_max(__m256i vec) { /* maximum of 8 int */
    auto perm_mask = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    vec = _mm256_max_epi32(vec, _mm256_permutevar8x32_epi32(vec, perm_mask));
    vec = _mm256_max_epi32(vec, _mm256_shuffle_epi32(vec, 0b10110001));
    vec = _mm256_max_epi32(vec, _mm256_shuffle_epi32(vec, 0b01001110));
    return _mm256_extract_epi32(vec, 0);
}

        /* vectorized random number generator xoroshiro128+ */
#define VROTL(x, k) /* rotate each uint64_t value in vector */               \
        _mm256_or_si256(_mm256_slli_epi64((x),(k)),_mm256_srli_epi64((x),64-(k)))

        inline __m256i vnext(__m256i &s0, __m256i &s1) {
            s1 = _mm256_xor_si256(s0, s1); /* modify vectors s1 and s0 */
            s0 = _mm256_xor_si256(_mm256_xor_si256(VROTL(s0, 24), s1),
                                  _mm256_slli_epi64(s1, 16));
            s1 = VROTL(s1, 37);
            return _mm256_add_epi64(s0, s1);
        } /* return random vector */

        /* transform random numbers to the range between 0 and bound - 1 */
        inline __m256i rnd_epu32(__m256i rnd_vec, __m256i bound) {
            __m256i even = _mm256_srli_epi64(_mm256_mul_epu32(rnd_vec, bound), 32);
            __m256i odd = _mm256_mul_epu32(_mm256_srli_epi64(rnd_vec, 32), bound);
            return _mm256_blend_epi32(odd, even, 0b01010101);
        }

/* end vectorized helping functions
* *********************************/


/* Miscellaneous helping functions
 * *******************************/

/* swaps two blocks */
inline void swap_blocks(int32_t *arr, uint32_t B, uint32_t block_a, uint32_t block_b) {
    auto *temp = new int32_t[B]; /* temp array */
    std::copy(arr + block_a, arr + block_a + B, temp); /* a to temp */
    std::copy(arr + block_b, arr + block_b + B, arr + block_a); /* b to a */
    std::copy(temp, temp + B, arr + block_b); /* temp to b */
    delete[] temp;
}

/* sorts the blocks and runners */
inline void sort_blocks(std::vector<uint32_t> &blocks,
                        std::vector<uint32_t> &runner) {

    if (blocks.size() < 2) { return; } /* sorted */

    /* selection sort */
    for (size_t i = 0; i < blocks.size() - 1; ++i) {
        size_t index = i;
        for (size_t j = i + 1; j < blocks.size(); ++j) {
            if (blocks[j] < blocks[index]) { index = j; }
        }
        std::swap(blocks[i], blocks[index]);
        std::swap(runner[i], runner[index]);
    }
}

inline void counting_sort(int32_t *arr, uint32_t left, uint32_t right,
                          int32_t s_key, int32_t b_key) {

    uint32_t num_keys = b_key + 1 - s_key;
    /* array with size number_distinct keys */
    auto *count_arr = new uint32_t[num_keys];

    /* initialize with 0 */
    for (uint32_t i = 0; i < num_keys; ++i) { count_arr[i] = 0; }

    /* count how often x appears */
    for (uint32_t i = left; i <= right; ++i) { ++count_arr[arr[i] - s_key]; }

    uint32_t index = left; /* current pointer */
    for (uint32_t i = 0; i < num_keys; ++i) { /* for every key value */
        /* place the value how often it appeared */
        for (uint32_t j = 0; j < count_arr[i]; ++j) { arr[index++] = i + s_key; }
    }
    delete[] count_arr;
}

/* average of two integers without overflow
* http://aggregate.org/MAGIC/#Average%20of%20Integers */
inline int average(int a, int b) { return (a & b) + ((a ^ b) >> 1); }

/* from Blacher
 * https://github.com/simd-sorting/fast-and-robust/blob/master/avx2_sort_demo/avx2sort.h */
inline int get_pivot(int *arr, int left, int right) {
    auto bound = _mm256_set1_epi32(right - left + 1);
    auto l_vec = _mm256_set1_epi32(left);

    /* seeds for vectorized random number generator */
    auto s0 = _mm256_setr_epi64x(8265987198341093849, 3762817312854612374,
                                 1324281658759788278, 6214952190349879213);
    auto s1 = _mm256_setr_epi64x(2874178529384792648, 1257248936691237653,
                                 7874578921548791257, 1998265912745817298);
    s0 = _mm256_add_epi64(s0, _mm256_set1_epi64x(left));
    s1 = _mm256_sub_epi64(s1, _mm256_set1_epi64x(right));

    __m256i v[9];
    for (int i = 0; i < 9; ++i) { /* fill 9 vectors with random numbers */
        auto result = vnext(s0, s1); /* vector with 4 random uint64_t */
        result = rnd_epu32(result, bound); /* random numbers between 0 and bound - 1 */
        result = _mm256_add_epi32(result, l_vec); /* indices for arr */
        v[i] = _mm256_i32gather_epi32(arr, result, sizeof(uint32_t));
    }

    /* median network for 9 elements */
    COEX(v[0], v[1]);
    COEX(v[2], v[3]); /* step 1 */
    COEX(v[4], v[5]);
    COEX(v[6], v[7]);
    COEX(v[0], v[2]);
    COEX(v[1], v[3]); /* step 2 */
    COEX(v[4], v[6]);
    COEX(v[5], v[7]);
    COEX(v[0], v[4]);
    COEX(v[1], v[2]); /* step 3 */
    COEX(v[5], v[6]);
    COEX(v[3], v[7]);
    COEX(v[1], v[5]);
    COEX(v[2], v[6]); /* step 4 */
    COEX(v[3], v[5]);
    COEX(v[2], v[4]); /* step 5 */
    COEX(v[3], v[4]);                   /* step 6 */
    COEX(v[3], v[8]);                   /* step 7 */
    COEX(v[4], v[8]);                   /* step 8 */

    SORT_8(v[4]); /* sort the eight medians in v[4] */
    return average(_mm256_extract_epi32(v[4], 3), /* compute next pv */
                   _mm256_extract_epi32(v[4], 4));
}

/* end Miscellaneous helping functions
 * ***********************************/


/* neutralize_side function
* ************************/

inline int32_t neutralize_side_vectorized(int32_t *arr, uint32_t l_block,
              uint32_t r_block, uint32_t &l_runner, uint32_t &r_runner, uint32_t B,
              int32_t pv, __m256i &pv_vec, int32_t &s_key, __m256i &s_vec,
              int32_t &b_key, __m256i &b_vec) {

    uint32_t l_end = l_block + B; /* end left block */
    uint32_t r_end = r_block; /* end right block */

    if (l_runner <= l_end - 64 && r_runner >= r_end + 63) {
        /* simulate vectors with size of 64 */

        /* save 8 vectors from left */
        __m256i l_vec1 = LOAD_VECTOR(arr + l_runner + 0);
        __m256i l_vec2 = LOAD_VECTOR(arr + l_runner + 8);
        __m256i l_vec3 = LOAD_VECTOR(arr + l_runner + 16);
        __m256i l_vec4 = LOAD_VECTOR(arr + l_runner + 24);
        __m256i l_vec5 = LOAD_VECTOR(arr + l_runner + 32);
        __m256i l_vec6 = LOAD_VECTOR(arr + l_runner + 40);
        __m256i l_vec7 = LOAD_VECTOR(arr + l_runner + 48);
        __m256i l_vec8 = LOAD_VECTOR(arr + l_runner + 56);

        /* save 8 vectors from right */
        __m256i r_vec1 = LOAD_VECTOR(arr + r_runner - 63 + 0);
        __m256i r_vec2 = LOAD_VECTOR(arr + r_runner - 63 + 8);
        __m256i r_vec3 = LOAD_VECTOR(arr + r_runner - 63 + 16);
        __m256i r_vec4 = LOAD_VECTOR(arr + r_runner - 63 + 24);
        __m256i r_vec5 = LOAD_VECTOR(arr + r_runner - 63 + 32);
        __m256i r_vec6 = LOAD_VECTOR(arr + r_runner - 63 + 40);
        __m256i r_vec7 = LOAD_VECTOR(arr + r_runner - 63 + 48);
        __m256i r_vec8 = LOAD_VECTOR(arr + r_runner - 63 + 56);

        uint32_t l_load = l_runner + 64; /* left loading point  */
        uint32_t l_store = l_runner; /* left storing point */
        uint32_t r_load = r_runner - 127; /* right loading point */
        uint32_t r_store = r_runner - 7; /* right storing point */

        /* partition the left and right vectors in a temp array */
        uint32_t temp_l_store = 0; /* left storing point in temp array */
        uint32_t temp_r_store = 120; /* right storing point in temp array */
        auto *temp = new int32_t[128];

        /* partition the left vectors */
        uint32_t num_b_keys1, num_b_keys2, num_b_keys3, num_b_keys4;
        uint32_t num_b_keys5, num_b_keys6, num_b_keys7, num_b_keys8;
        num_b_keys1 = _internal::partition_vec(l_vec1, pv_vec, s_vec, b_vec);
        num_b_keys2 = _internal::partition_vec(l_vec2, pv_vec, s_vec, b_vec);
        num_b_keys3 = _internal::partition_vec(l_vec3, pv_vec, s_vec, b_vec);
        num_b_keys4 = _internal::partition_vec(l_vec4, pv_vec, s_vec, b_vec);
        num_b_keys5 = _internal::partition_vec(l_vec5, pv_vec, s_vec, b_vec);
        num_b_keys6 = _internal::partition_vec(l_vec6, pv_vec, s_vec, b_vec);
        num_b_keys7 = _internal::partition_vec(l_vec7, pv_vec, s_vec, b_vec);
        num_b_keys8 = _internal::partition_vec(l_vec8, pv_vec, s_vec, b_vec);

        /* save the left vectors to the left side of temp */
        STORE_VECTOR(temp + temp_l_store, l_vec1);
        temp_l_store += 8 - num_b_keys1;
        STORE_VECTOR(temp + temp_l_store, l_vec2);
        temp_l_store += 8 - num_b_keys2;
        STORE_VECTOR(temp + temp_l_store, l_vec3);
        temp_l_store += 8 - num_b_keys3;
        STORE_VECTOR(temp + temp_l_store, l_vec4);
        temp_l_store += 8 - num_b_keys4;
        STORE_VECTOR(temp + temp_l_store, l_vec5);
        temp_l_store += 8 - num_b_keys5;
        STORE_VECTOR(temp + temp_l_store, l_vec6);
        temp_l_store += 8 - num_b_keys6;
        STORE_VECTOR(temp + temp_l_store, l_vec7);
        temp_l_store += 8 - num_b_keys7;
        STORE_VECTOR(temp + temp_l_store, l_vec8);
        temp_l_store += 8 - num_b_keys8;

        /* save the left vectors to the right side of temp */
        STORE_VECTOR(temp + temp_r_store, l_vec1);
        temp_r_store -= num_b_keys1;
        STORE_VECTOR(temp + temp_r_store, l_vec2);
        temp_r_store -= num_b_keys2;
        STORE_VECTOR(temp + temp_r_store, l_vec3);
        temp_r_store -= num_b_keys3;
        STORE_VECTOR(temp + temp_r_store, l_vec4);
        temp_r_store -= num_b_keys4;
        STORE_VECTOR(temp + temp_r_store, l_vec5);
        temp_r_store -= num_b_keys5;
        STORE_VECTOR(temp + temp_r_store, l_vec6);
        temp_r_store -= num_b_keys6;
        STORE_VECTOR(temp + temp_r_store, l_vec7);
        temp_r_store -= num_b_keys7;
        STORE_VECTOR(temp + temp_r_store, l_vec8);
        temp_r_store -= num_b_keys8;

        /* partition the right vectors */
        num_b_keys1 = _internal::partition_vec(r_vec1, pv_vec, s_vec, b_vec);
        num_b_keys2 = _internal::partition_vec(r_vec2, pv_vec, s_vec, b_vec);
        num_b_keys3 = _internal::partition_vec(r_vec3, pv_vec, s_vec, b_vec);
        num_b_keys4 = _internal::partition_vec(r_vec4, pv_vec, s_vec, b_vec);
        num_b_keys5 = _internal::partition_vec(r_vec5, pv_vec, s_vec, b_vec);
        num_b_keys6 = _internal::partition_vec(r_vec6, pv_vec, s_vec, b_vec);
        num_b_keys7 = _internal::partition_vec(r_vec7, pv_vec, s_vec, b_vec);
        num_b_keys8 = _internal::partition_vec(r_vec8, pv_vec, s_vec, b_vec);

        /* save the right vectors to the left side of temp */
        STORE_VECTOR(temp + temp_l_store, r_vec1);
        temp_l_store += 8 - num_b_keys1;
        STORE_VECTOR(temp + temp_l_store, r_vec2);
        temp_l_store += 8 - num_b_keys2;
        STORE_VECTOR(temp + temp_l_store, r_vec3);
        temp_l_store += 8 - num_b_keys3;
        STORE_VECTOR(temp + temp_l_store, r_vec4);
        temp_l_store += 8 - num_b_keys4;
        STORE_VECTOR(temp + temp_l_store, r_vec5);
        temp_l_store += 8 - num_b_keys5;
        STORE_VECTOR(temp + temp_l_store, r_vec6);
        temp_l_store += 8 - num_b_keys6;
        STORE_VECTOR(temp + temp_l_store, r_vec7);
        temp_l_store += 8 - num_b_keys7;
        STORE_VECTOR(temp + temp_l_store, r_vec8);
        temp_l_store += 8 - num_b_keys8;

        /* save the right vectors to the right side of temp */
        STORE_VECTOR(temp + temp_r_store, r_vec1);
        temp_r_store -= num_b_keys1;
        STORE_VECTOR(temp + temp_r_store, r_vec2);
        temp_r_store -= num_b_keys2;
        STORE_VECTOR(temp + temp_r_store, r_vec3);
        temp_r_store -= num_b_keys3;
        STORE_VECTOR(temp + temp_r_store, r_vec4);
        temp_r_store -= num_b_keys4;
        STORE_VECTOR(temp + temp_r_store, r_vec5);
        temp_r_store -= num_b_keys5;
        STORE_VECTOR(temp + temp_r_store, r_vec6);
        temp_r_store -= num_b_keys6;
        STORE_VECTOR(temp + temp_r_store, r_vec7);
        temp_r_store -= num_b_keys7;
        STORE_VECTOR(temp + temp_r_store, r_vec8);

        /* while we can load 8 vectors from left and from right */
        while (l_load <= l_end - 64 && r_load >= r_end) {
            /* variables to save vectors */
            __m256i vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8;

            /* load from the side where less keys are loaded */
            if (r_store - r_load - 56 < l_load - l_store) {
                /* load from the right */
                vec1 = LOAD_VECTOR(arr + r_load + 0);
                vec2 = LOAD_VECTOR(arr + r_load + 8);
                vec3 = LOAD_VECTOR(arr + r_load + 16);
                vec4 = LOAD_VECTOR(arr + r_load + 24);
                vec5 = LOAD_VECTOR(arr + r_load + 32);
                vec6 = LOAD_VECTOR(arr + r_load + 40);
                vec7 = LOAD_VECTOR(arr + r_load + 48);
                vec8 = LOAD_VECTOR(arr + r_load + 56);
                r_load -= 64;
            } else {
                /* load from the left */
                vec1 = LOAD_VECTOR(arr + l_load + 0);
                vec2 = LOAD_VECTOR(arr + l_load + 8);
                vec3 = LOAD_VECTOR(arr + l_load + 16);
                vec4 = LOAD_VECTOR(arr + l_load + 24);
                vec5 = LOAD_VECTOR(arr + l_load + 32);
                vec6 = LOAD_VECTOR(arr + l_load + 40);
                vec7 = LOAD_VECTOR(arr + l_load + 48);
                vec8 = LOAD_VECTOR(arr + l_load + 56);
                l_load += 64;
            }

            /* partition the current_vectors */
            num_b_keys1 = _internal::partition_vec(vec1, pv_vec, s_vec, b_vec);
            num_b_keys2 = _internal::partition_vec(vec2, pv_vec, s_vec, b_vec);
            num_b_keys3 = _internal::partition_vec(vec3, pv_vec, s_vec, b_vec);
            num_b_keys4 = _internal::partition_vec(vec4, pv_vec, s_vec, b_vec);
            num_b_keys5 = _internal::partition_vec(vec5, pv_vec, s_vec, b_vec);
            num_b_keys6 = _internal::partition_vec(vec6, pv_vec, s_vec, b_vec);
            num_b_keys7 = _internal::partition_vec(vec7, pv_vec, s_vec, b_vec);
            num_b_keys8 = _internal::partition_vec(vec8, pv_vec, s_vec, b_vec);

            /* save the current vectors to the left */
            STORE_VECTOR(arr + l_store, vec1);
            l_store += 8 - num_b_keys1;
            STORE_VECTOR(arr + l_store, vec2);
            l_store += 8 - num_b_keys2;
            STORE_VECTOR(arr + l_store, vec3);
            l_store += 8 - num_b_keys3;
            STORE_VECTOR(arr + l_store, vec4);
            l_store += 8 - num_b_keys4;
            STORE_VECTOR(arr + l_store, vec5);
            l_store += 8 - num_b_keys5;
            STORE_VECTOR(arr + l_store, vec6);
            l_store += 8 - num_b_keys6;
            STORE_VECTOR(arr + l_store, vec7);
            l_store += 8 - num_b_keys7;
            STORE_VECTOR(arr + l_store, vec8);
            l_store += 8 - num_b_keys8;

            /* save the current vector to the right */
            STORE_VECTOR(arr + r_store, vec1);
            r_store -= num_b_keys1;
            STORE_VECTOR(arr + r_store, vec2);
            r_store -= num_b_keys2;
            STORE_VECTOR(arr + r_store, vec3);
            r_store -= num_b_keys3;
            STORE_VECTOR(arr + r_store, vec4);
            r_store -= num_b_keys4;
            STORE_VECTOR(arr + r_store, vec5);
            r_store -= num_b_keys5;
            STORE_VECTOR(arr + r_store, vec6);
            r_store -= num_b_keys6;
            STORE_VECTOR(arr + r_store, vec7);
            r_store -= num_b_keys7;
            STORE_VECTOR(arr + r_store, vec8);
            r_store -= num_b_keys8;
        }

        r_load += 56; /* move loading point to actual position */

        /* 63 keys could be in front of right end which were not loaded */
        while (r_load >= r_end && l_load - l_store > 8) {
            /* load vector from right block */
            __m256i vec = LOAD_VECTOR(arr + r_load);
            r_load -= 8;

            /* partition the vector and save it */
            uint32_t num_b_keys;
            num_b_keys = _internal::partition_vec(vec, pv_vec, s_vec, b_vec);
            /* save vector to left and right block */
            STORE_VECTOR(arr + l_store, vec);
            STORE_VECTOR(arr + r_store, vec);
            l_store += 8 - num_b_keys;
            r_store -= num_b_keys;
        }

        /* 63 keys could be in front of left end which were not loaded */
        while (l_load <= l_end - 8 && r_store - r_load > 8) {
            /* load vector from left block */
            __m256i vec = LOAD_VECTOR(arr + l_load);
            l_load += 8;

            /* partition the vector and save it */
            uint32_t num_b_keys;
            num_b_keys = _internal::partition_vec(vec, pv_vec, s_vec, b_vec);
            /* save vector to left and right block */
            STORE_VECTOR(arr + l_store, vec);
            STORE_VECTOR(arr + r_store, vec);
            l_store += 8 - num_b_keys;
            r_store -= num_b_keys;
        }

        uint32_t space_l = l_load - l_store; /* keys can be overwritten left */
        uint32_t space_r = r_store - r_load; /* keys can be overwritten right */

        /* copy */
        std::copy(temp, temp + space_l, arr + l_store);
        std::copy(temp + space_l, temp + 128, arr + (r_store + 8 - space_r));
        delete[] temp;

        l_store += std::min(space_l, temp_l_store); /* update left store */
        r_store += (7 - std::min(space_r, 128 - temp_l_store)); /* update right store */

        l_runner = l_store; /* update left runner */
        r_runner = r_store; /* update right runner */

    } else if (l_runner <= l_end - 8 && r_runner >= r_end + 7) {
        /* use vectorized with 1 vector */
        __m256i l_vec = LOAD_VECTOR(arr + l_runner); /* vector from left block */
        __m256i r_vec = LOAD_VECTOR(arr + r_runner - 7); /* vector from right block */

        /* set load and store */
        uint32_t l_load = l_runner + 8;
        uint32_t l_store = l_runner;
        uint32_t r_load = r_runner - 15;
        uint32_t r_store = r_runner - 7;

        /* partition the left and right vector in a temp array */
        uint32_t temp_l_store = 0;
        uint32_t temp_r_store = 8;
        auto *temp = new int32_t[16];

        /* partition the left vector */
        uint32_t num_b_keys = _internal::partition_vec(l_vec, pv_vec, s_vec, b_vec);
        /* save to left and right */
        STORE_VECTOR(temp + temp_l_store, l_vec);
        STORE_VECTOR(temp + temp_r_store, l_vec);
        temp_l_store += 8 - num_b_keys;
        temp_r_store -= num_b_keys;

        /* partition the right vector */
        num_b_keys = _internal::partition_vec(r_vec, pv_vec, s_vec, b_vec);
        /* save to left and right */
        STORE_VECTOR(temp + temp_l_store, r_vec);
        STORE_VECTOR(temp + temp_r_store, r_vec);
        temp_l_store += 8 - num_b_keys;

        /* while we can load 1 vector from the left and from the right */
        while (l_load <= l_end - 8 && r_load >= r_end) {
            __m256i vec; /* variable to save vector from left or right */

            /* load from the side where less keys are stored */
            if (r_store - r_load < l_load - l_store) {
                /* load from the right block */
                vec = LOAD_VECTOR(arr + r_load);
                r_load -= 8;
            } else {
                /* load from the left block */
                vec = LOAD_VECTOR(arr + l_load);
                l_load += 8;
            }

            /* partition the current_vector */
            num_b_keys = _internal::partition_vec(vec, pv_vec, s_vec, b_vec);
            /* save to the left and right */
            STORE_VECTOR(arr + l_store, vec);
            STORE_VECTOR(arr + r_store, vec);
            l_store += 8 - num_b_keys;
            r_store -= num_b_keys;
        }

        uint32_t space_l = l_load - l_store; /* keys can be overwritten left */
        uint32_t space_r = r_store - r_load; /* keys can be overwritten right */

        /* copy */
        std::copy(temp, temp + space_l, arr + l_store);
        std::copy(temp + space_l, temp + 16, arr + (r_store + 8 - space_r));
        delete[] temp;

        l_store += std::min(space_l, temp_l_store); /* update left store */
        r_store += (7 - std::min(space_r, 16 - temp_l_store)); /* update right store */

        l_runner = l_store; /* update left runner */
        r_runner = r_store; /* update right runner */
    }

    /* neutralize the rest with the normal neutralizing function */
    while (true) {
        /* while not at end and not find key > pv */
        while (l_runner < l_end && arr[l_runner] <= pv) {
            s_key = std::min(s_key, arr[l_runner]);
            b_key = std::max(b_key, arr[l_runner]);
            l_runner++;
        }

        /* while not at end and not find key <= pv */
        while (r_runner >= r_end && arr[r_runner] > pv) {
            s_key = std::min(s_key, arr[r_runner]);
            b_key = std::max(b_key, arr[r_runner]);
            r_runner--;
        }

        /* if one arrived at end */
        if (l_runner == l_end || r_runner == r_end - 1) {
            /* return side */
            return (l_runner == l_end) + (r_runner == r_end - 1) * 2;
        }

        /* swap keys */
        s_key = std::min(s_key, arr[l_runner]);
        b_key = std::max(b_key, arr[l_runner]);
        s_key = std::min(s_key, arr[r_runner]);
        b_key = std::max(b_key, arr[r_runner]);
        std::swap(arr[l_runner++], arr[r_runner--]);
    }
}

/* end neutralize_side function
 * ****************************/


/* vectorized partitioning
 * **********************/

/* function from Blacher:
 * https://github.com/simd-sorting/fast-and-robust/blob/master/avx2_sort_demo/avx2sort.h */
inline uint32_t partition_vectorized_8(int32_t *arr, uint32_t left, uint32_t right,
                                       int32_t pv, int32_t &s_key, int32_t &b_key) {

    /* make array length divisible by eight, shortening the array */
    for (uint32_t i = (right - left) % 8; i > 0; --i) {
        s_key = std::min(s_key, arr[left]);
        b_key = std::max(b_key, arr[left]);
        if (arr[left] > pv) { std::swap(arr[left], arr[--right]); }
        else { ++left; }
    }

    if (left == right) return left; /* less than 8 elements in the array */

    auto pv_vec = _mm256_set1_epi32(pv); /* fill vector with pivot */
    auto s_vec = _mm256_set1_epi32(s_key); /* vector for s_key keys */
    auto b_vec = _mm256_set1_epi32(b_key); /* vector for b_key keys */

    if (right - left == 8) { /* if 8 elements left after shortening */
        auto v = LOAD_VECTOR(arr + left);
        int num_b_keys = _internal::partition_vec(v, pv_vec, s_vec, b_vec);
        STORE_VECTOR(arr + left, v);
        s_key = _internal::calc_min(s_vec);
        b_key = _internal::calc_max(b_vec);
        return left + (8 - num_b_keys);
    }

    /* first and last 8 values are partitioned at the end */
    auto l_vec = LOAD_VECTOR(arr + left); /* first 8 values */
    auto r_vec = LOAD_VECTOR(arr + (right - 8)); /* last 8 values  */
    /* store points of the vectors */
    uint32_t r_store = right - 8; /* right store point */
    uint32_t l_store = left; /* left store point */
    /* indices for loading the elements */
    left += 8; /* increase, because first 8 elements are cached */
    right -= 8; /* decrease, because last 8 elements are cached */

    while (right - left != 0) { /* partition 8 elements per iteration */
        __m256i vec; /* vector to be partitioned */
        /* if fewer elements are stored on the right side of the array,
         * then next elements are loaded from the right side,
         * otherwise from the left side */
        if ((r_store + 8) - right < left - l_store) {
            right -= 8;
            vec = LOAD_VECTOR(arr + right);
        } else {
            vec = LOAD_VECTOR(arr + left);
            left += 8;
        }
        /* partition the current vector and save it on both sides of the array */
        int num_b_keys = _internal::partition_vec(vec, pv_vec, s_vec, b_vec);;
        STORE_VECTOR(arr + l_store, vec);
        STORE_VECTOR(arr + r_store, vec);
        /* update store points */
        r_store -= num_b_keys;
        l_store += (8 - num_b_keys);
    }

    /* partition and save left vector */
    int num_b_keys = _internal::partition_vec(l_vec, pv_vec, s_vec, b_vec);
    STORE_VECTOR(arr + l_store, l_vec);
    STORE_VECTOR(arr + r_store, l_vec);
    l_store += (8 - num_b_keys);
    /* partition and save right vector */
    num_b_keys = _internal::partition_vec(r_vec, pv_vec, s_vec, b_vec);
    STORE_VECTOR(arr + l_store, r_vec);
    l_store += (8 - num_b_keys);

    s_key = _internal::calc_min(s_vec); /* determine s_key value in vector */
    b_key = _internal::calc_max(b_vec); /* determine b_key value in vector */
    return l_store;
}

/* simulate wider vector registers to speedup sorting
 * original function from Blacher: 
 * https://github.com/simd-sorting/fast-and-robust/blob/master/avx2_sort_demo/avx2sort.h */
inline uint32_t partition_vectorized_64(int32_t *arr, uint32_t left, uint32_t right,
                                        int32_t pv, int32_t &s_key, int32_t &b_key) {

    if (right + 1 - left < 129) {
        /* do not optimize if less than 129 elements */
        return partition_vectorized_8(arr, left, right + 1, pv, s_key, b_key);
    }

    /* make array length divisible by eight, shortening the array */
    for (uint32_t i = (right + 1 - left) % 8; i > 0; --i) {
        s_key = std::min(s_key, arr[left]);
        b_key = std::max(b_key, arr[left]);
        if (arr[left] > pv) { std::swap(arr[left], arr[right--]); }
        else { ++left; }
    }

    auto pv_vec = _mm256_set1_epi32(pv); /* fill vector with pivot */
    auto s_vec = _mm256_set1_epi32(s_key); /* vector for s_key key */
    auto b_vec = _mm256_set1_epi32(b_key); /* vector for b_key key */

    /* make array length divisible by 64, shortening the array */
    for (uint32_t i = ((right + 1 - left) % 64) / 8; i > 0; --i) {
        __m256i vec_L = LOAD_VECTOR(arr + left);
        __m256i compared = _mm256_cmpgt_epi32(vec_L, pv_vec);
        s_vec = _mm256_min_epi32(vec_L, s_vec);
        b_vec = _mm256_max_epi32(vec_L, b_vec);
        int mm = _mm256_movemask_ps(reinterpret_cast<__m256>(compared));
        int num_b_keys = _mm_popcnt_u32((mm));
        __m256i permuted =
                _mm256_permutevar8x32_epi32(vec_L, _internal::permutation_masks[mm]);

        /* this is a slower way to partition an array with vector instructions */
        __m256i blend_mask = _mm256_cmpgt_epi32(permuted, pv_vec);
        __m256i vec_R = LOAD_VECTOR(arr + right - 7);
        __m256i vec_L_new = _mm256_blendv_epi8(permuted, vec_R, blend_mask);
        __m256i vec_R_new = _mm256_blendv_epi8(vec_R, permuted, blend_mask);
        STORE_VECTOR(arr + left, vec_L_new);
        STORE_VECTOR(arr + right - 7, vec_R_new);
        left += (8 - num_b_keys);
        right -= num_b_keys;
    }

    /* buffer 8 vectors from both sides of the array */
    auto l_vec1 = LOAD_VECTOR(arr + left + 0);
    auto l_vec2 = LOAD_VECTOR(arr + left + 8);
    auto l_vec3 = LOAD_VECTOR(arr + left + 16);
    auto l_vec4 = LOAD_VECTOR(arr + left + 24);
    auto l_vec5 = LOAD_VECTOR(arr + left + 32);
    auto l_vec6 = LOAD_VECTOR(arr + left + 40);
    auto l_vec7 = LOAD_VECTOR(arr + left + 48);
    auto l_vec8 = LOAD_VECTOR(arr + left + 56);

    auto r_vec1 = LOAD_VECTOR(arr + right - 63 + 0);
    auto r_vec2 = LOAD_VECTOR(arr + right - 63 + 8);
    auto r_vec3 = LOAD_VECTOR(arr + right - 63 + 16);
    auto r_vec4 = LOAD_VECTOR(arr + right - 63 + 24);
    auto r_vec5 = LOAD_VECTOR(arr + right - 63 + 32);
    auto r_vec6 = LOAD_VECTOR(arr + right - 63 + 40);
    auto r_vec7 = LOAD_VECTOR(arr + right - 63 + 48);
    auto r_vec8 = LOAD_VECTOR(arr + right - 63 + 56);

    uint32_t r_load = right - 127; /* right load point */
    uint32_t l_load = left + 64; /* left load point */
    uint32_t r_store = right - 7; /* right store point */
    uint32_t l_store = left; /* left store point */

    while (l_load != r_load + 64) {
        /* partition 64 elements per iteration */
        __m256i vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8;

        /* if less elements are stored on the right side of the array,
         * then next 8 vectors load from the right side,
         * otherwise load from the left side */
        if (r_store - r_load - 56 < l_load - l_store) {
            vec1 = LOAD_VECTOR(arr + r_load + 0);
            vec2 = LOAD_VECTOR(arr + r_load + 8);
            vec3 = LOAD_VECTOR(arr + r_load + 16);
            vec4 = LOAD_VECTOR(arr + r_load + 24);
            vec5 = LOAD_VECTOR(arr + r_load + 32);
            vec6 = LOAD_VECTOR(arr + r_load + 40);
            vec7 = LOAD_VECTOR(arr + r_load + 48);
            vec8 = LOAD_VECTOR(arr + r_load + 56);
            r_load -= 64;
        } else {
            vec1 = LOAD_VECTOR(arr + l_load + 0);
            vec2 = LOAD_VECTOR(arr + l_load + 8);
            vec3 = LOAD_VECTOR(arr + l_load + 16);
            vec4 = LOAD_VECTOR(arr + l_load + 24);
            vec5 = LOAD_VECTOR(arr + l_load + 32);
            vec6 = LOAD_VECTOR(arr + l_load + 40);
            vec7 = LOAD_VECTOR(arr + l_load + 48);
            vec8 = LOAD_VECTOR(arr + l_load + 56);
            l_load += 64;
        }

        /* partition 8 vectors and store them on both sides of the array */
        uint32_t num_b_keys1, num_b_keys2, num_b_keys3, num_b_keys4;
        uint32_t num_b_keys5, num_b_keys6, num_b_keys7, num_b_keys8;
        num_b_keys1 = _internal::partition_vec(vec1, pv_vec, s_vec, b_vec);
        num_b_keys2 = _internal::partition_vec(vec2, pv_vec, s_vec, b_vec);
        num_b_keys3 = _internal::partition_vec(vec3, pv_vec, s_vec, b_vec);
        num_b_keys4 = _internal::partition_vec(vec4, pv_vec, s_vec, b_vec);
        num_b_keys5 = _internal::partition_vec(vec5, pv_vec, s_vec, b_vec);
        num_b_keys6 = _internal::partition_vec(vec6, pv_vec, s_vec, b_vec);
        num_b_keys7 = _internal::partition_vec(vec7, pv_vec, s_vec, b_vec);
        num_b_keys8 = _internal::partition_vec(vec8, pv_vec, s_vec, b_vec);

        /* store vectors to left */
        STORE_VECTOR(arr + l_store, vec1);
        l_store += (8 - num_b_keys1);
        STORE_VECTOR(arr + l_store, vec2);
        l_store += (8 - num_b_keys2);
        STORE_VECTOR(arr + l_store, vec3);
        l_store += (8 - num_b_keys3);
        STORE_VECTOR(arr + l_store, vec4);
        l_store += (8 - num_b_keys4);
        STORE_VECTOR(arr + l_store, vec5);
        l_store += (8 - num_b_keys5);
        STORE_VECTOR(arr + l_store, vec6);
        l_store += (8 - num_b_keys6);
        STORE_VECTOR(arr + l_store, vec7);
        l_store += (8 - num_b_keys7);
        STORE_VECTOR(arr + l_store, vec8);
        l_store += (8 - num_b_keys8);

        /* store vectors to right */
        STORE_VECTOR(arr + r_store, vec1);
        r_store -= num_b_keys1;
        STORE_VECTOR(arr + r_store, vec2);
        r_store -= num_b_keys2;
        STORE_VECTOR(arr + r_store, vec3);
        r_store -= num_b_keys3;
        STORE_VECTOR(arr + r_store, vec4);
        r_store -= num_b_keys4;
        STORE_VECTOR(arr + r_store, vec5);
        r_store -= num_b_keys5;
        STORE_VECTOR(arr + r_store, vec6);
        r_store -= num_b_keys6;
        STORE_VECTOR(arr + r_store, vec7);
        r_store -= num_b_keys7;
        STORE_VECTOR(arr + r_store, vec8);
        r_store -= num_b_keys8;
    }

    /* partition and store 8 vectors coming from the left side of the array */
    uint32_t num_b_keys1, num_b_keys2, num_b_keys3, num_b_keys4;
    uint32_t num_b_keys5, num_b_keys6, num_b_keys7, num_b_keys8;
    num_b_keys1 = _internal::partition_vec(l_vec1, pv_vec, s_vec, b_vec);
    num_b_keys2 = _internal::partition_vec(l_vec2, pv_vec, s_vec, b_vec);
    num_b_keys3 = _internal::partition_vec(l_vec3, pv_vec, s_vec, b_vec);
    num_b_keys4 = _internal::partition_vec(l_vec4, pv_vec, s_vec, b_vec);
    num_b_keys5 = _internal::partition_vec(l_vec5, pv_vec, s_vec, b_vec);
    num_b_keys6 = _internal::partition_vec(l_vec6, pv_vec, s_vec, b_vec);
    num_b_keys7 = _internal::partition_vec(l_vec7, pv_vec, s_vec, b_vec);
    num_b_keys8 = _internal::partition_vec(l_vec8, pv_vec, s_vec, b_vec);

    /* store left vectors to left */
    STORE_VECTOR(arr + l_store, l_vec1);
    l_store += (8 - num_b_keys1);
    STORE_VECTOR(arr + l_store, l_vec2);
    l_store += (8 - num_b_keys2);
    STORE_VECTOR(arr + l_store, l_vec3);
    l_store += (8 - num_b_keys3);
    STORE_VECTOR(arr + l_store, l_vec4);
    l_store += (8 - num_b_keys4);
    STORE_VECTOR(arr + l_store, l_vec5);
    l_store += (8 - num_b_keys5);
    STORE_VECTOR(arr + l_store, l_vec6);
    l_store += (8 - num_b_keys6);
    STORE_VECTOR(arr + l_store, l_vec7);
    l_store += (8 - num_b_keys7);
    STORE_VECTOR(arr + l_store, l_vec8);
    l_store += (8 - num_b_keys8);

    /* store left vectors to right */
    STORE_VECTOR(arr + r_store, l_vec1);
    r_store -= num_b_keys1;
    STORE_VECTOR(arr + r_store, l_vec2);
    r_store -= num_b_keys2;
    STORE_VECTOR(arr + r_store, l_vec3);
    r_store -= num_b_keys3;
    STORE_VECTOR(arr + r_store, l_vec4);
    r_store -= num_b_keys4;
    STORE_VECTOR(arr + r_store, l_vec5);
    r_store -= num_b_keys5;
    STORE_VECTOR(arr + r_store, l_vec6);
    r_store -= num_b_keys6;
    STORE_VECTOR(arr + r_store, l_vec7);
    r_store -= num_b_keys7;
    STORE_VECTOR(arr + r_store, l_vec8);
    r_store -= num_b_keys8;

    /* partition and store 8 vectors coming from the right side of the array */
    num_b_keys1 = _internal::partition_vec(r_vec1, pv_vec, s_vec, b_vec);
    num_b_keys2 = _internal::partition_vec(r_vec2, pv_vec, s_vec, b_vec);
    num_b_keys3 = _internal::partition_vec(r_vec3, pv_vec, s_vec, b_vec);
    num_b_keys4 = _internal::partition_vec(r_vec4, pv_vec, s_vec, b_vec);
    num_b_keys5 = _internal::partition_vec(r_vec5, pv_vec, s_vec, b_vec);
    num_b_keys6 = _internal::partition_vec(r_vec6, pv_vec, s_vec, b_vec);
    num_b_keys7 = _internal::partition_vec(r_vec7, pv_vec, s_vec, b_vec);
    num_b_keys8 = _internal::partition_vec(r_vec8, pv_vec, s_vec, b_vec);

    /* store right vectors to left */
    STORE_VECTOR(arr + l_store, r_vec1);
    l_store += (8 - num_b_keys1);
    STORE_VECTOR(arr + l_store, r_vec2);
    l_store += (8 - num_b_keys2);
    STORE_VECTOR(arr + l_store, r_vec3);
    l_store += (8 - num_b_keys3);
    STORE_VECTOR(arr + l_store, r_vec4);
    l_store += (8 - num_b_keys4);
    STORE_VECTOR(arr + l_store, r_vec5);
    l_store += (8 - num_b_keys5);
    STORE_VECTOR(arr + l_store, r_vec6);
    l_store += (8 - num_b_keys6);
    STORE_VECTOR(arr + l_store, r_vec7);
    l_store += (8 - num_b_keys7);
    STORE_VECTOR(arr + l_store, r_vec8);
    l_store += (8 - num_b_keys8);

    /* store right vectors to left */
    STORE_VECTOR(arr + r_store, r_vec1);
    r_store -= num_b_keys1;
    STORE_VECTOR(arr + r_store, r_vec2);
    r_store -= num_b_keys2;
    STORE_VECTOR(arr + r_store, r_vec3);
    r_store -= num_b_keys3;
    STORE_VECTOR(arr + r_store, r_vec4);
    r_store -= num_b_keys4;
    STORE_VECTOR(arr + r_store, r_vec5);
    r_store -= num_b_keys5;
    STORE_VECTOR(arr + r_store, r_vec6);
    r_store -= num_b_keys6;
    STORE_VECTOR(arr + r_store, r_vec7);
    r_store -= num_b_keys7;
    STORE_VECTOR(arr + r_store, r_vec8);

    /* determine s_key and b_key key */
    s_key = _internal::calc_min(s_vec);
    b_key = _internal::calc_max(b_vec);
    return l_store;
}

/* end vectorized partitioning
 * ***************************/

/* vectorized core functions
 *  ************************/

/* core for vectorized sort */
inline void vectorized_sort_core(int32_t *arr, uint32_t left, uint32_t right, 
                                 bool use_avg, int32_t avg,uint32_t th_counting) {

    if (right + 1 - left < 513) { /* sort in sorting network */
        __m256i buffer[66];
        int *buff = reinterpret_cast<int *>(buffer);
        _internal::sort_int_sorting_network(arr + left, buff, right - left + 1);
        return;
    }

    int32_t s_key = INT32_MAX, b_key = INT32_MIN; /* s_key, b_key*/
    uint32_t sp; /* split point */
    int32_t pv; /* pivot */
    double ratio;

    /* pivot strategy */
    pv = use_avg ? avg : _internal::get_pivot(arr, left, right);
    /* partition */
    sp = partition_vectorized_64(arr, left, right, pv, s_key, b_key);
    /* ratio small to bigger subarray */
    ratio = std::min((right + 1) - sp, sp - left) / (double) ((right + 1) - left);
    /* change pivot strategy */
    if (ratio < 0.2) { use_avg = !use_avg; }

    if (s_key != pv) { /* different keys in lower array */
        uint64_t num_keys = (pv + 1) - s_key; /* number of distinct keys */
        if (num_keys < th_counting) {
            /* small amount of distinct keys -> counting sort */
            counting_sort(arr, left, sp - 1, s_key, pv);
        } else {
            /* large amount of distinct keys -> Quicksort */
            avg = _internal::average(s_key, pv);
            vectorized_sort_core(arr, left, sp - 1, use_avg, avg, th_counting);
        }
    }

    if (b_key != pv + 1) { /* different keys in upper array */
        uint64_t num_keys = (b_key + 1) - pv; /* number of distinct keys */
        if (num_keys < th_counting) {
            /* small amount of distinct keys -> counting sort */
            counting_sort(arr, sp, right, pv, b_key);
        } else {
            /* large amount of distinct keys -> Quicksort */
            avg = _internal::average(b_key, pv);
            vectorized_sort_core(arr, sp, right, use_avg, avg, th_counting);
        }
    }
}

/* core for parallel Divide and Conquer vectorized sort */
inline void dac_vectorized_sort_core(int32_t *arr, uint32_t left, uint32_t right, uint32_t depth, uint32_t max_depth, bool use_avg, int32_t avg, uint32_t th_counting) {
    if (right + 1 - left < 513) { /* sort in sorting network */
        __m256i buffer[66];
        int *buff = reinterpret_cast<int *>(buffer);
        _internal::sort_int_sorting_network(arr + left, buff, right - left + 1);
        return;
    }

    int32_t s_key = INT32_MAX, b_key = INT32_MIN; /* s_key, b_key*/
    uint32_t sp; /* split point */
    int32_t pv; /* pivot */
    double ratio;

    /* pivot strategy */
    pv = use_avg ? avg : _internal::get_pivot(arr, left, right);
    /* partition */
    sp = partition_vectorized_64(arr, left, right, pv, s_key, b_key);
    /* ratio small to bigger subarray */
    ratio = std::min((right + 1) - sp, sp - left) / (double) ((right + 1) - left);
    /* change pivot strategy */
    if (ratio < 0.2) { use_avg = !use_avg; }

    if (s_key != pv) { /* different keys in lower part */
        uint64_t num_keys = (pv + 1) - s_key;
        if (num_keys < th_counting) {
            /* small amount of distinct keys -> counting sort */
#pragma omp task firstprivate(arr, left, sp, s_key, pv)
            { counting_sort(arr, left, sp - 1, s_key, pv); }
        } else {
            /* large amount of distinct keys -> Quicksort */
            avg = _internal::average(s_key, pv);
            if (depth <= max_depth) {
                /* not at max_depth -> parallel core */
#pragma omp task firstprivate(arr, left, sp, depth, max_depth, use_avg, avg, th_counting)
                { dac_vectorized_sort_core(arr, left, sp - 1, depth + 1, max_depth, use_avg, avg, th_counting); }
            } else {
                /* at max_depth -> serial core */
#pragma omp task firstprivate(arr, left, sp, use_avg, avg, th_counting)
                { vectorized_sort_core(arr, left, sp - 1, use_avg, avg, th_counting); }
            }
        }
    }

    if (b_key != pv + 1) { /* different keys */
        uint64_t num_keys = (b_key + 1) - pv;
        if (num_keys < th_counting) {
            /* small amount of distinct keys -> counting sort */
#pragma omp task firstprivate(arr, sp, right, pv, b_key)
            { counting_sort(arr, sp, right, pv, b_key); }
        } else {
            /* large amount of distinct keys -> Quicksort */
            avg = _internal::average(pv, b_key);
            if (depth <= max_depth) {
                /* not at max_depth -> parallel core */
#pragma omp task firstprivate(arr, sp, right, depth, max_depth, use_avg, avg, th_counting)
                {
                    dac_vectorized_sort_core(arr, sp, right, depth + 1,
                                             max_depth, use_avg, avg, th_counting);
                }
            } else {
                /* at max_depth -> serial core */
#pragma omp task firstprivate(arr, sp, right, use_avg, avg, th_counting)
                { vectorized_sort_core(arr, sp, right, use_avg, avg, th_counting); }
            }
        }
    }
}

/* core for vectorized quickselect */
inline void vectorized_quickselect_core(int32_t *arr, uint32_t left, uint32_t right,
                                        uint32_t k, bool use_avg, int32_t avg) {
    
    if (right + 1 - left < 257) {
        /* for few elements use C++'s nth_element */
        std::nth_element(arr + left, arr + k, arr + right + 1);
        return;
    }
    
    int32_t s_key = INT32_MAX, b_key = INT32_MIN; /* s_key, b_key*/
    uint32_t sp; /* split point */
    int32_t pv; /* pivot */
    double ratio;

    /* pivot strategy */
    pv = use_avg ? avg : _internal::get_pivot(arr, left, right);
    /* partition */
    sp = partition_vectorized_64(arr, left, right, pv, s_key, b_key);
    /* ratio small to bigger subarray */
    ratio = std::min((right + 1) - sp, sp - left) / (double) ((right + 1) - left);
    /* change pivot strategy */
    if (ratio < 0.2) { use_avg = !use_avg; }

    if (k < sp) { /* k is in the lower sub array */
        if (pv != s_key) { /* different values in lower sub array */
            avg = _internal::average(s_key, pv);
            vectorized_quickselect_core(arr, left, sp - 1, k, use_avg, avg);
        }
    } else { /* k is in upper sub array */
        if (pv + 1 != b_key) { /* different values in upper sub array */
            avg = _internal::average(pv, b_key);
            vectorized_quickselect_core(arr, sp, right, k, use_avg, avg);
        }
    }
}

/* end vectorized core functions
 *  ************************/

/* parallel vectorized partition
 * *****************************/

/* parallel partitioning phase */
inline void parallel_partition_phase(int32_t *arr, uint32_t left, uint32_t right,
            uint32_t B, int32_t pv, uint32_t n_threads, std::vector<uint32_t> &blocks,
            std::vector<uint32_t> &runner, uint32_t &l_max, uint32_t &r_min,
            int32_t &s_key, int32_t &b_key) {
    
    /* the left 32 bits is the number of left keys taken
     * the right 32 bits is the number of right keys taken */
    uint64_t blocks_to_use = B;

    /* aligned vectors for better cache use */
    alignas(64) std::vector<uint32_t> vec_l_max(n_threads * 16);
    alignas(64) std::vector<uint32_t> vec_r_min(n_threads * 16);
    alignas(64) std::vector<int32_t> vec_s_key(n_threads * 16);
    alignas(64) std::vector<int32_t> vec_b_key(n_threads * 16);

    /* start n_thread tasks, with unique id */
    for (uint32_t task_id = 0; task_id < n_threads; ++task_id) {
#pragma omp task shared(blocks, runner, blocks_to_use, vec_l_max, vec_r_min, vec_s_key, vec_b_key) firstprivate(task_id, arr, left, right, B, pv, s_key, b_key)
        {
            uint32_t local_l_max = left; /* left_max block */
            uint32_t local_r_min = right + 1 - B; /* right_min block */

            int32_t local_s_key = INT32_MAX; /* save smallest key */
            int32_t local_b_key = INT32_MIN; /* save biggest key */

            __m256i pv_vec = CONST_VECTOR(pv); /* pivot vector */
            __m256i s_vec = CONST_VECTOR(INT32_MAX); /* vector smallest key */
            __m256i b_vec = CONST_VECTOR(INT32_MIN); /* vector biggest keys */

            /* left 32 bits hold left block, right 32 bit hold right block */
            uint64_t local_blocks_to_use = 0;
            uint32_t l_block = 0;
            uint32_t r_block = 0;
            uint32_t l_runner = 0; /* runner for left block */
            uint32_t r_runner = 0; /* runner for right block */
            bool l_block_valid = false; /* shows if left block is valid */
            bool r_block_valid = false; /* shows if right block is valid */
            
#pragma omp atomic capture /* get a block from left and right */
            {
                local_blocks_to_use = blocks_to_use; /* read current blocks */
                blocks_to_use += ((uint64_t) B << 32) + B; /* 2 blocks were taken */
            }

            /* determine left and right block */
            l_block = left + (local_blocks_to_use >> 32);
            r_block = (right + 1) - (local_blocks_to_use & UINT32_MAX);

            l_runner = l_block; /* set left runner to start */
            r_runner = r_block + B - 1; /* set right runner to end */

            /* check if right block caused underflow */
            bool no_overflow = right + 1 > (local_blocks_to_use & UINT32_MAX);
            
            /* check if the chosen blocks are valid
             * if the l_block ends in the r_block, than r_block is invalid or
             * right+1 is smaller than what we want to subtract 
             * -> (would cause overflow)
             * if the l_block starts further right than the r_block,
             * than both blocks are invalid */
            r_block_valid = l_block + B - 1 < r_block && no_overflow;
            l_block_valid = l_block <= r_block && no_overflow;

            /* loop until not valid */
            while (l_block_valid && r_block_valid) {
                /* neutralize the blocks */
                int32_t neut_block 
                    = neutralize_side_vectorized(arr, l_block, r_block, l_runner,
                                                 r_runner, B, pv, pv_vec, local_s_key,
                                                 s_vec, local_b_key, b_vec);

                if (neut_block == 1 || neut_block == 3) { /* LEFT or BOTH */
                    local_l_max = l_block; /* update max */

#pragma omp atomic capture /* get a new l_block */
                    {
                        local_blocks_to_use = blocks_to_use;
                        blocks_to_use += (uint64_t) B << 32; /* add B on left 32 bits */
                    }

                    /* determine new left block */
                    l_block = left + (local_blocks_to_use >> 32);
                    /* determine temp right block */
                    uint32_t temp_r_block = (right + 1) - 
                            (local_blocks_to_use & UINT32_MAX);

                    l_runner = l_block; /* reset runner */

                    /* check if the chosen l_block is valid */
                    no_overflow = (right + 1) > (local_blocks_to_use & UINT32_MAX);
                    /* can write in temp_right_block */
                    l_block_valid = (l_block <= temp_r_block) && no_overflow;
                }

                if (neut_block == 2 || neut_block == 3) { /* RIGHT or BOTH */
                    local_r_min = r_block; /* update min */

#pragma omp atomic capture /* get a new r_block */
                    {
                        local_blocks_to_use = blocks_to_use;
                        blocks_to_use += B; /* add B on right 32 bits */
                    }

                    /* determine temp_left_block */
                    uint32_t temp_l_block = left + (local_blocks_to_use >> 32);
                    /* determine new right block */
                    r_block = (right + 1) - (local_blocks_to_use & UINT32_MAX);

                    r_runner = r_block + B - 1; /* reset runner */

                    /* check if the chosen r_block is valid */
                    no_overflow = (right + 1) > (local_blocks_to_use & UINT32_MAX);
                    /* can write in temp_left_block */
                    r_block_valid = (temp_l_block <= r_block) && no_overflow;
                }
            } /* end while loop */

            /* update smallest and biggest key from vector */
            local_s_key = std::min(local_s_key, _internal::calc_min(s_vec));
            local_b_key = std::max(local_b_key, _internal::calc_max(b_vec));

            /* put valid block in blocks and runner */
            if (l_block_valid) {
                blocks[task_id] = l_block;
                runner[task_id] = l_runner;
            } else if (r_block_valid) {
                blocks[task_id] = r_block;
                runner[task_id] = r_runner;
            }

            vec_l_max[task_id * 16] = local_l_max; /* insert left_max */
            vec_r_min[task_id * 16] = local_r_min; /* insert right_min */
            vec_s_key[task_id * 16] = local_s_key; /* insert smallest_key */
            vec_b_key[task_id * 16] = local_b_key; /* insert biggest_key */
        } /* end task */
    } /* end for loop */
#pragma omp taskwait /* wait for all tasks to end */

    for (size_t i = 0; i < n_threads * 16; i += 16) { /* determine global l_max */
        l_max = std::max(l_max, vec_l_max[i]); }
    for (size_t i = 0; i < n_threads * 16; i += 16) { /* determine global r_min */
        r_min = std::min(r_min, vec_r_min[i]); }
    for (size_t i = 0; i < n_threads * 16; i += 16) { /* determine global s_key */
        s_key = std::min(s_key, vec_s_key[i]); }
    for (size_t i = 0; i < n_threads * 16; i += 16) { /* determine global b_key */
        b_key = std::max(b_key, vec_b_key[i]); }
}

inline uint32_t single_partition_phase(int32_t *arr, uint32_t left, uint32_t right,
                uint32_t B, int32_t pv, uint32_t n_threads,
                std::deque<uint32_t> &blocks, std::deque<uint32_t> &runner,
                uint32_t &l_max, uint32_t &r_min, int32_t &s_key, int32_t &b_key) {

    __m256i pv_vec = CONST_VECTOR(pv); /* pivot vector */
    __m256i s_vec = CONST_VECTOR(s_key); /* vector for smallest key */
    __m256i b_vec = CONST_VECTOR(b_key); /* vector for biggest key */

    /* while block in front of l_max and block after r_min */
    while (blocks.front() <= l_max && blocks.back() >= r_min) {
        int32_t neut_block =
                neutralize_side_vectorized(arr, blocks.front(), blocks.back(),
                                           runner.front(), runner.back(), B,
                                           pv, pv_vec, s_key, s_vec,
                                           b_key, b_vec);

        if (neut_block == 1 || neut_block == 3) { /* LEFT or BOTH */
            blocks.pop_front(); /* remove front block */
            runner.pop_front(); /* and runner */
        }

        if (neut_block == 2 || neut_block == 3) { /* RIGHT or BOTH */
            blocks.pop_back(); /* remove back block */
            runner.pop_back(); /* and runner */
        }
    }
    /* no remaining blocks in front l_max or after r_min */

    if (blocks.front() == left && l_max == left) {
        /* no block was ever LEFT neutralized */
        s_key = std::min(s_key, _internal::calc_min(s_vec)); /* update smallest */
        b_key = std::max(b_key, _internal::calc_max(b_vec)); /* update biggest */
        /* partition last chunk */
        return partition_vectorized_64(arr, left, r_min - 1, pv, s_key, b_key);
    }

    if (blocks.back() == right + 1 - B && r_min == right + 1 - B) {
        /* no block was ever RIGHT neutralized */
        s_key = std::min(s_key, _internal::calc_min(s_vec)); /* update smallest */
        b_key = std::max(b_key, _internal::calc_max(b_vec)); /* update biggest */
        /* partition last chunk */
        return partition_vectorized_64(arr, l_max + B, right, pv, s_key, b_key);
    }

    while (true) {
        if (blocks.empty()) { /* left and right part of array completely neutralized */
            s_key = std::min(s_key, _internal::calc_min(s_vec)); /* update s_key */
            b_key = std::max(b_key, _internal::calc_max(b_vec)); /* update b_key */
            /* partition last chunk */

            return partition_vectorized_64(arr, l_max + B, r_min - 1, pv, s_key, b_key);
        } else if (blocks.front() > l_max) { /* only blocks after r_min */
            if (l_max + 2 * B > r_min) { /* not enough space */
                /* swap neutralized and not neutralized block */
                swap_blocks(arr, B, r_min, blocks.back());

                /* remove back block and runner */
                blocks.pop_back();
                runner.pop_back();

                /* search new r_min */
                r_min += B;
                while (std::find(blocks.begin(), blocks.end(), r_min) != blocks.end()) {
                    r_min += B;
                }

                /* delete blocks in front of r_min */
                for (int i = blocks.size() - 1; i >= 0; i--) {
                    if (blocks[i] < r_min) {
                        /* delete block and runner */
                        blocks.erase(blocks.begin() + i);
                        runner.erase(runner.begin() + i);
                    }
                }
            } else {
                /* neutralize two blocks */
                uint32_t l_runner = l_max + B; /* l_runner at block start */
                int32_t neut_block =
                        neutralize_side_vectorized(arr, l_max + B, blocks.back(),
                                                   l_runner, runner.back(), B,
                                                   pv, pv_vec, s_key, s_vec,
                                                   b_key, b_vec);

                if (neut_block == 1 || neut_block == 3) { /* LEFT or BOTH */
                    l_max += B; /* move l_max one block further */
                }

                if (neut_block == 2 || neut_block == 3) { /* RIGHT or BOTH */
                    blocks.pop_back(); /* remove back block */
                    runner.pop_back(); /* and runner */
                }
            }
        } else { /* only blocks in front of l_max */
            if (l_max + 2 * B > r_min) { /* not enough space */

                /* swap neutralized and not neutralized block */
                swap_blocks(arr, B, blocks.front(), l_max);

                /* remove front block and runner */
                blocks.pop_front();
                runner.pop_front();

                /* search new l_max */
                l_max -= B;
                while (std::find(blocks.begin(), blocks.end(), l_max) != blocks.end()) {
                    l_max -= B;
                }

                /* delete blocks after l_max */
                for (int i = blocks.size() - 1; i >= 0; i--) {
                    if (blocks[i] > l_max) {
                        /* remove block and runner */
                        blocks.erase(blocks.begin() + i);
                        runner.erase(runner.begin() + i);
                    }
                }
            } else { /* neutralize two blocks */
                uint32_t r_runner = r_min - 1; /* r_runner at block end */
                int32_t neut_block =
                        neutralize_side_vectorized(arr, blocks.front(), r_min - B,
                                                   runner.front(), r_runner, B,
                                                   pv, pv_vec, s_key, s_vec,
                                                   b_key, b_vec);

                if (neut_block == 1 || neut_block == 3) { /* LEFT or BOTH */
                    blocks.pop_front(); /* remove front block */
                    runner.pop_front(); /* and runner */
                }

                if (neut_block == 2 || neut_block == 3) { /* RIGHT or BOTH */
                    r_min -= B; /* move right_min block one further */
                }
            }
        }
    }
}

/* core for partitioning */
inline uint32_t partition_core(int32_t *arr, uint32_t left, uint32_t right, uint32_t B,
                               int32_t pv, uint32_t n_threads, int32_t &s_key,
                               int32_t &b_key) {

    if (n_threads == 1 || right + 1 - left < 2 * B) {
        /* one thread -> use non parallel vectorized partition */
        return partition_vectorized_64(arr, left, right, pv, s_key, b_key);
    }

    uint32_t l_max = left; /* holds left_max */
    uint32_t r_min = right + 1 - B; /* holds right_min */
    /* std::vector for blocks and runner, transform in deque later */
    alignas(64) std::vector<uint32_t> blocks(n_threads, UINT32_MAX);
    alignas(64) std::vector<uint32_t> runner(n_threads, UINT32_MAX);

    /* start parallel partition */
    parallel_partition_phase(arr, left, right, B, pv, n_threads, blocks,
                             runner, l_max, r_min, s_key, b_key);

    /* remove invalid/not usable blocks */
    for (int i = blocks.size() - 1; i >= 0; i--) {
        bool not_used = blocks[i] == UINT32_MAX && runner[i] == UINT32_MAX;
        bool in_middle = blocks[i] > l_max && blocks[i] < r_min;
        if (not_used || in_middle) {
            /* remove block and runner */
            blocks.erase(blocks.begin() + i);
            runner.erase(runner.begin() + i);
        }
    }

    /* sort in ascending order */
    sort_blocks(blocks, runner);

    /* convert std::vector in std::deque */
    std::deque<uint32_t> deq_blocks;
    std::deque<uint32_t> deq_runner;
    for (auto b: blocks) { deq_blocks.push_back(b); }
    for (auto r: runner) { deq_runner.push_back(r); }

    uint32_t sp; /* split point */
    if (!blocks.empty()) {
        /* execute the single partition phase */
        sp = single_partition_phase(arr, left, right, B, pv, n_threads, deq_blocks, deq_runner, l_max, r_min, s_key, b_key);
    } else {
        /* partition middle part */
        sp = partition_vectorized_64(arr, l_max + B, r_min - 1, pv, s_key, b_key);
    }
    return sp;
}

/* end parallel vectorized partition
 * *********************************/

/* parallel core functions
 * ***********************/

/* core which does not spawn new threads */
inline void sort_core(int32_t *arr, uint32_t left, uint32_t right, uint32_t B,
                      uint32_t n_threads, bool use_avg, int32_t avg,
                      uint32_t th_counting) {

    if (right + 1 - left < 513) { /* sort in sorting network */
        __m256i buffer[66];
        int *buff = reinterpret_cast<int *>(buffer);
        _internal::sort_int_sorting_network(arr + left, buff, right - left + 1);
        return;
    }

    if (right + 1 - left < 2 * B || n_threads == 1) {
        /* small array or one thread -> non-parallel vectorized sorting */
        vectorized_sort_core(arr, left, right, use_avg, avg, th_counting);
        return;
    }

    int32_t s_key = INT32_MAX, b_key = INT32_MIN; /* smallest, biggest */
    int32_t pv; /* pivot */
    uint32_t sp; /* split point */

    /* estimate how many threads needed for partitioning */
    n_threads = std::max(std::min(n_threads, (right + 1 - left) / 2 * B), (unsigned) 1);

    /* pivot strategy */
    pv = use_avg ? avg : _internal::get_pivot(arr, left, right);
    /* partition */
    sp = partition_core(arr, left, right, B, pv, n_threads, s_key, b_key);
    /* calculate ratio, big to small subarray */
    double ratio = std::min(right + 1 - sp, sp - left) / (double) (right + 1 - left);
    /* change pivot strategy */
    if (ratio < 0.2) { use_avg = !use_avg; }

    if (s_key != pv) { /* different keys in lower subarray */
        uint32_t num_keys = (pv + 1) - s_key; /* how many distinct keys */
        if (num_keys < th_counting) {
            /* small amount of distinct keys -> use counting sort */
            counting_sort(arr, left, sp - 1, s_key, pv);
        } else {
            /* large number of distinct keys -> use Quicksort */
            avg = _internal::average(s_key, pv);
            sort_core(arr, left, sp - 1, B, n_threads, use_avg, avg, th_counting);
        }
    }

    if (b_key != pv + 1) { /* different keys in upper subarray */
        uint32_t num_keys = (b_key + 1) - pv; /* how many distinct keys */
        if (num_keys < th_counting) {
            /* small amount of distinct keys -> use counting sort */
            counting_sort(arr, sp, right, pv, b_key);
        } else {
            /* large number of distinct keys -> use Quicksort */
            avg = _internal::average(pv, b_key);
            sort_core(arr, sp, right, B, n_threads, use_avg, avg, th_counting);
        }
    }
}

/* core which spawns new tasks */
inline void parallel_sort_core(int32_t *arr, uint32_t left, uint32_t right, uint32_t B,
                               uint32_t n_threads, uint32_t depth, uint32_t max_depth,
                               bool use_avg, int32_t avg, uint32_t th_counting) {

    if (right + 1 - left < 513) { /* sort in sorting networks */
        __m256i buffer[66];
        int *buff = reinterpret_cast<int *>(buffer);
        _internal::sort_int_sorting_network(arr + left, buff, right - left + 1);
        return;
    }

    if (right + 1 - left < 2 * B || n_threads == 1) {
        /* small array or one thread -> non-parallel vectorized sort */
        vectorized_sort_core(arr, left, right, use_avg, avg, th_counting);
        return;
    }

    int32_t s_key = INT32_MAX, b_key = INT32_MIN; /* smallest, biggest */
    int32_t pv; /* pivot */
    uint32_t sp; /* split point */

    /* estimate how many threads needed for partitioning */
    n_threads = std::max(std::min(n_threads, (right + 1 - left) / 2 * B), (unsigned) 1);

    /* pivot strategy */
    pv = use_avg ? avg : _internal::get_pivot(arr, left, right);
    /* partition */
    sp = partition_core(arr, left, right, B, pv, n_threads, s_key, b_key);
    /* calculate ratio, big to small subarray */
    double ratio = std::min(right + 1 - sp, sp - left) / (double) (right + 1 - left);
    /* change pivot strategy */
    if (ratio < 0.2) { use_avg = !use_avg; }

    if (s_key != pv) { /* different keys in lower subarray */
        uint32_t num_keys = (pv + 1) - s_key; /* number distinct keys */
        if (num_keys < th_counting) {
            /* small amount of distinct keys -> counting sort */
#pragma omp task firstprivate(arr, left, sp, s_key, pv)
            { counting_sort(arr, left, sp - 1, s_key, pv); }
        } else {
            /* large amount of distinct keys -> quicksort */
            avg = _internal::average(s_key, pv);
            if (depth <= max_depth) {
                /* not at max_depth -> parallel core */
#pragma omp task firstprivate(arr, left, sp, B, n_threads, depth, max_depth, use_avg, avg, th_counting)
                {
                    parallel_sort_core(arr, left, sp - 1, B, n_threads,
                                       depth + 1, max_depth, use_avg, avg, th_counting);
                }
            } else {
                /* at max_depth -> vectorized quicksort */
#pragma omp task firstprivate(arr, left, sp, use_avg, avg, th_counting)
                {
                    vectorized_sort_core(arr, left, sp - 1, use_avg, avg, th_counting);
                }
            }
        }
    }

    if (b_key != pv + 1) { /* different keys in upper subarray */
        uint64_t num_keys = (b_key + 1) - pv; /* number of distinct keys */
        if (num_keys < th_counting) {
            /* small amount of distinct keys -> counting sort */
#pragma omp task firstprivate(arr, sp, right, pv, b_key)
            { counting_sort(arr, sp, right, pv, b_key); }
        } else {
            /* large amount of distinct keys -> quicksort */
            avg = _internal::average(s_key, pv);
            if (depth <= max_depth) {
                /* not at max_depth -> parallel core */
#pragma omp task firstprivate(arr, sp, right, B, n_threads, depth, max_depth, use_avg, avg, th_counting)
                {
                    parallel_sort_core(arr, sp, right, B, n_threads, depth + 1,
                                       max_depth, use_avg, avg, th_counting);
                }
            } else {
                /* at max_depth -> vectorized quicksort */
#pragma omp task default(none) firstprivate(arr, sp, right, use_avg, avg, th_counting)
                {
                    vectorized_sort_core(arr, sp, right, use_avg, avg, th_counting);
                }
            }
        }
    }
}

/* core for quickselect */
inline void quickselect_core(int32_t *arr, uint32_t left, uint32_t right, uint32_t k,
                             uint32_t B, uint32_t n_threads, bool use_avg,
                             int32_t avg) {

    if (right + 1 - left < 257) { /* for few elements use C++'s nth_element */
        std::nth_element(arr + left, arr + k, arr + right + 1);
        return;
    }

    if (right + 1 - left < 2 * B || n_threads == 1) {
        /* one thread or small array -> vectorized */
        vectorized_quickselect_core(arr, left, right, k, use_avg, avg);
        return;
    }

    int32_t s_key = INT32_MAX, b_key = INT32_MIN; /* smallest, biggest */
    int32_t pv; /* pivot */
    uint32_t sp; /* split point */

    /* estimate how many threads needed for partitioning */
    n_threads = std::max(std::min(n_threads, (right + 1 - left) / 2 * B), (unsigned) 1);

    /* pivot strategy */
    pv = use_avg ? avg : _internal::get_pivot(arr, left, right);
    /* partition */
    sp = partition_core(arr, left, right, B, pv, n_threads, s_key, b_key);
    /* calculate ratio, big to small subarray */
    double ratio = std::min(right + 1 - sp, sp - left) / (double) (right + 1 - left);
    /* change pivot strategy */
    if (ratio < 0.2) { use_avg = !use_avg; }

    if (k < sp) { /* k is in the lower sub array */
        if (pv != s_key) /* different values in lower sub array */
            avg = _internal::average(s_key, pv);
            quickselect_core(arr, left, sp - 1, k, B, n_threads, use_avg, avg);
    } else { /* k is in upper sub array */
        if (pv + 1 != b_key) /* different values in upper sub array */
            avg = _internal::average(pv, b_key);
            quickselect_core(arr, sp, right, k, B, n_threads, use_avg, avg);
    }
}

/* end parallel core functions
 * ***************************/
} // end _internal namespace

namespace parallel {
/* parallel divide and conquer vectorized sort */
inline void dac_sort(int32_t *arr, uint32_t size, uint32_t n_threads = omp_get_num_procs(),
                     uint32_t max_depth = 0, uint32_t th_counting = 256) {

    if (max_depth == 0) { /* default value for max_depth */
        max_depth = log2(n_threads) + 3;
    }

    /* start parallel region */
#pragma omp parallel num_threads(n_threads) firstprivate(arr, size, n_threads, max_depth, th_counting)
    {
#pragma omp single /* only one thread */
        _internal::dac_vectorized_sort_core(arr, 0, size - 1,0,
                                            max_depth, false,0, th_counting);
    }
}

/* function to partition */
inline uint32_t partition(int32_t *arr, uint32_t size, int32_t pv, uint32_t B = 512,
                          uint32_t n_threads = omp_get_num_procs()) {

    if (size < 2 * B || n_threads == 1) {
        /* small array or one thread -> vectorized partition */
        int32_t s_key = INT32_MAX;
        int32_t b_key = INT32_MIN;
        return _internal::partition_vectorized_64(arr, 0, size - 1, pv,
                                                  s_key, b_key);
    }

    /* parallel vectorized partition */
    int32_t s_key = INT32_MAX;
    int32_t b_key = INT32_MIN;
    uint32_t sp = 0;

#pragma omp parallel num_threads(n_threads) shared(sp, arr, size, pv, B, n_threads, s_key, b_key)
    {
#pragma omp single
        sp = _internal::partition_core(arr, 0, size - 1, B, pv, n_threads,
                                       s_key, b_key);
    }
    return sp;
}

inline void sort(int32_t *arr, uint32_t size, uint32_t B = 512,
                 uint32_t n_threads = omp_get_num_procs(), uint32_t max_depth = 0,
                 uint32_t th_counting = 0) {

    /* default values */
    if (max_depth == 0) { max_depth = log2(n_threads) + 3; }
    if (th_counting == 0) { th_counting = B / 2; }

    if (size < 2 * B || n_threads == 1) {
        /* small array or one thread -> non-parallel vectorized sort */
        _internal::vectorized_sort_core(arr, 0, size - 1, false,
                                        0, th_counting);
        return;
    }

    /* parallel vectorized sort */
#pragma omp parallel num_threads(n_threads) firstprivate(arr, size, B, n_threads, max_depth, th_counting)
    {
#pragma omp single
        _internal::parallel_sort_core(arr, 0, size - 1, B, n_threads,
                                      0, max_depth, false, 0, th_counting);
    }
}

inline void quickselect(int32_t *arr, uint32_t size, uint32_t k, uint32_t B = 512,
                        uint32_t n_threads = omp_get_num_procs()) {

    if (size < 2 * B || n_threads == 1) {
        /* small array or one thread -> non-parallel vectorized quickselect */
        _internal::vectorized_quickselect_core(arr, 0, size - 1, k,
                                               false, 0);
        return;
    }

#pragma omp parallel num_threads(n_threads) firstprivate(arr, size, k, B, n_threads)
    {
#pragma omp single
        _internal::quickselect_core(arr, 0, size - 1, k, B, n_threads,
                                    false, 0);
    }
}

} // end parallel namespace

inline void sort(int32_t *arr, uint32_t size, uint32_t th_counting = 256) {
    _internal::vectorized_sort_core(arr, 0, size - 1, false,
                                    0, th_counting);
}

inline void quickselect(int32_t *arr, uint32_t size, uint32_t k) {
    _internal::vectorized_quickselect_core(arr, 0, size - 1, k,
                                           false, 0);
}

inline uint32_t partition(int32_t *arr, uint32_t size, int32_t pv) {
    int32_t s = INT32_MAX, b = INT32_MIN;
    return _internal::partition_vectorized_64(arr, 0, size - 1, pv,
                                              s, b);
}

} // end PAvx2 namespace

#endif //QUICKSORT_PAVX2_H
