; ==============================================================================
; Project PULSAR-ASM | Raw Material 1-F16C: mat_f16c_gemv_avx2_flat.asm
; ------------------------------------------------------------------------------
; Full-Precision FP16-Weight GEMV Kernel via Hardware F16C Expansion (AVX2+FMA3)
;
; Designed for Google Gemma-2B (and any FP16 LLM):
;   - Halves DRAM memory bandwidth demand (2 bytes/weight instead of 4 bytes)
;   - Decompresses 8 FP16 weights into 8 IEEE-754 FP32 floats in 1 cycle (VCVTPH2PS)
;   - Maintains 100% full original model precision (no lossy 4-bit quantization!)
;   - Non-Temporal streaming + Software lookahead prefetching (PREFETCHNTA)
;   - Accumulates in FP32 registers using 4 parallel FMA3 pipelines
;
; Win64 ABI:
;   RCX = out (float* [M], FP32 output vector)
;   RDX = W_f16 (uint16_t* [M * K], row-major FP16 weight matrix)
;   R8  = x (float* [K], FP32 activation vector, resident in L1D cache)
;   R9  = K (number of columns / input dimension, elements)
;   [RSP + 40] = M (number of rows / output dimension)
;
; Zero C-Runtime, Pure x86-64 Machine Code
; ==============================================================================

use64

f16c_gemv_avx2:
    mov     rax, [rsp + 40]        ; RAX = M (Total number of rows)
    test    rax, rax
    jz      .l_done
    test    r9, r9
    jz      .l_done

    xor     r10, r10               ; R10 = Current row index i = 0

.l_row_loop:
    ; Prefetch start of next matrix row from DRAM into LFB
    prefetchnta [rdx + r9 * 2]

    ; Clear 4 parallel FP32 accumulators (32 floats = 4 x 8-wide AVX2)
    vxorps  ymm0, ymm0, ymm0
    vxorps  ymm1, ymm1, ymm1
    vxorps  ymm2, ymm2, ymm2
    vxorps  ymm3, ymm3, ymm3

    xor     r11, r11               ; R11 = Current column element index j = 0

.l_col_loop_unrolled:
    lea     rax, [r11 + 32]
    cmp     rax, r9
    ja      .l_col_loop_tail

    ; Issue Non-Temporal Prefetch ~512 bytes (256 FP16 weights = 4 cachelines) ahead
    prefetchnta [rdx + r11 * 2 + 512]

    ; Block 0: Elements j .. j+7 (16 bytes W -> 8 FP32, 32 bytes x)
    vmovups     ymm4, [r8 + r11 * 4]
    vcvtph2ps   ymm5, [rdx + r11 * 2]
    vfmadd231ps ymm0, ymm4, ymm5

    ; Block 1: Elements j+8 .. j+15
    vmovups     ymm4, [r8 + r11 * 4 + 32]
    vcvtph2ps   ymm5, [rdx + r11 * 2 + 16]
    vfmadd231ps ymm1, ymm4, ymm5

    ; Block 2: Elements j+16 .. j+23
    vmovups     ymm4, [r8 + r11 * 4 + 64]
    vcvtph2ps   ymm5, [rdx + r11 * 2 + 32]
    vfmadd231ps ymm2, ymm4, ymm5

    ; Block 3: Elements j+24 .. j+31
    vmovups     ymm4, [r8 + r11 * 4 + 96]
    vcvtph2ps   ymm5, [rdx + r11 * 2 + 48]
    vfmadd231ps ymm3, ymm4, ymm5

    add     r11, 32
    jmp     .l_col_loop_unrolled

.l_col_loop_tail:
    lea     rax, [r11 + 8]
    cmp     rax, r9
    ja      .l_horizontal_reduction

    ; Handle 8-element residue block
    vmovups     ymm4, [r8 + r11 * 4]
    vcvtph2ps   ymm5, [rdx + r11 * 2]
    vfmadd231ps ymm0, ymm4, ymm5

    add     r11, 8
    jmp     .l_col_loop_tail

.l_horizontal_reduction:
    ; Tree-reduction across 4 accumulators: YMM0 = YMM0 + YMM1 + YMM2 + YMM3
    vaddps  ymm0, ymm0, ymm1
    vaddps  ymm2, ymm2, ymm3
    vaddps  ymm0, ymm0, ymm2

    ; Horizontal sum 8 FP32 elements in YMM0 to single scalar in XMM0
    vextractf128 xmm1, ymm0, 1
    vaddps  xmm0, xmm0, xmm1
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0

    ; Store scalar result to out[i]
    vmovss  dword [rcx + r10 * 4], xmm0

    ; Advance weight pointer RDX by row byte stride (K * 2 bytes)
    lea     rdx, [rdx + r9 * 2]
    inc     r10                    ; i++

    mov     rax, [rsp + 40]        ; RAX = M
    cmp     r10, rax
    jb      .l_row_loop

.l_done:
    vzeroupper
    ret
