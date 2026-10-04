; ==============================================================================
; Project PULSAR-ASM | Raw Material 2-Gemma: mat_gemma_rmsnorm_avx2_flat.asm
; ------------------------------------------------------------------------------
; Google Gemma Specification Root-Mean-Square Normalization (RMSNorm)
;
; Mathematical Formulation:
;   SS = (1 / N) * sum(x_i^2) + epsilon
;   rsqrt = 1.0 / sqrt(SS)
;   out_i = x_i * rsqrt * (1.0 + weight_i)  <-- Gemma Unit Offset Weight
;
; Win64 ABI:
;   RCX = out (float* [N])
;   RDX = weight (float* [N])
;   R8  = x (float* [N])
;   R9  = N (uint64_t, vector element count, multiple of 8)
;   [RSP + 40] = epsilon (float)
;
; Zero CRT, Pure AVX2 Machine Code
; ==============================================================================

use64

gemma_rmsnorm_avx2:
    test    r9, r9
    jz      .l_done

    ; Step 1: Accumulate sum of squares in 4 parallel YMM accumulators
    vxorps  ymm0, ymm0, ymm0
    vxorps  ymm1, ymm1, ymm1
    vxorps  ymm2, ymm2, ymm2
    vxorps  ymm3, ymm3, ymm3

    shl     r9, 2                  ; R9 = N * 4 bytes
    xor     r10, r10               ; byte offset = 0

.l_ss_unrolled:
    lea     rax, [r10 + 128]
    cmp     rax, r9
    ja      .l_ss_tail

    vmovups ymm4, [r8 + r10]
    vfmadd231ps ymm0, ymm4, ymm4

    vmovups ymm4, [r8 + r10 + 32]
    vfmadd231ps ymm1, ymm4, ymm4

    vmovups ymm4, [r8 + r10 + 64]
    vfmadd231ps ymm2, ymm4, ymm4

    vmovups ymm4, [r8 + r10 + 96]
    vfmadd231ps ymm3, ymm4, ymm4

    add     r10, 128
    jmp     .l_ss_unrolled

.l_ss_tail:
    lea     rax, [r10 + 32]
    cmp     rax, r9
    ja      .l_ss_reduce

    vmovups ymm4, [r8 + r10]
    vfmadd231ps ymm0, ymm4, ymm4

    add     r10, 32
    jmp     .l_ss_tail

.l_ss_reduce:
    vaddps  ymm0, ymm0, ymm1
    vaddps  ymm2, ymm2, ymm3
    vaddps  ymm0, ymm0, ymm2

    vextractf128 xmm1, ymm0, 1
    vaddps  xmm0, xmm0, xmm1
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0       ; XMM0[0] = sum(x_i^2)

    ; Step 2: SS = (sum / N) + epsilon
    shr     r9, 2                  ; Restore R9 = N
    vcvtsi2ss xmm2, xmm2, r9d      ; XMM2 = (float)N
    vdivss  xmm0, xmm0, xmm2       ; XMM0 = sum / N

    vmovss  xmm3, dword [rsp + 40] ; Load epsilon
    vaddss  xmm0, xmm0, xmm3       ; XMM0 = SS = (sum / N) + epsilon

    ; Step 3: Fast rsqrt with Newton-Raphson Refinement
    vrsqrtss xmm1, xmm0, xmm0      ; XMM1 = initial guess y0 (11-bit)

    mov     eax, 0x3F000000        ; 0.5f in IEEE-754
    vmovd   xmm4, eax
    mov     eax, 0x40400000        ; 3.0f in IEEE-754
    vmovd   xmm5, eax

    vmulss  xmm2, xmm1, xmm1       ; y0 * y0
    vmulss  xmm2, xmm2, xmm0       ; x * y0 * y0
    vsubss  xmm2, xmm5, xmm2       ; 3.0 - (x * y0 * y0)
    vmulss  xmm2, xmm2, xmm4       ; 0.5 * (3.0 - ...)
    vmulss  xmm1, xmm1, xmm2       ; XMM1 = refined rsqrt

    vbroadcastss ymm0, xmm1        ; YMM0 = refined rsqrt

    ; Load 1.0f constant for Gemma unit offset: (1.0 + weight)
    mov     eax, 0x3F800000        ; 1.0f
    vmovd   xmm6, eax
    vbroadcastss ymm6, xmm6        ; YMM6 = 1.0f

    ; Step 4: Normalization & Gemma Weight Scaling: y_i = x_i * rsqrt * (1.0 + weight_i)
    shl     r9, 2                  ; R9 = N * 4 bytes
    xor     r10, r10               ; byte offset = 0

.l_norm_loop:
    lea     rax, [r10 + 32]
    cmp     rax, r9
    ja      .l_done

    vmovups ymm1, [r8 + r10]       ; x
    vmovups ymm2, [rdx + r10]      ; weight
    vaddps  ymm2, ymm2, ymm6       ; (1.0 + weight)
    vmulps  ymm1, ymm1, ymm0       ; x * rsqrt
    vmulps  ymm1, ymm1, ymm2       ; x * rsqrt * (1.0 + weight)
    vmovups [rcx + r10], ymm1

    add     r10, 32
    jmp     .l_norm_loop

.l_done:
    vzeroupper
    ret
