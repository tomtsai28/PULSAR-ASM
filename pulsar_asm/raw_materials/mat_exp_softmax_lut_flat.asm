; ==============================================================================
; Project PULSAR-ASM | Raw Material 3: mat_exp_softmax_lut_flat.asm
; ------------------------------------------------------------------------------
; Flat Binary x86-64 Machine Code for Vectorized Stable Softmax
; ==============================================================================

use64

softmax_avx2:
    test    r8, r8
    jz      .l_done

    ; Pass 1: Find scalar maximum m = max(x_j)
    mov     eax, 0xFF800000        ; -inf
    vmovd   xmm0, eax
    vbroadcastss ymm0, xmm0        ; YMM0 = [-inf, ...]

    shl     r8, 2                  ; R8 = N * 4 bytes
    xor     r10, r10               ; byte offset = 0

.l_max_loop:
    lea     rax, [r10 + 32]
    cmp     rax, r8
    ja      .l_max_reduce

    vmovups ymm1, [rdx + r10]
    vmaxps  ymm0, ymm0, ymm1
    add     r10, 32
    jmp     .l_max_loop

.l_max_reduce:
    vextractf128 xmm1, ymm0, 1
    vmaxps  xmm0, xmm0, xmm1
    vmovshdup xmm1, xmm0
    vmaxps  xmm0, xmm0, xmm1
    vpermilps xmm1, xmm0, 00000010b
    vmaxps  xmm0, xmm0, xmm1       ; XMM0[0] = global max m
    vbroadcastss ymm0, xmm0        ; YMM0 = [m, m, ...]

    ; Pass 2: e_j = exp(x_j - m) and S = sum(e_j)
    mov     eax, 0x3FB8AA3B        ; log2(e) = 1.44269504f
    vmovd   xmm6, eax
    vbroadcastss ymm6, xmm6

    mov     eax, 0x3F000000        ; 0.5f
    vmovd   xmm7, eax
    vbroadcastss ymm7, xmm7

    mov     eax, 0x3E2AAAAB        ; 1/6 = 0.16666667f
    vmovd   xmm8, eax
    vbroadcastss ymm8, xmm8

    mov     eax, 0x3D2AAAAB        ; 1/24 = 0.041666668f
    vmovd   xmm9, eax
    vbroadcastss ymm9, xmm9

    mov     eax, 0x3F800000        ; 1.0f
    vmovd   xmm10, eax
    vbroadcastss ymm10, xmm10

    mov     eax, 0x3F317218        ; ln(2)_hi = 0.69314718f
    vmovd   xmm11, eax
    vbroadcastss ymm11, xmm11

    vxorps  ymm5, ymm5, ymm5       ; YMM5 = Accumulator S = 0
    xor     r10, r10               ; byte offset = 0

.l_exp_loop:
    lea     rax, [r10 + 32]
    cmp     rax, r8
    ja      .l_exp_reduce

    vmovups ymm1, [rdx + r10]
    vsubps  ymm1, ymm1, ymm0       ; z = x_j - m <= 0

    mov     eax, 0xC2B00000        ; -88.0f
    vmovd   xmm2, eax
    vbroadcastss ymm2, xmm2
    vmaxps  ymm1, ymm1, ymm2       ; clamp to -88.0f

    ; t = z * log2(e)
    vmulps  ymm2, ymm1, ymm6
    vroundps ymm3, ymm2, 00000000b ; k = round(t)

    ; r = z - k * ln(2)
    vfnmadd231ps ymm1, ymm3, ymm11

    ; Horner polynomial P(r)
    vmovaps ymm2, ymm9
    vfmadd213ps ymm2, ymm1, ymm8
    vfmadd213ps ymm2, ymm1, ymm7
    vfmadd213ps ymm2, ymm1, ymm10
    vfmadd213ps ymm2, ymm1, ymm10  ; P(r)

    ; 2^k
    vcvtps2dq ymm3, ymm3
    mov     eax, 127
    vmovd   xmm4, eax
    vbroadcastss ymm4, xmm4
    vpaddd  ymm3, ymm3, ymm4
    vpslld  ymm3, ymm3, 23

    ; e_j = P(r) * 2^k
    vmulps  ymm1, ymm2, ymm3

    vmovups [rcx + r10], ymm1      ; store e_j
    vaddps  ymm5, ymm5, ymm1       ; S += e_j

    add     r10, 32
    jmp     .l_exp_loop

.l_exp_reduce:
    vextractf128 xmm1, ymm5, 1
    vaddps  xmm5, xmm5, xmm1
    vhaddps xmm5, xmm5, xmm5
    vhaddps xmm5, xmm5, xmm5       ; XMM5[0] = S

    mov     eax, 0x3F800000        ; 1.0f
    vmovd   xmm1, eax
    vdivss  xmm1, xmm1, xmm5       ; 1.0 / S
    vbroadcastss ymm0, xmm1        ; YMM0 = [1/S, ...]

    ; Pass 3: Normalize y_j = e_j * (1 / S)
    xor     r10, r10

.l_norm_loop:
    lea     rax, [r10 + 32]
    cmp     rax, r8
    ja      .l_done

    vmovups ymm1, [rcx + r10]
    vmulps  ymm1, ymm1, ymm0
    vmovups [rcx + r10], ymm1

    add     r10, 32
    jmp     .l_norm_loop

.l_done:
    vzeroupper
    ret
