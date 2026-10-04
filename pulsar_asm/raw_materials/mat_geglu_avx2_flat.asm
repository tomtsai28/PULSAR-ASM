; ==============================================================================
; Project PULSAR-ASM | Raw Material 5-Gemma: mat_geglu_avx2_flat.asm
; ------------------------------------------------------------------------------
; Gemma-Specific GeGLU Vector Activation Kernel (AVX2 + FMA3)
;
; Mathematical Formulation:
;   GeGLU(g, u) = GELU(g) * u
;   where GELU(g) ~ 0.5 * g * (1 + tanh(sqrt(2/pi) * (g + 0.044715 * g^3)))
;                 = g / (1 + exp(-w))
;   with  w = 1.59576912 * g + 0.071354816 * g^3
;   and   exp(-w) evaluated via degree-4 minimax Horner polynomial + exponent bias
;
; Win64 ABI:
;   RCX = out  (float* [N])
;   RDX = gate (float* [N])
;   R8  = up   (float* [N])
;   R9  = N    (uint64_t, vector element count, multiple of 8)
;
; Zero CRT, Pure AVX2 Machine Code
; ==============================================================================

use64

geglu_avx2:
    test    r9, r9
    jz      .l_done

    xor     r10, r10               ; R10 = element index i = 0

    ; Load constants
    mov     eax, 0x3FCC422A        ; c1 = 1.59576912f
    vmovd   xmm6, eax
    vbroadcastss ymm6, xmm6

    mov     eax, 0x3D922279        ; c3 = 0.071354816f
    vmovd   xmm7, eax
    vbroadcastss ymm7, xmm7

    mov     eax, 0xC2B00000        ; -88.0f
    vmovd   xmm8, eax
    vbroadcastss ymm8, xmm8

    mov     eax, 0x42B00000        ; +88.0f
    vmovd   xmm9, eax
    vbroadcastss ymm9, xmm9

    mov     eax, 0x3FB8AA3B        ; log2(e) = 1.44269504f
    vmovd   xmm10, eax
    vbroadcastss ymm10, xmm10

    mov     eax, 0x3F317218        ; ln(2) = 0.69314718f
    vmovd   xmm11, eax
    vbroadcastss ymm11, xmm11

    mov     eax, 0x3D2AAAAB        ; 1/24 = 0.041666668f
    vmovd   xmm12, eax
    vbroadcastss ymm12, xmm12

    mov     eax, 0x3E2AAAAB        ; 1/6 = 0.16666667f
    vmovd   xmm13, eax
    vbroadcastss ymm13, xmm13

    mov     eax, 0x3F000000        ; 0.5f
    vmovd   xmm14, eax
    vbroadcastss ymm14, xmm14

    mov     eax, 0x3F800000        ; 1.0f
    vmovd   xmm15, eax
    vbroadcastss ymm15, xmm15

.l_geglu_loop:
    lea     rax, [r10 + 8]
    cmp     rax, r9
    ja      .l_done

    vmovups ymm0, [rdx + r10 * 4]  ; YMM0 = g (gate)
    vmovups ymm1, [r8  + r10 * 4]  ; YMM1 = u (up)

    ; g2 = g * g
    vmulps  ymm2, ymm0, ymm0

    ; w = g * (c1 + c3 * g2)
    vmovaps ymm3, ymm7             ; YMM3 = c3
    vfmadd213ps ymm3, ymm2, ymm6   ; YMM3 = c3 * g2 + c1
    vmulps  ymm3, ymm3, ymm0       ; YMM3 = w = g * (c1 + c3 * g2)

    ; z = -w
    vxorps  ymm2, ymm2, ymm2
    vsubps  ymm2, ymm2, ymm3       ; YMM2 = -w

    ; Clamp z to [-88.0f, +88.0f]
    vmaxps  ymm2, ymm2, ymm8
    vminps  ymm2, ymm2, ymm9

    ; t = z * log2(e)
    vmulps  ymm3, ymm2, ymm10
    vroundps ymm4, ymm3, 00000000b ; YMM4 = k = round(t)

    ; r = z - k * ln(2)
    vfnmadd231ps ymm2, ymm4, ymm11 ; YMM2 = r

    ; Horner Polynomial P(r) = 1 + r*(1 + r*(0.5 + r*(1/6 + r/24)))
    vmovaps ymm3, ymm12            ; 1/24
    vfmadd213ps ymm3, ymm2, ymm13  ; 1/6 + r/24
    vfmadd213ps ymm3, ymm2, ymm14  ; 0.5 + r*(...)
    vfmadd213ps ymm3, ymm2, ymm15  ; 1.0 + r*(...)
    vfmadd213ps ymm3, ymm2, ymm15  ; P(r) = 1.0 + r*(...)

    ; Compute 2^k via IEEE-754 exponent bitfield
    vcvtps2dq ymm4, ymm4
    ; Exponent bias = 127
    mov     eax, 127
    vmovd   xmm5, eax
    vpbroadcastd ymm5, xmm5
    vpaddd  ymm4, ymm4, ymm5
    vpslld  ymm4, ymm4, 23         ; YMM4 = 2^k as float32

    ; exp(-w) = P(r) * 2^k
    vmulps  ymm3, ymm3, ymm4

    ; denom = 1.0 + exp(-w)
    vaddps  ymm3, ymm3, ymm15

    ; out = (g * u) / (1.0 + exp(-w))
    vmulps  ymm0, ymm0, ymm1       ; YMM0 = g * u
    vdivps  ymm0, ymm0, ymm3       ; YMM0 = (g * u) / denom

    ; Store result to out[i..i+7]
    vmovups [rcx + r10 * 4], ymm0

    add     r10, 8
    jmp     .l_geglu_loop

.l_done:
    vzeroupper
    ret
