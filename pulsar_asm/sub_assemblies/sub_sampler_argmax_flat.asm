; ==============================================================================
; Project PULSAR-ASM | Sub-Assembly D: sub_sampler_argmax_flat.asm
; ------------------------------------------------------------------------------
; High-Performance Vectorized Argmax Sampler (AVX2)
;
; Win64 ABI:
;   RCX = logits pointer (float* [vocab_size])
;   RDX = vocab_size (uint64_t)
; Return:
;   RAX = best_token_id (uint64_t)
;
; Zero CRT, Pure AVX2 Machine Code
; ==============================================================================

use64

sampler_argmax_avx2:
    test    rdx, rdx
    jnz     .l_start
    xor     rax, rax
    ret

.l_start:
    push    rbx
    push    rsi

    ; Initialize with first element
    vmovss  xmm0, dword [rcx]      ; XMM0[0] = current max_val
    vbroadcastss ymm0, xmm0        ; YMM0 = [max_val, ...]
    xor     rax, rax               ; RAX = best_idx = 0

    mov     r8, rdx
    and     r8, not 7              ; R8 = vocab_size & ~7 (multiples of 8)
    shl     r8, 2                  ; R8 = byte limit for vector portion

    xor     r10, r10               ; byte offset = 0

.l_vec_loop:
    cmp     r10, r8
    jae     .l_tail_loop

    vmovups ymm1, [rcx + r10]      ; Load 8 logits
    vcmpps  ymm2, ymm1, ymm0, 14   ; 14 = _CMP_GT_OS (Greater-than compare)
    vmovmskps r11d, ymm2           ; Extract 8-bit mask of candidates

    test    r11d, r11d
    jz      .l_next_vec            ; 99.9% of iterations skip (no new maximum)

    ; One or more elements in YMM1 are strictly greater than current YMM0
    lea     rbx, [rcx + r10]       ; RBX = current chunk pointer
    xor     r9, r9
.l_inner_cand:
    test    r11d, 1
    jz      .l_skip_cand

    vmovss  xmm3, dword [rbx + r9 * 4]
    ucomiss xmm3, xmm0
    jbe     .l_skip_cand

    vmovaps xmm0, xmm3
    vbroadcastss ymm0, xmm0        ; update running max
    lea     rsi, [r10 + r9 * 4]
    shr     rsi, 2                 ; RSI = best_idx
    mov     rax, rsi

.l_skip_cand:
    shr     r11d, 1
    inc     r9
    cmp     r9, 8
    jb      .l_inner_cand

.l_next_vec:
    add     r10, 32
    jmp     .l_vec_loop

.l_tail_loop:
    mov     r9, rdx
    shl     r9, 2                  ; total bytes
    cmp     r10, r9
    jae     .l_done

    vmovss  xmm1, dword [rcx + r10]
    ucomiss xmm1, xmm0
    jbe     .l_next_tail

    vmovaps xmm0, xmm1
    mov     rax, r10
    shr     rax, 2                 ; best_idx = r10 / 4

.l_next_tail:
    add     r10, 4
    jmp     .l_tail_loop

.l_done:
    pop     rsi
    pop     rbx
    vzeroupper
    ret
