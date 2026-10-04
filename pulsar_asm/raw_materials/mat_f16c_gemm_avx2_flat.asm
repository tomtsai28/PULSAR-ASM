; ==============================================================================
; Project PULSAR-ASM | Component: mat_f16c_gemm_avx2_flat.asm
; ------------------------------------------------------------------------------
; High-Throughput Batch-8 FP16 GEMM Micro-Kernel (AVX2 + F16C + FMA3)
;
; Zero Stack Loads in Inner Loop:
;   All 8 token base pointers pinned into x86-64 GP registers (R8, R13, R14, RBX, RBP, RSI, RDI, RCX)
;
; Win64 ABI:
;   RCX = out (float* [B * full_M])
;   RDX = W_f16 (uint16_t* [M * K])
;   R8  = X_batch (float* [B * K])
;   R9  = K (columns)
;   [RSP + 40] = M (rows for this worker)
;   [RSP + 48] = B (batch size, multiple of 8)
;   [RSP + 56] = full_M (full M for output stride, or 0)
; ==============================================================================

use64

f16c_gemm_batch8_avx2:
    push    rbx
    push    rbp
    push    rsi
    push    rdi
    push    r12
    push    r13
    push    r14
    push    r15
    sub     rsp, 80                ; 80 bytes for local frame variables

    ; Caller stack offsets:
    ; 8 pushes (64B) + sub rsp 80 (80B) = 144B
    ; Arg 5 (M)      at [rsp + 144 + 40] = [rsp + 184]
    ; Arg 6 (B)      at [rsp + 144 + 48] = [rsp + 192]
    ; Arg 7 (full_M) at [rsp + 144 + 56] = [rsp + 200]
    mov     rax, [rsp + 184]       ; M
    mov     [rsp + 0],  rax        ; var_M
    mov     rax, [rsp + 192]       ; B
    mov     [rsp + 8],  rax        ; var_B
    mov     rax, [rsp + 200]       ; full_M
    test    rax, rax
    cmovz   rax, [rsp + 0]         ; If 0, fallback to M
    mov     [rsp + 16], rax        ; var_full_M
    shl     rax, 2
    mov     [rsp + 24], rax        ; var_out_stride_bytes = full_M * 4

    mov     [rsp + 32], rcx        ; var_out_base
    mov     [rsp + 40], rdx        ; var_W_base
    mov     [rsp + 48], r8         ; var_X_base
    mov     [rsp + 56], r9         ; var_K

    cmp     qword [rsp + 0], 0
    jz      .l_exit
    cmp     qword [rsp + 8], 0
    jz      .l_exit

    ; Token stride in X: K * 4 bytes
    mov     rax, r9
    shl     rax, 2
    mov     [rsp + 64], rax        ; var_x_stride_bytes = K * 4

    ; Loop over rows i = 0 .. M-1
    xor     r15, r15               ; R15 = row index i = 0
    mov     rax, [rsp + 40]
    mov     [rsp + 72], rax        ; var_current_W_row = var_W_base

.l_row_loop:
    cmp     r15, [rsp + 0]         ; i < var_M
    jae     .l_exit

    ; Prefetch next row of weights into L2/L3 non-temporally
    mov     rax, [rsp + 72]
    mov     r9,  [rsp + 56]        ; var_K
    prefetchnta [rax + r9 * 2]

    ; Inner loop over tokens in chunks of 8: b_start = 0, 8, 16 ... < var_B
    xor     r12, r12               ; R12 = b_start = 0

.l_token_chunk_loop:
    cmp     r12, [rsp + 8]         ; b_start < var_B
    jae     .l_next_row

    ; Pin the 8 token pointers into GP registers:
    ; X_base for token (b_start + b) = var_X_base + (b_start + b) * var_x_stride_bytes
    mov     rax, r12
    imul    rax, [rsp + 64]
    add     rax, [rsp + 48]        ; RAX = pointer to token (b_start + 0)

    mov     r8,  rax               ; R8  = Token 0
    mov     r10, [rsp + 64]        ; R10 = stride
    lea     r13, [r8  + r10]       ; R13 = Token 1
    lea     r14, [r13 + r10]       ; R14 = Token 2
    lea     rbx, [r14 + r10]       ; RBX = Token 3
    lea     rbp, [rbx + r10]       ; RBP = Token 4
    lea     rsi, [rbp + r10]       ; RSI = Token 5
    lea     rdi, [rsi + r10]       ; RDI = Token 6
    lea     rcx, [rdi + r10]       ; RCX = Token 7

    ; RDX = current W row pointer (resident in L1 cache after chunk 0!)
    mov     rdx, [rsp + 72]        ; RDX = var_current_W_row
    mov     r9,  [rsp + 56]        ; R9  = var_K

    ; Clear 8 FP32 accumulators (YMM0 ~ YMM7)
    vxorps  ymm0, ymm0, ymm0
    vxorps  ymm1, ymm1, ymm1
    vxorps  ymm2, ymm2, ymm2
    vxorps  ymm3, ymm3, ymm3
    vxorps  ymm4, ymm4, ymm4
    vxorps  ymm5, ymm5, ymm5
    vxorps  ymm6, ymm6, ymm6
    vxorps  ymm7, ymm7, ymm7

    ; --------------------------------------------------------------------------
    ; Inner Loop over columns j = 0 .. K - 1 step 8 (100% REGISTERS, ZERO STACK!)
    ; --------------------------------------------------------------------------
    xor     r11, r11               ; R11 = j = 0

.l_col_loop:
    cmp     r11, r9
    jae     .l_reduce_and_store

    ; Load 8 FP16 weights, decompress to 8 FP32 into YMM15
    vcvtph2ps   ymm15, [rdx + r11 * 2]

    ; Token 0 (R8)
    vmovups     ymm8, [r8 + r11 * 4]
    vfmadd231ps ymm0, ymm8, ymm15

    ; Token 1 (R13)
    vmovups     ymm8, [r13 + r11 * 4]
    vfmadd231ps ymm1, ymm8, ymm15

    ; Token 2 (R14)
    vmovups     ymm8, [r14 + r11 * 4]
    vfmadd231ps ymm2, ymm8, ymm15

    ; Token 3 (RBX)
    vmovups     ymm8, [rbx + r11 * 4]
    vfmadd231ps ymm3, ymm8, ymm15

    ; Token 4 (RBP)
    vmovups     ymm8, [rbp + r11 * 4]
    vfmadd231ps ymm4, ymm8, ymm15

    ; Token 5 (RSI)
    vmovups     ymm8, [rsi + r11 * 4]
    vfmadd231ps ymm5, ymm8, ymm15

    ; Token 6 (RDI)
    vmovups     ymm8, [rdi + r11 * 4]
    vfmadd231ps ymm6, ymm8, ymm15

    ; Token 7 (RCX)
    vmovups     ymm8, [rcx + r11 * 4]
    vfmadd231ps ymm7, ymm8, ymm15

    add         r11, 8
    jmp         .l_col_loop

.l_reduce_and_store:
    ; Compute base out pointer for row i:
    ; out_token_b_start = var_out_base + (b_start * var_out_stride_bytes) + (i * 4)
    mov     rax, r12
    imul    rax, [rsp + 24]
    add     rax, [rsp + 32]
    lea     rax, [rax + r15 * 4]   ; RAX = destination for Token 0 in this chunk
    mov     r10, [rsp + 24]        ; R10 = var_out_stride_bytes

    ; Token 0
    vextractf128 xmm14, ymm0, 1
    vaddps       xmm0, xmm0, xmm14
    vhaddps      xmm0, xmm0, xmm0
    vhaddps      xmm0, xmm0, xmm0
    vmovss       [rax], xmm0

    ; Token 1
    vextractf128 xmm14, ymm1, 1
    vaddps       xmm1, xmm1, xmm14
    vhaddps      xmm1, xmm1, xmm1
    vhaddps      xmm1, xmm1, xmm1
    vmovss       [rax + r10 * 1], xmm1

    ; Token 2
    vextractf128 xmm14, ymm2, 1
    vaddps       xmm2, xmm2, xmm14
    vhaddps      xmm2, xmm2, xmm2
    vhaddps      xmm2, xmm2, xmm2
    lea          r11, [rax + r10 * 2]
    vmovss       [r11], xmm2

    ; Token 3
    vextractf128 xmm14, ymm3, 1
    vaddps       xmm3, xmm3, xmm14
    vhaddps      xmm3, xmm3, xmm3
    vhaddps      xmm3, xmm3, xmm3
    vmovss       [r11 + r10 * 1], xmm3

    ; Token 4
    vextractf128 xmm14, ymm4, 1
    vaddps       xmm4, xmm4, xmm14
    vhaddps      xmm4, xmm4, xmm4
    vhaddps      xmm4, xmm4, xmm4
    lea          r11, [r11 + r10 * 2]
    vmovss       [r11], xmm4

    ; Token 5
    vextractf128 xmm14, ymm5, 1
    vaddps       xmm5, xmm5, xmm14
    vhaddps      xmm5, xmm5, xmm5
    vhaddps      xmm5, xmm5, xmm5
    vmovss       [r11 + r10 * 1], xmm5

    ; Token 6
    vextractf128 xmm14, ymm6, 1
    vaddps       xmm6, xmm6, xmm14
    vhaddps      xmm6, xmm6, xmm6
    vhaddps      xmm6, xmm6, xmm6
    lea          r11, [r11 + r10 * 2]
    vmovss       [r11], xmm6

    ; Token 7
    vextractf128 xmm14, ymm7, 1
    vaddps       xmm7, xmm7, xmm14
    vhaddps      xmm7, xmm7, xmm7
    vhaddps      xmm7, xmm7, xmm7
    vmovss       [r11 + r10 * 1], xmm7

    ; Next chunk of 8 tokens for the SAME row i
    add     r12, 8
    jmp     .l_token_chunk_loop

.l_next_row:
    ; Advance W to next row: var_current_W_row += K * 2
    mov     rax, [rsp + 56]        ; var_K
    shl     rax, 1
    add     [rsp + 72], rax

    inc     r15
    jmp     .l_row_loop

.l_exit:
    add     rsp, 80
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rdi
    pop     rsi
    pop     rbp
    pop     rbx
    vzeroupper
    ret
