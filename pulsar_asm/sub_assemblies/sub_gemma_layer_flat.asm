; ==============================================================================
; Project PULSAR-ASM | Sub-Assembly E: sub_gemma_layer_flat.asm
; ------------------------------------------------------------------------------
; Google Gemma-2B Full-Precision Transformer Layer Module (AVX2 + F16C + FMA3)
;
; Specifications (Gemma-2B):
;   dim = 2048, hidden_dim = 16384, q_heads = 8, kv_heads = 1 (MQA), head_dim = 256
;   Weights: FP16 DRAM Non-Temporal Streaming (F16C Hardware Decompression)
;   Activations: FP32 L1/L2 Cache Resident
;   Normalization: Gemma Unit-Offset (1.0 + W) RMSNorm
;   FFN: GeGLU (GELU(gate) * up) Degree-4 Horner Polynomial
;   RoPE: Contiguous Half-Rotation (rotate_half)
;
; GemmaLayerParams Struct (passed via RCX):
;   [RCX + 0]   : x (float* [dim], in/out residual stream)
;   [RCX + 8]   : rms_att_w (float* [dim])
;   [RCX + 16]  : w_q_f16 (uint16_t* [dim * dim])
;   [RCX + 24]  : w_k_f16 (uint16_t* [head_dim * dim])
;   [RCX + 32]  : w_v_f16 (uint16_t* [head_dim * dim])
;   [RCX + 40]  : w_o_f16 (uint16_t* [dim * dim])
;   [RCX + 48]  : rms_ffn_w (float* [dim])
;   [RCX + 56]  : w_gate_f16 (uint16_t* [hidden_dim * dim])
;   [RCX + 64]  : w_up_f16 (uint16_t* [hidden_dim * dim])
;   [RCX + 72]  : w_down_f16 (uint16_t* [dim * hidden_dim])
;   [RCX + 80]  : key_cache (float* [max_seq * head_dim])
;   [RCX + 88]  : val_cache (float* [max_seq * head_dim])
;   [RCX + 96]  : rope_cos (float* [head_dim / 2])
;   [RCX + 104] : rope_sin (float* [head_dim / 2])
;   [RCX + 112] : scratch (float* [buffer >= 200 KB])
;   [RCX + 120] : pos (uint64_t, current sequence position)
;   [RCX + 128] : dim (uint64_t, 2048)
;   [RCX + 136] : hidden_dim (uint64_t, 16384)
;   [RCX + 144] : n_heads (uint64_t, 8)
;   [RCX + 152] : head_dim (uint64_t, 256)
;   [RCX + 160] : smp_state (SmpSharedState*)
;
; Zero C-Runtime, 100% Full-Precision Native Machine Code
; ==============================================================================

use64

sub_gemma_layer:
    ; 1. Preserve non-volatile registers (64 bytes)
    push    rbx
    push    rbp
    push    rsi
    push    rdi
    push    r12
    push    r13
    push    r14
    push    r15

    ; 2. Stack frame: 32 bytes shadow + 8 bytes 5th arg + 8 bytes 6th arg = 48 bytes
    sub     rsp, 48

    mov     r15, rcx               ; R15 = pointer to GemmaLayerParams
    mov     rsi, [r15 + 112]       ; RSI = scratch base

    ; Memory layout within scratch buffer:
    ; [rsi + 0]      : x_norm (dim = 2048 floats = 8,192 bytes)
    ; [rsi + 8192]   : res_buf (dim = 2048 floats = 8,192 bytes)
    ; [rsi + 16384]  : q_buf (dim = 2048 floats = 8,192 bytes)
    ; [rsi + 24576]  : k_buf (head_dim = 256 floats = 1,024 bytes)
    ; [rsi + 25600]  : v_buf (head_dim = 256 floats = 1,024 bytes)
    ; [rsi + 26624]  : attn_out (dim = 2048 floats = 8,192 bytes)
    ; [rsi + 34816]  : scores_buf (1024 floats = 4,096 bytes)
    ; [rsi + 38912]  : s_gate (hidden_dim = 16384 floats = 65,536 bytes)
    ; [rsi + 104448] : s_up (hidden_dim = 16384 floats = 65,536 bytes)

    ; --------------------------------------------------------------------------
    ; Step 1: Pre-Attention RMSNorm (Gemma (1.0 + W) formula)
    ; --------------------------------------------------------------------------
    mov     rcx, rsi               ; out = x_norm
    mov     rdx, [r15 + 8]         ; weight = rms_att_w
    mov     r8,  [r15 + 0]         ; x = input x
    mov     r9,  [r15 + 128]       ; N = dim (2048)
    mov     dword [rsp + 32], 0x3727C5AC ; eps = 1e-5
    call    gemma_rmsnorm_avx2

    ; --------------------------------------------------------------------------
    ; Step 2: Q, K, V Projections via SMP F16C GEMV
    ; --------------------------------------------------------------------------
    mov     rax, [r15 + 160]       ; smp_state
    mov     [rsp + 40], rax        ; 6th arg = smp_state

    ; Q = gemv_f16c(q_buf, w_q_f16, x_norm, dim, dim)
    lea     rcx, [rsi + 16384]     ; out = q_buf
    mov     rdx, [r15 + 16]        ; W = w_q_f16
    mov     r8,  rsi               ; x = x_norm
    mov     r9,  [r15 + 128]       ; K = dim (2048)
    mov     rax, [r15 + 128]       ; M = dim (2048)
    mov     [rsp + 32], rax        ; 5th arg = M
    call    smp_f16c_gemv_avx2

    ; K = gemv_f16c(k_buf, w_k_f16, x_norm, dim, head_dim) (MQA: 1 head = 256)
    lea     rcx, [rsi + 24576]     ; out = k_buf
    mov     rdx, [r15 + 24]        ; W = w_k_f16
    mov     r8,  rsi               ; x = x_norm
    mov     r9,  [r15 + 128]       ; K = dim (2048)
    mov     rax, [r15 + 152]       ; M = head_dim (256)
    mov     [rsp + 32], rax
    call    smp_f16c_gemv_avx2

    ; V = gemv_f16c(v_buf, w_v_f16, x_norm, dim, head_dim) (MQA: 1 head = 256)
    lea     rcx, [rsi + 25600]     ; out = v_buf
    mov     rdx, [r15 + 32]        ; W = w_v_f16
    mov     r8,  rsi               ; x = x_norm
    mov     r9,  [r15 + 128]       ; K = dim (2048)
    mov     rax, [r15 + 152]       ; M = head_dim (256)
    mov     [rsp + 32], rax
    call    smp_f16c_gemv_avx2

    ; --------------------------------------------------------------------------
    ; Step 3: RoPE Rotation (rotate_half) on Q and K
    ; --------------------------------------------------------------------------
    mov     rbx, [r15 + 96]        ; RBX = rope_cos (128 floats)
    mov     rdi, [r15 + 104]       ; RDI = rope_sin (128 floats)

    ; 3a. Rotate 8 heads of Q (each head has 256 floats, half = 128 floats)
    xor     r12, r12               ; head idx = 0
.l_rope_q:
    ; head base in q_buf: rsi + 16384 + r12 * 256 * 4
    mov     rax, r12
    shl     rax, 10                ; 256 * 4 = 1024 bytes per head
    lea     r13, [rsi + 16384 + rax] ; R13 = head pointer

    xor     r14, r14               ; pair idx = 0 .. 127
.l_rope_q_vec:
    ; x1 = head[r14], x2 = head[r14 + 128]
    vmovups ymm0, [r13 + r14 * 4]          ; x1
    vmovups ymm1, [r13 + r14 * 4 + 512]    ; x2 (128 * 4 = 512 bytes)
    vmovups ymm2, [rbx + r14 * 4]          ; cos
    vmovups ymm3, [rdi + r14 * 4]          ; sin

    ; out1 = x1 * cos - x2 * sin
    vmulps      ymm4, ymm0, ymm2
    vfnmadd231ps ymm4, ymm1, ymm3
    vmovups     [r13 + r14 * 4], ymm4

    ; out2 = x2 * cos + x1 * sin
    vmulps      ymm5, ymm1, ymm2
    vfmadd231ps ymm5, ymm0, ymm3
    vmovups     [r13 + r14 * 4 + 512], ymm5

    add     r14, 8
    cmp     r14, 128
    jb      .l_rope_q_vec

    inc     r12
    cmp     r12, 8
    jb      .l_rope_q

    ; 3b. Rotate 1 head of K (k_buf: rsi + 24576)
    lea     r13, [rsi + 24576]
    xor     r14, r14
.l_rope_k_vec:
    vmovups ymm0, [r13 + r14 * 4]
    vmovups ymm1, [r13 + r14 * 4 + 512]
    vmovups ymm2, [rbx + r14 * 4]
    vmovups ymm3, [rdi + r14 * 4]

    vmulps      ymm4, ymm0, ymm2
    vfnmadd231ps ymm4, ymm1, ymm3
    vmovups     [r13 + r14 * 4], ymm4

    vmulps      ymm5, ymm1, ymm2
    vfmadd231ps ymm5, ymm0, ymm3
    vmovups     [r13 + r14 * 4 + 512], ymm5

    add     r14, 8
    cmp     r14, 128
    jb      .l_rope_k_vec

    ; --------------------------------------------------------------------------
    ; Step 4: Write K and V into KV Cache Ring Buffer
    ; cache offset = pos * 256 * 4 = pos * 1024 bytes
    ; --------------------------------------------------------------------------
    mov     rax, [r15 + 120]       ; RAX = pos
    shl     rax, 10                ; RAX = pos * 1024 bytes
    mov     rcx, [r15 + 80]        ; key_cache
    add     rcx, rax               ; dest_k = key_cache + pos * 1024
    lea     rdx, [rsi + 24576]     ; src_k = k_buf
    mov     r8,  [r15 + 88]        ; val_cache
    add     r8,  rax               ; dest_v = val_cache + pos * 1024
    lea     r9,  [rsi + 25600]     ; src_v = v_buf

    xor     r10, r10
.l_cache_copy:
    vmovups ymm0, [rdx + r10]
    vmovups [rcx + r10], ymm0
    vmovups ymm1, [r9  + r10]
    vmovups [r8  + r10], ymm1
    add     r10, 32
    cmp     r10, 1024
    jb      .l_cache_copy

    ; --------------------------------------------------------------------------
    ; Step 5: Multi-Query Attention (MQA) Computation
    ; For each Q head h in [0..7]:
    ;   For each p in [0..pos]:
    ;     score[p] = (1 / sqrt(256)) * (Q_h . K_p) = 0.0625 * (Q_h . K_p)
    ;   Softmax(score[0..pos])
    ;   attn_out_h = sum(score[p] * V_p)
    ; --------------------------------------------------------------------------
    mov     r12, [r15 + 120]       ; R12 = pos
    inc     r12                    ; seq_len = pos + 1

    xor     rbx, rbx               ; RBX = head h = 0 .. 7
.l_head_loop:
    ; Q_h pointer: rsi + 16384 + h * 1024
    mov     rax, rbx
    shl     rax, 10
    lea     r13, [rsi + 16384 + rax] ; R13 = Q_h

    ; Scale factor: 0.0625f = 1 / sqrt(256)
    mov     eax, 0x3D800000
    vmovd   xmm7, eax

    ; 5a. Dot Q_h with all cached K_p (p = 0 .. pos)
    xor     r14, r14               ; R14 = p
.l_score_p_loop:
    mov     rax, r14
    shl     rax, 10
    mov     rcx, [r15 + 80]
    add     rcx, rax               ; RCX = K_p

    vxorps  ymm0, ymm0, ymm0
    vxorps  ymm1, ymm1, ymm1
    xor     r10, r10               ; byte offset in 256-float head
.l_dot_k:
    vmovups ymm2, [r13 + r10]
    vmovups ymm3, [rcx + r10]
    vfmadd231ps ymm0, ymm2, ymm3

    vmovups ymm2, [r13 + r10 + 32]
    vmovups ymm3, [rcx + r10 + 32]
    vfmadd231ps ymm1, ymm2, ymm3

    add     r10, 64
    cmp     r10, 1024
    jb      .l_dot_k

    ; Sum reduction
    vaddps  ymm0, ymm0, ymm1
    vextractf128 xmm1, ymm0, 1
    vaddps  xmm0, xmm0, xmm1
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0       ; XMM0 = dot product

    vmulss  xmm0, xmm0, xmm7       ; scale by 0.0625f

    ; Save to scores_buf[p] (rsi + 34816)
    vmovss  dword [rsi + 34816 + r14 * 4], xmm0

    inc     r14
    cmp     r14, r12
    jb      .l_score_p_loop

    ; 5b. Softmax(scores_buf[0 .. pos])
    mov     rax, [r15 + 120]       ; RAX = pos
    test    rax, rax
    jnz     .l_multi_token_softmax

    ; If pos == 0, single token: attention weight is 1.0f!
    mov     dword [rsi + 34816], 0x3F800000 ; scores_buf[0] = 1.0f
    jmp     .l_val_accum

.l_multi_token_softmax:
    mov     r8, rax
    inc     r8                     ; R8 = N = pos + 1

    ; N_padded = ((N + 7) / 8) * 8
    lea     rcx, [r8 + 7]
    and     rcx, not 7             ; RCX = N_padded

    ; Pad tail elements [N .. N_padded-1] with -10000.0f (0xC61C4000)
    mov     edx, 0xC61C4000
    mov     r9, r8
.l_pad_loop:
    cmp     r9, rcx
    jae     .l_pad_done
    mov     dword [rsi + 34816 + r9 * 4], edx
    inc     r9
    jmp     .l_pad_loop

.l_pad_done:
    lea     rcx, [rsi + 34816]     ; out = scores_buf
    mov     rdx, rcx               ; in = scores_buf
    mov     r8,  r9                ; N = N_padded (multiple of 8)
    call    softmax_avx2

.l_val_accum:
    ; 5c. Weighted sum of V_p: attn_out_h = sum(score[p] * V_p)
    ; Clear destination head buffer in attn_out: rsi + 26624 + h * 1024
    mov     rax, rbx
    shl     rax, 10
    lea     rdi, [rsi + 26624 + rax] ; RDI = attn_out_h

    xor     r10, r10
.l_clear_attn_out:
    vxorps  ymm0, ymm0, ymm0
    vmovups [rdi + r10], ymm0
    vmovups [rdi + r10 + 32], ymm0
    add     r10, 64
    cmp     r10, 1024
    jb      .l_clear_attn_out

    xor     r14, r14               ; p = 0 .. pos
.l_weighted_v_loop:
    vbroadcastss ymm6, dword [rsi + 34816 + r14 * 4] ; YMM6 = score[p]

    mov     rax, r14
    shl     rax, 10
    mov     rcx, [r15 + 88]
    add     rcx, rax               ; RCX = V_p

    xor     r10, r10
.l_accum_v:
    vmovups ymm0, [rdi + r10]
    vmovups ymm1, [rcx + r10]
    vfmadd231ps ymm0, ymm6, ymm1
    vmovups [rdi + r10], ymm0

    vmovups ymm2, [rdi + r10 + 32]
    vmovups ymm3, [rcx + r10 + 32]
    vfmadd231ps ymm2, ymm6, ymm3
    vmovups [rdi + r10 + 32], ymm2

    add     r10, 64
    cmp     r10, 1024
    jb      .l_accum_v

    inc     r14
    cmp     r14, r12
    jb      .l_weighted_v_loop

    inc     rbx                    ; next head
    cmp     rbx, 8
    jb      .l_head_loop

    ; --------------------------------------------------------------------------
    ; --------------------------------------------------------------------------
    ; Step 6: Output Projection O = gemv_f16c(res_buf, w_o_f16, attn_out, dim, dim)
    ; --------------------------------------------------------------------------
    mov     rax, [r15 + 160]       ; smp_state
    mov     [rsp + 40], rax        ; 6th arg = smp_state
    lea     rcx, [rsi + 8192]      ; out = res_buf
    mov     rdx, [r15 + 40]        ; W = w_o_f16
    lea     r8,  [rsi + 26624]     ; x = attn_out
    mov     r9,  [r15 + 128]       ; K = dim (2048)
    mov     rax, [r15 + 128]       ; M = dim (2048)
    mov     [rsp + 32], rax
    call    smp_f16c_gemv_avx2

    ; --------------------------------------------------------------------------
    ; Step 7: Residual Connection 1: x = x + res_buf
    ; --------------------------------------------------------------------------
    mov     rcx, [r15 + 0]         ; x
    lea     rdx, [rsi + 8192]      ; res_buf
    mov     r8,  [r15 + 128]
    shl     r8,  2                 ; dim * 4 bytes
    xor     r10, r10
.l_res1:
    vmovups ymm0, [rcx + r10]
    vmovups ymm1, [rdx + r10]
    vaddps  ymm0, ymm0, ymm1
    vmovups [rcx + r10], ymm0
    add     r10, 32
    cmp     r10, r8
    jb      .l_res1

    ; --------------------------------------------------------------------------
    ; Step 8: Pre-FFN RMSNorm (Gemma (1.0 + W) formula)
    ; --------------------------------------------------------------------------
    mov     rcx, rsi               ; out = x_norm
    mov     rdx, [r15 + 48]        ; weight = rms_ffn_w
    mov     r8,  [r15 + 0]         ; x = updated residual stream
    mov     r9,  [r15 + 128]       ; N = dim (2048)
    mov     dword [rsp + 32], 0x3727C5AC ; eps = 1e-5
    call    gemma_rmsnorm_avx2

    ; --------------------------------------------------------------------------
    ; Step 9: GeGLU FFN Projections
    ; gate = gemv_f16c(s_gate, w_gate_f16, x_norm, dim, hidden_dim)
    ; up   = gemv_f16c(s_up, w_up_f16, x_norm, dim, hidden_dim)
    ; --------------------------------------------------------------------------
    mov     rax, [r15 + 160]       ; smp_state
    mov     [rsp + 40], rax        ; 6th arg = smp_state

    lea     rcx, [rsi + 38912]     ; out = s_gate
    mov     rdx, [r15 + 56]        ; W = w_gate_f16
    mov     r8,  rsi               ; x = x_norm
    mov     r9,  [r15 + 128]       ; K = dim (2048)
    mov     rax, [r15 + 136]       ; M = hidden_dim (16384)
    mov     [rsp + 32], rax
    call    smp_f16c_gemv_avx2

    mov     rax, [r15 + 160]       ; smp_state
    mov     [rsp + 40], rax        ; 6th arg = smp_state
    lea     rcx, [rsi + 104448]    ; out = s_up
    mov     rdx, [r15 + 64]        ; W = w_up_f16
    mov     r8,  rsi               ; x = x_norm
    mov     r9,  [r15 + 128]       ; K = dim (2048)
    mov     rax, [r15 + 136]       ; M = hidden_dim (16384)
    mov     [rsp + 32], rax
    call    smp_f16c_gemv_avx2

    ; --------------------------------------------------------------------------
    ; Step 10: Vector GeGLU: s_gate = geglu(s_gate, s_up, hidden_dim)
    ; --------------------------------------------------------------------------
    lea     rcx, [rsi + 38912]     ; out = s_gate
    lea     rdx, [rsi + 38912]     ; gate = s_gate
    lea     r8,  [rsi + 104448]    ; up = s_up
    mov     r9,  [r15 + 136]       ; N = hidden_dim (16384)
    call    geglu_avx2

    ; --------------------------------------------------------------------------
    ; Step 11: Down Projection: res_buf = gemv_f16c(res_buf, w_down_f16, s_gate, hidden_dim, dim)
    ; --------------------------------------------------------------------------
    mov     rax, [r15 + 160]       ; smp_state
    mov     [rsp + 40], rax        ; 6th arg = smp_state
    lea     rcx, [rsi + 8192]      ; out = res_buf
    mov     rdx, [r15 + 72]        ; W = w_down_f16
    lea     r8,  [rsi + 38912]     ; x = s_gate
    mov     r9,  [r15 + 136]       ; K = hidden_dim (16384)
    mov     rax, [r15 + 128]       ; M = dim (2048)
    mov     [rsp + 32], rax
    call    smp_f16c_gemv_avx2

    ; --------------------------------------------------------------------------
    ; Step 12: Residual Connection 2: x = x + res_buf
    ; --------------------------------------------------------------------------
    mov     rcx, [r15 + 0]         ; x
    lea     rdx, [rsi + 8192]      ; res_buf
    mov     r8,  [r15 + 128]
    shl     r8,  2                 ; dim * 4 bytes
    xor     r10, r10
.l_res2:
    vmovups ymm0, [rcx + r10]
    vmovups ymm1, [rdx + r10]
    vaddps  ymm0, ymm0, ymm1
    vmovups [rcx + r10], ymm0
    add     r10, 32
    cmp     r10, r8
    jb      .l_res2

    ; Epilogue
    add     rsp, 48
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

; ==============================================================================
; Integrated Gemma Kernels (Assembled directly into layer binary)
; ==============================================================================
include '../raw_materials/mat_smp_f16c_gemv_avx2_flat.asm'
include '../raw_materials/mat_geglu_avx2_flat.asm'
include '../raw_materials/mat_gemma_rmsnorm_avx2_flat.asm'
include '../raw_materials/mat_exp_softmax_lut_flat.asm'
