; ==============================================================================
; Project PULSAR-ASM | Finished Good: gemma_engine_flat.asm
; ------------------------------------------------------------------------------
; Google Gemma-2B Full-Precision Autoregressive Inference Engine (AVX2 + F16C)
;
; Complete Pure Assembly Pipeline:
;   - Embed Lookup + sqrt(2048) scaling
;   - 18 Layers of Gemma-2B (MQA Attention + RoPE rotate_half + GeGLU + (1+W) RMSNorm)
;   - Final RMSNorm
;   - 256,000-class Vocab Classifier Projection (tied embedding)
;   - Vectorized Argmax Sampler
;
; Zero C-Runtime, Zero PyTorch, 100% Native Machine Code
; ==============================================================================

use64

; GemmaEngineContext Layout (passed in RCX):
;   [RCX + 0]   : x (float* [2048], active residual stream)
;   [RCX + 8]   : logits (float* [256000])
;   [RCX + 16]  : scratch (float* [200 KB buffer])
;   [RCX + 24]  : rope_cos (float* [128])
;   [RCX + 32]  : rope_sin (float* [128])
;   [RCX + 40]  : key_cache_base (float* [18 * max_seq * 256])
;   [RCX + 48]  : val_cache_base (float* [18 * max_seq * 256])
;   [RCX + 56]  : token_emb_f16 (uint16_t* [256000 * 2048])
;   [RCX + 64]  : layer_weights_base (uint16_t* [18 * layer_bytes])
;   [RCX + 72]  : final_norm_w (float* [2048])
;   [RCX + 80]  : token_id (uint64_t)
;   [RCX + 88]  : pos (uint64_t)
;   [RCX + 96]  : max_seq (uint64_t)
;   [RCX + 112] : smp_state (SmpSharedState*)
;   [RCX + 120] : rep_tokens (uint64_t* [32], circular history buffer)
;   [RCX + 128] : rep_count (uint64_t, number of active tokens <= 32)

; ------------------------------------------------------------------------------
; Engine Entry Dispatch Table:
;   Offset 0  : jmp gemma_forward_step
;   Offset 16 : jmp smp_worker_proc
; ------------------------------------------------------------------------------
gemma_engine_entry:
    jmp     gemma_forward_step
    align   16
    jmp     smp_worker_proc
    align   16

gemma_forward_step:
    ; 1. Preserve callee-saved registers (64 bytes)
    push    rbx
    push    rbp
    push    rsi
    push    rdi
    push    r12
    push    r13
    push    r14
    push    r15

    ; 2. Stack frame: 48 bytes (shadow + arg5/6) + 168 bytes GemmaLayerParams = 216 -> 224 bytes
    sub     rsp, 224

    mov     r15, rcx               ; R15 = pointer to GemmaEngineContext

    ; --------------------------------------------------------------------------
    ; Step 1: Token Embedding Lookup + sqrt(dim) scaling
    ; embed_scale = sqrt(2048) = 45.254834f (0x423504F3)
    ; --------------------------------------------------------------------------
    mov     rax, [r15 + 80]        ; RAX = token_id
    ; Offset in FP16 table = token_id * 2048 * 2 = token_id * 4096 bytes
    shl     rax, 12                ; RAX = token_id * 4096
    mov     rdx, [r15 + 56]        ; token_emb_f16 base
    add     rdx, rax               ; RDX = src token embedding (2048 FP16 values)

    mov     rcx, [r15 + 0]         ; RCX = x (2048 floats destination)
    mov     eax, 0x423504F3        ; 45.254834f = sqrt(2048)
    vmovd   xmm7, eax
    vbroadcastss ymm7, xmm7        ; YMM7 = embed_scale

    xor     r10, r10               ; element index 0 .. 2047
.l_embed_loop:
    ; Load 8 FP16 weights, decompress via F16C to 8 FP32, scale by sqrt(dim)
    vcvtph2ps   ymm0, [rdx + r10 * 2]
    vmulps      ymm0, ymm0, ymm7
    vmovups     [rcx + r10 * 4], ymm0
    add     r10, 8
    cmp     r10, 2048
    jb      .l_embed_loop

    ; --------------------------------------------------------------------------
    ; Step 2: Loop through 18 Transformer Layers
    ; --------------------------------------------------------------------------
    ; Layer byte sizes in FP16 (dim=2048, hidden=16384, head_dim=256):
    ; w_q:    2048 * 2048 * 2 = 8,388,608 B (8 MB)
    ; w_k:    256 * 2048 * 2  = 1,048,576 B (1 MB)
    ; w_v:    256 * 2048 * 2  = 1,048,576 B (1 MB)
    ; w_o:    2048 * 2048 * 2 = 8,388,608 B (8 MB)
    ; w_gate: 16384 * 2048 * 2 = 67,108,864 B (64 MB)
    ; w_up:   16384 * 2048 * 2 = 67,108,864 B (64 MB)
    ; w_down: 2048 * 16384 * 2 = 67,108,864 B (64 MB)
    ; Total FP16 weights per layer = 220,200,960 Bytes (~210.0 MB)
    ; Norm weights: rms_att_w (8 KB), rms_ffn_w (8 KB) = 16 KB

    xor     r12, r12               ; R12 = layer index L = 0 .. 17
.l_layer_loop:
    ; Construct GemmaLayerParams at [rsp + 48] (shadow + args are at [rsp + 0..47])
    lea     rbp, [rsp + 48]

    mov     rax, [r15 + 0]
    mov     [rbp + 0],   rax       ; x
    mov     rax, [r15 + 16]
    mov     [rbp + 112], rax       ; scratch
    mov     rax, [r15 + 24]
    mov     [rbp + 96],  rax       ; rope_cos
    mov     rax, [r15 + 32]
    mov     [rbp + 104], rax       ; rope_sin
    mov     rax, [r15 + 88]
    mov     [rbp + 120], rax       ; pos
    mov     qword [rbp + 128], 2048 ; dim
    mov     qword [rbp + 136], 16384 ; hidden_dim
    mov     qword [rbp + 144], 8    ; n_heads
    mov     qword [rbp + 152], 256  ; head_dim
    mov     rax, [r15 + 112]
    mov     [rbp + 160], rax       ; smp_state

    ; Compute KV cache offsets for layer L:
    ; layer_kv_stride = max_seq * 256 * 4 bytes
    mov     rax, [r15 + 96]        ; max_seq
    shl     rax, 10                ; max_seq * 1024 bytes (256 * 4)
    imul    rax, r12               ; L * layer_kv_stride
    mov     rcx, [r15 + 40]
    add     rcx, rax
    mov     [rbp + 80],  rcx       ; key_cache for layer L
    mov     rcx, [r15 + 48]
    add     rcx, rax
    mov     [rbp + 88],  rcx       ; val_cache for layer L

    ; Compute weight pointers for layer L:
    ; Base of layer L = layer_weights_base + L * layer_stride
    ; layer_stride = 220,217,344 bytes (~210.01 MB including norm weights)
    mov     rax, 220217344
    imul    rax, r12
    mov     rbx, [r15 + 64]
    add     rbx, rax               ; RBX = layer L weight base

    ; Assign pointers
    lea     rax, [rbx + 0]
    mov     [rbp + 8],   rax       ; rms_att_w (8192 bytes FP32)
    lea     rax, [rbx + 8192]
    mov     [rbp + 48],  rax       ; rms_ffn_w (8192 bytes FP32)

    lea     rax, [rbx + 16384]
    mov     [rbp + 16],  rax       ; w_q (8MB)
    add     rax, 8388608
    mov     [rbp + 24],  rax       ; w_k (1MB)
    add     rax, 1048576
    mov     [rbp + 32],  rax       ; w_v (1MB)
    add     rax, 1048576
    mov     [rbp + 40],  rax       ; w_o (8MB)
    add     rax, 8388608
    mov     [rbp + 56],  rax       ; w_gate (64MB)
    add     rax, 67108864
    mov     [rbp + 64],  rax       ; w_up (64MB)
    add     rax, 67108864
    mov     [rbp + 72],  rax       ; w_down (64MB)

    ; Execute Layer L
    mov     rcx, rbp               ; RCX = pointer to GemmaLayerParams
    call    sub_gemma_layer

    inc     r12
    cmp     r12, [r15 + 104]       ; 18 layers
    jb      .l_layer_loop

    ; --------------------------------------------------------------------------
    ; Step 3: Final RMSNorm
    ; x = gemma_rmsnorm(x, final_norm_w, dim=2048, eps=1e-5)
    ; --------------------------------------------------------------------------
    mov     rcx, [r15 + 0]         ; out = x
    mov     rdx, [r15 + 72]        ; weight = final_norm_w
    mov     r8,  [r15 + 0]         ; x = x
    mov     r9,  2048              ; N = 2048
    mov     dword [rsp + 32], 0x3727C5AC ; eps = 1e-5
    call    gemma_rmsnorm_avx2

    ; --------------------------------------------------------------------------
    ; Step 4: Classifier Projection (Tied Embedding) via 4-Core SMP
    ; logits = smp_f16c_gemv_avx2(logits, token_emb_f16, x, dim=2048, M=256000, smp_state)
    ; --------------------------------------------------------------------------
    mov     rcx, [r15 + 8]         ; out = logits
    mov     rdx, [r15 + 56]        ; W = token_emb_f16
    mov     r8,  [r15 + 0]         ; x = final normalized x
    mov     r9,  2048              ; K = 2048
    mov     qword [rsp + 32], 256000 ; M = vocab_size (256,000)
    mov     rax, [r15 + 112]       ; smp_state
    mov     [rsp + 40], rax        ; 6th arg = smp_state
    call    smp_f16c_gemv_avx2

    ; --------------------------------------------------------------------------
    ; Step 4.5: L1-Cache Resident Repetition Penalty
    ; Suppresses logits of recent tokens by 1.5f to eliminate attractor loops
    ; --------------------------------------------------------------------------
    mov     r8,  [r15 + 120]       ; rep_tokens
    mov     r9,  [r15 + 128]       ; rep_count
    test    r8,  r8
    jz      .skip_rep_penalty
    test    r9,  r9
    jz      .skip_rep_penalty

    mov     rcx, [r15 + 8]         ; logits base
    mov     eax, 0x3FC00000        ; 1.5f in IEEE-754
    vmovd   xmm1, eax
    xor     r10, r10
.l_rep_penalty_loop:
    mov     r11, [r8 + r10 * 8]    ; tid = rep_tokens[r10]
    cmp     r11, 256000
    jae     .next_rep_step
    vmovss  xmm0, [rcx + r11 * 4]
    vsubss  xmm0, xmm0, xmm1
    vmovss  [rcx + r11 * 4], xmm0
.next_rep_step:
    inc     r10
    cmp     r10, r9
    jb      .l_rep_penalty_loop

.skip_rep_penalty:
    ; --------------------------------------------------------------------------
    ; Step 5: Argmax Sampler
    ; next_token = argmax(logits, 256000)
    ; --------------------------------------------------------------------------
    mov     rcx, [r15 + 8]         ; logits
    mov     rdx, 256000            ; N = vocab_size
    call    sampler_argmax_avx2
    ; RAX = predicted token ID

    ; Epilogue
    add     rsp, 224
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
; Integrated Gemma Sub-Assemblies & Kernels
; ==============================================================================
include '../sub_assemblies/sub_gemma_layer_flat.asm'
include '../sub_assemblies/sub_sampler_argmax_flat.asm'
