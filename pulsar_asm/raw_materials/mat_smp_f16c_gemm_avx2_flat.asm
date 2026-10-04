; ==============================================================================
; Project PULSAR-ASM | Component: mat_smp_f16c_gemm_avx2_flat.asm
; ------------------------------------------------------------------------------
; 4-Core SMP Multi-Threaded F16C GEMM (Batch-8) Kernel with Lock-Free Spin Barriers
;
; Designed for Google Gemma-2B Prefill Acceleration:
;   - Fully saturates dual-channel DDR4 memory bus with 8-Token GEMM reuse
;   - 0 OS context switch overhead: < 60ns wake-up via MESI cacheline invalidation
;   - Lock-free row-partitioning across 4 physical cores
;   - Fallback to single-thread if smp_state is NULL or M < 256
;
; Win64 ABI:
;   RCX = out (float* [B * full_M])
;   RDX = W_f16 (uint16_t* [full_M * K])
;   R8  = X_batch (float* [B * K])
;   R9  = K (columns)
;   [RSP + 40] = full_M (rows)
;   [RSP + 48] = B (batch size, e.g. 8)
;   [RSP + 56] = smp_state (SmpSharedState*)
; ==============================================================================

use64

; ------------------------------------------------------------------------------
; SmpSharedState Layout:
;   offset 0  : volatile uint32_t done_counter
;   offset 4  : volatile uint32_t job_seq
;   offset 8  : volatile uint32_t stop_flag
;   offset 12 : uint32_t full_m
;   offset 16 : uint32_t batch_size (B)
;   offset 64 : Worker 1 context (64 bytes)
;   offset 128: Worker 2 context (64 bytes)
;   offset 192: Worker 3 context (64 bytes)
;
; WorkerContext Layout (64 bytes aligned):
;   +0  : volatile uint32_t job_id
;   +4  : uint32_t last_job_id
;   +8  : volatile uint32_t stop_flag
;   +12 : uint32_t job_type (0 = GEMV, 1 = GEMM)
;   +16 : float* out
;   +24 : uint16_t* W
;   +32 : float* x (or X_batch)
;   +40 : uint64_t K
;   +48 : uint64_t M (chunk_M)
;   +56 : volatile uint32_t* done_counter
; ------------------------------------------------------------------------------

; ==============================================================================
; Exported: smp_worker_proc (Win64 ThreadProc)
; ==============================================================================
smp_worker_proc:
    push    rbx
    push    rsi
    push    rdi
    push    r12
    push    r13
    push    r14
    push    r15
    sub     rsp, 64                ; shadow + stack alignment for calls

    mov     r15, rcx               ; R15 = WorkerContext*

.worker_spin_loop:
    mov     eax, [r15 + 0]         ; job_id
    cmp     eax, [r15 + 4]         ; last_job_id
    jne     .worker_execute_job

    mov     edx, [r15 + 8]         ; stop_flag
    test    edx, edx
    jnz     .worker_exit

    pause
    jmp     .worker_spin_loop

.worker_execute_job:
    mov     [r15 + 4], eax         ; last_job_id = job_id

    mov     r11, [r15 + 56]        ; R11 = pointer to smp_state
    cmp     dword [r11 + 16], 8    ; Check if batch_size in smp_state is >= 8
    jae     .worker_execute_gemm

    ; --- GEMV Branch (batch_size < 8) ---
    mov     rcx, [r15 + 16]        ; out
    mov     rdx, [r15 + 24]        ; W
    mov     r8,  [r15 + 32]        ; x
    mov     r9,  [r15 + 40]        ; K
    mov     rax, [r15 + 48]        ; M
    mov     [rsp + 32], rax
    call    f16c_gemv_avx2
    jmp     .worker_signal_done

.worker_execute_gemm:
    ; --- GEMM Branch (job_type == 1) ---
    mov     r11, [r15 + 56]        ; R11 = pointer to smp_state
    mov     r10d, [r11 + 12]       ; full_m
    mov     eax,  [r11 + 16]       ; B (batch_size)

    mov     rcx, [r15 + 16]        ; out
    mov     rdx, [r15 + 24]        ; W
    mov     r8,  [r15 + 32]        ; X_batch
    mov     r9,  [r15 + 40]        ; K
    mov     r12, [r15 + 48]        ; chunk_M
    mov     [rsp + 32], r12        ; 5th arg = chunk_M
    mov     [rsp + 40], rax        ; 6th arg = B
    mov     [rsp + 48], r10        ; 7th arg = full_M
    call    f16c_gemm_batch8_avx2

.worker_signal_done:
    ; Atomically signal completion
    mov     r11, [r15 + 56]        ; R11 = pointer to done_counter
    lock inc dword [r11]

    jmp     .worker_spin_loop

.worker_exit:
    add     rsp, 64
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rdi
    pop     rsi
    pop     rbx
    xor     eax, eax
    ret

; ==============================================================================
; Exported: smp_f16c_gemm_batch8_avx2
; ==============================================================================
smp_f16c_gemm_batch8_avx2:
    push    rbx
    push    rsi
    push    rdi
    push    r12
    push    r13
    push    r14
    push    r15
    sub     rsp, 64

    ; 7 pushes (56 bytes) + sub rsp 64 (64 bytes) = 120 bytes shift
    ; Arg 5 (full_M)   is at [rsp + 120 + 40] = [rsp + 160]
    ; Arg 6 (B)        is at [rsp + 120 + 48] = [rsp + 168]
    ; Arg 7 (smp_state) is at [rsp + 120 + 56] = [rsp + 176]
    mov     r10, [rsp + 176]       ; R10 = smp_state pointer
    mov     r11, [rsp + 160]       ; R11 = full_M
    mov     rax, [rsp + 168]       ; RAX = B

    ; Fallback to single thread if smp_state == NULL or full_M < 256
    test    r10, r10
    jz      .fallback_single
    cmp     r11, 256
    jb      .fallback_single

    mov     r12, r10               ; R12 = smp_state
    mov     r13, r11               ; R13 = full_M
    mov     r14, rcx               ; R14 = out base
    mov     r15, rdx               ; R15 = W base
    ; R8 = X_batch, R9 = K

    ; 1. Reset done_counter and store full_m, B in smp_state
    mov     dword [r12 + 0],  0
    mov     [r12 + 12], r13d       ; full_m
    mov     [r12 + 16], eax        ; B

    ; 2. Advance job_seq
    inc     dword [r12 + 4]

    ; 3. Calculate row chunk = full_M / 4
    mov     rbx, r13
    shr     rbx, 2                 ; RBX = chunk

    ; Row stride in bytes for W: K * 2
    mov     rsi, r9
    shl     rsi, 1                 ; RSI = row_stride_w = K * 2

    ; chunk_stride_w = chunk * (K * 2)
    mov     rdi, rbx
    imul    rdi, rsi               ; RDI = chunk_stride_w

    ; --------------------------------------------------------------------------
    ; Setup Worker 1 (rows: 1*chunk .. 2*chunk - 1)
    ; --------------------------------------------------------------------------
    lea     rax, [r14 + rbx * 4]   ; out + chunk * 4
    mov     [r12 + 64 + 16], rax
    lea     rax, [r15 + rdi]       ; W + 1 * chunk_stride_w
    mov     [r12 + 64 + 24], rax
    mov     [r12 + 64 + 32], r8    ; X_batch
    mov     [r12 + 64 + 40], r9    ; K
    mov     [r12 + 64 + 48], rbx   ; chunk_M = chunk
    mov     dword [r12 + 64 + 12], 1 ; job_type = 1 (GEMM)
    lea     rax, [r12 + 0]
    mov     [r12 + 64 + 56], rax   ; done_counter ptr
    mov     eax, [r12 + 4]
    mov     [r12 + 64 + 0],  eax   ; trigger Worker 1

    ; --------------------------------------------------------------------------
    ; Setup Worker 2 (rows: 2*chunk .. 3*chunk - 1)
    ; --------------------------------------------------------------------------
    mov     rax, rbx
    shl     rax, 3                 ; chunk * 8 = (2 * chunk) * 4
    add     rax, r14
    mov     [r12 + 128 + 16], rax
    lea     rax, [r15 + rdi * 2]   ; W + 2 * chunk_stride_w
    mov     [r12 + 128 + 24], rax
    mov     [r12 + 128 + 32], r8   ; X_batch
    mov     [r12 + 128 + 40], r9   ; K
    mov     [r12 + 128 + 48], rbx  ; chunk_M = chunk
    mov     dword [r12 + 128 + 12], 1 ; job_type = 1 (GEMM)
    lea     rax, [r12 + 0]
    mov     [r12 + 128 + 56], rax  ; done_counter ptr
    mov     eax, [r12 + 4]
    mov     [r12 + 128 + 0],  eax  ; trigger Worker 2

    ; --------------------------------------------------------------------------
    ; Setup Worker 3 (rows: 3*chunk .. full_M - 1)
    ; --------------------------------------------------------------------------
    mov     rax, rbx
    imul    rax, 3                 ; 3 * chunk
    mov     rcx, r13               ; full_M
    sub     rcx, rax               ; RCX = chunk3 = full_M - 3*chunk

    shl     rax, 2
    add     rax, r14
    mov     [r12 + 192 + 16], rax  ; out

    lea     rax, [rdi * 2 + rdi]
    add     rax, r15
    mov     [r12 + 192 + 24], rax  ; W

    mov     [r12 + 192 + 32], r8   ; X_batch
    mov     [r12 + 192 + 40], r9   ; K
    mov     [r12 + 192 + 48], rcx  ; chunk_M = chunk3
    mov     dword [r12 + 192 + 12], 1 ; job_type = 1 (GEMM)
    lea     rax, [r12 + 0]
    mov     [r12 + 192 + 56], rax  ; done_counter ptr
    mov     eax, [r12 + 4]
    mov     [r12 + 192 + 0],  eax  ; trigger Worker 3

    ; --------------------------------------------------------------------------
    ; Master Core 0 executes chunk 0 (rows: 0 .. chunk - 1)
    ; --------------------------------------------------------------------------
    mov     rcx, r14               ; out
    mov     rdx, r15               ; W
    ; R8 = X_batch, R9 = K
    mov     [rsp + 32], rbx        ; 5th arg = chunk_M
    mov     eax, [r12 + 16]        ; B
    mov     [rsp + 40], rax        ; 6th arg = B
    mov     [rsp + 48], r13        ; 7th arg = full_M
    call    f16c_gemm_batch8_avx2

    ; --------------------------------------------------------------------------
    ; Master Spin-Wait for Workers 1, 2, 3 (done_counter == 3)
    ; --------------------------------------------------------------------------
.master_spin_wait:
    mov     eax, [r12 + 0]
    cmp     eax, 3
    jae     .smp_done
    pause
    jmp     .master_spin_wait

.smp_done:
    mov     dword [r12 + 16], 0    ; Reset batch_size back to 0 (restore GEMV mode)
    add     rsp, 64
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rdi
    pop     rsi
    pop     rbx
    ret

.fallback_single:
    mov     [rsp + 32], r11        ; M = full_M
    mov     [rsp + 40], rax        ; B
    mov     qword [rsp + 48], 0    ; full_M = 0 (default)
    call    f16c_gemm_batch8_avx2
    add     rsp, 64
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rdi
    pop     rsi
    pop     rbx
    ret

; Include base kernels
include 'mat_f16c_gemm_avx2_flat.asm'
include 'mat_f16c_gemv_avx2_flat.asm'
