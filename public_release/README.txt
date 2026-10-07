================================================================================
PULSAR · Standalone Native Inference Engine (v1.0.0-release)
================================================================================
Architected & Directed by Kuo-Ting Tsai (@tomtsai28) & Shin-Rung Tsai (@bella-tsai0123)
Technical Inquiries & Discussion / 技術交流信箱: tom.tsai28@gmail.com
Co-Engineered with Antigravity (DeepMind Agentic Co-Pilot)
Built with Gemma by Google · Physical Silicon Telemetry Verified

================================================================================
【繁體中文版說明書】
================================================================================

【一、 硬體架構需求】
• 作業系統：Windows 10 / 11 (64-bit)
• 硬體支援：x86-64 處理器，強制要求支援 AVX2 / F16C 指令集 (Intel 4 代 Core / AMD Zen 以上)
• 記憶體  ：8 GB RAM 以上 (4.3 GB 模型需常駐記憶體映射)
• 運行環境：Zero-Python、Zero-CUDA、Zero-Runtime (純原生獨立執行檔，外部 0 依賴)

【二、 文字使用範圍與操作邊界】
• 支援語系：
  - 標準 UTF-8 編碼之繁體中文（台灣標準常用字）、簡體中文、英文及標準 ASCII 程式符號。
  - 內嵌 256,000 詞元 Byte-fallback BPE 詞表，全面相容 Google Gemma 語意單元。
• 輸入長度與航情視野：
  - 建議最佳長度：單句提問或短篇指令（1 ~ 256 字元 / Tokens 最佳）。
  - 極限安全緩衝：內部硬體緩衝區上限 512 Tokens，超出自動啟動邊界截斷防護。
• 適用任務範疇 (In-Scope)：
  - 概念意圖推論、常識問答、邏輯與數值計算測試。
  - 18 層殘差流隱變數雷達飛行軌跡即時遙測 (T, S, C, D 座標)。
  - 實體暫存器 RAX Token 捕獲與 Layer 14~16 幻覺突變點確診。
• 不適用場景與限制 (Out-of-Scope)：
  - 本核心專為「單輪思維航情觀測與首段 Token 發射」設計，非多輪歷史對話聊天伺服器。
  - 不支援萬字長篇文本閱讀理解或批次文件處理。
• 輸出客觀性聲明：
  - 本版本輸出之文字為 Google Gemma-2B 裸機自回歸生成。
  - 在未外掛 PULSAR 專利業務校正替代料時，輸出包含該模型之原生固有幻覺（如展示站所示之第 15 層真實度跌落），數據僅供研究與底層遙測之用。

【三、 快速上手指南】
1. 自備權重：請將轉檔完成的 Google 官方「gemma2b_fp16.bin」(4.3 GB) 置於本執行檔同目錄下。
2. 啟動推論：開啟 CMD 或 PowerShell 終端機，執行任意提示詞：
   .\pulsar_tracer_release.exe "請問台灣加值型營業稅法定稅率是多少？"
   .\pulsar_tracer_release.exe "日本的首都 是哪"
   .\pulsar_tracer_release.exe "台灣最高峰是哪座山？"
3. 即時觀測：終端機將逐行噴出 18 層思維飛行軌跡與 RAX 暫存器捕獲紀錄！

【四、 官方技術文檔與先發技術備忘錄 (Technical Publications)】
• Technical Whitepaper (REV-035): PULSAR_Technical_Whitepaper.pdf
• Technical Note Part 2: PULSAR_Technote_Part2_NullSpace.pdf (零空間正交投影與閉環阻尼)
• Technical Note Part 3: PULSAR_Technote_Part3_TopologicalBarrier.pdf (深層殘差流形之拓撲維度障礙)
• Technical Note Part 4: PULSAR_Technote_Part4_DynamicAnchors.pdf (動態語境子空間與 18 層階梯分流機制)
• Architectural Note: PULSAR_Architecture_Guide.pdf (雙軌防呆與工廠神經架構)

【五、 技術交流與討論 (Technical Discussion)】
• 聯絡信箱：tom.tsai28@gmail.com
• 交流方向：歡迎底層架構探討、隱空間雷達遙測驗證與學術合作交流。

【六、 授權協議】
本軟體遵循 Google Gemma Terms of Use 開放模型授權協議。


================================================================================
[ ENGLISH SPECIFICATIONS & MANUAL ]
================================================================================

[I. Hardware Architecture & System Requirements]
• Operating System : Windows 10 / 11 (64-bit)
• Target CPU       : x86-64 v3 with AVX2 & F16C instruction support (Intel Haswell 4th Gen+ / AMD Zen 1st Gen+)
• System Memory    : 8 GB+ RAM (Required for physical mapping of 4.3 GB model weights)
• Runtime Engine   : Standalone Native PE64 (Zero-Python, Zero-CUDA, Zero-Runtime dependencies)

[II. Linguistic Scope & Operational Boundaries]
• Supported Languages & Charsets:
  - UTF-8 encoded Traditional Chinese, Simplified Chinese, English, digits, and standard ASCII code symbols.
  - Embedded 256,000-token byte-fallback BPE vocabulary with full Gemma multi-byte compatibility.
• Token Horizon & Prompt Length:
  - Recommended: Single-turn prompt or short directive (1 to 256 tokens / characters optimal).
  - Hardware Safety Cutoff: 512 tokens maximum input buffer; excess input is safely truncated.
• In-Scope Operational Domains:
  - Single-turn intent reasoning, factual QA, arithmetic and logic benchmark traces.
  - Real-time 18-layer residual stream radar trajectory telemetry (T, S, C, D coordinates).
  - Hardware register RAX token capture and Layer 14~16 hallucination inflection diagnosis.
• Out-of-Scope & Operational Limits:
  - Designed specifically as an inference flight telemetry probe; NOT a multi-turn chat server.
  - NOT designed for bulk long-document summarization (>10,000 tokens).
• Objective Output Disclosure:
  - Emits native autoregressive tokens from base Google Gemma-2B weights.
  - Without PULSAR proprietary business substitute materials engaged, outputs inherently exhibit the base model's unmitigated native hallucinations (e.g. Layer 15 reality plunge). Provided strictly for low-level silicon telemetry research.

[III. Quick Start Guide (Bring Your Own Compute / Model)]
1. Model Weight Setup:
   Place the converted official Google Gemma-2-2B-it FP16 model ("gemma2b_fp16.bin", ~4.3 GB) in the same directory as this executable.
2. Launch Inference:
   Open Command Prompt (CMD) or PowerShell and pass your prompt:
   .\pulsar_tracer_release.exe "What is the capital of Japan?"
   .\pulsar_tracer_release.exe "Explain the concept of quantum superposition in one sentence."
3. Observe Physical Telemetry:
   The terminal will stream real-time 18-layer latent flight coordinates and register RAX token emissions.

[IV. Technical Publications & Academic Notes]
• Technical Whitepaper (REV-035): PULSAR_Technical_Whitepaper.pdf
• Technical Note Part 2: PULSAR_Technote_Part2_NullSpace.pdf (Null-Space Projection & Closed-Loop Damping)
• Technical Note Part 3: PULSAR_Technote_Part3_TopologicalBarrier.pdf (Topological Dimension Barrier in Deep Residual Manifolds)
• Technical Note Part 4: PULSAR_Technote_Part4_DynamicAnchors.pdf (Dynamic Contextual Subspaces & Cascaded Manifold Sharding)
• Architectural Note: PULSAR_Architecture_Guide.pdf (Dual-Track Architecture & Factory Poka-Yoke)

[V. Technical Inquiries & Discussions]
• Contact Email : tom.tsai28@gmail.com
• Scope         : Technical inquiries, latent flight telemetry verification, and academic collaboration.

[VI. Authorship & Licensing]
• Authorship: Architected & Directed by Kuo-Ting Tsai (@tomtsai28) & Shin-Rung Tsai (@bella-tsai0123)
• Co-Engineered with Antigravity (DeepMind Agentic Co-Pilot)
• Built with Gemma by Google · Physical Silicon Telemetry Verified
• License: Distributed under Google Gemma Terms of Use.
================================================================================
