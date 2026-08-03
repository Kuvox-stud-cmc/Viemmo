# Local Air-Gapped Serving & Deployment Guide

This document defines the deployment verification steps for **Viemmo-1B** in offline and air-gapped environments (Phase 9 - Issue #43).

---

## 1. Objectives & Implemented Criteria

Due to **Issue #42** (GGUF Conversion) being a prerequisite for full Ollama Modelfile packaging, the current local deployment implements and validates the following criteria:

- [x] **Derive Chat Template**: Extracted and documented the exact chat template delimiters (`<|im_start|>`, `<|im_end|>`) compatible with OLMo-2 Instruct.
- [x] **Loopback Binding Startup Script**: Created `deploy/scripts/serve_local.py` enforcing strict `127.0.0.1` API binding.
- [x] **Disable Telemetry & Update Check**: Programmatically configured environment properties (`OLLAMA_HOST`, `OLLAMA_NO_HISTORY`).
- [x] **Outbound Firewall Rule Documentation**: Documented Windows PowerShell cmdlets to prevent LAN leakages.
- [ ] **Ollama Modelfile**: *Pending completion of Issue #42 (GGUF model conversion).*

---

## 2. Server Configuration

The local deployment configuration uses:
- **Binding Address**: `127.0.0.1` (localhost only)
- **Default Port**: `11434`
- **Telemetry Restriction**: Blocks outgoing external connection attempts.

### Windows Firewall Outbound Rule
To isolate the execution environment from local network scans and updates, run the following PowerShell command in Admin mode:

```powershell
New-NetFirewallRule -DisplayName "Block Ollama Outbound" -Direction Outbound -Program "ollama.exe" -Action Block
```

---

## 3. Execution & Verification

### Running the loopback validation server:
```bash
python deploy/scripts/serve_local.py
```

### Verification Curl:
```bash
curl -X POST http://127.0.0.1:11434/v1/chat/completions -d '{"messages":[{"role":"user","content":"Xin chào"}]}'
```
*Expected Output*: Verification success confirmation demonstrating the server is active on the local loopback interface.
