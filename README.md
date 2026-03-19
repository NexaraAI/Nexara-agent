# Nexara Agent V1.0 🤖

<div align="center">

![License](https://img.shields.io/badge/license-Apache%202.0-blue)
![Python](https://img.shields.io/badge/python-3.10%2B-brightgreen)
![Platform](https://img.shields.io/badge/platform-cross--platform-black)
![Skills](https://img.shields.io/badge/skills-86%2B-orange)
![Status](https://img.shields.io/badge/status-production--architecture-purple)

**Production-grade autonomous AI agent framework by [NexaraAI](https://github.com/NexaraAI) and DemonZ Development**

</div>

---

## Overview

Nexara V1.0 is a modular autonomous agent framework built for resilient execution across multiple environments.

It combines:

- Multi-provider LLM failover
- Semantic long-term memory
- Dynamic skill loading
- Autonomous task replanning
- Cross-platform execution awareness

Designed to operate consistently across mobile and desktop environments while loading only the capabilities supported by the current host system.

---

## Why Nexara Exists

Most autonomous agent frameworks fail in real-world deployment because they assume:

- one provider will always stay online  
- every system exposes identical tools  
- failures should terminate execution  

Nexara was built to solve that.

It routes intelligently across providers, isolates execution, and replans when skills fail instead of stopping.

---

## 🚀 Architecture Overview

```text
User Message
    │
    ▼
┌─────────────────────────────────────────────────────┐
│                    LLM Router                       │
│     Gemini → Groq → Ollama → OpenAI / NVIDIA        │
│           Automatic failover with backoff           │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│                    ReAct Loop                       │
│   THINK → ACT → OBSERVE → REPLAN (up to 14 cycles) │
│       Automatic retry on skill failure (3x)        │
└──────────────────────┬──────────────────────────────┘
                       │
         ┌─────────────┼─────────────┐
         ▼             ▼             ▼
      Skills         Memory        Planner
   Dynamic Load   SQLite + Vectors  Background Tasks
```

---

## 📦 Dynamic Skill Warehouse

Nexara pulls skills dynamically from the official skill warehouse and activates only host-compatible modules.

### Supported Platforms

#### 📱 Android (Termux)

- Native device control  
- SMS access  
- Camera integration  
- Battery telemetry  
- Call log access  

#### 🐧 Linux

- APT package management  
- systemd service control  
- Docker orchestration  
- cron automation  

#### 🪟 Windows

- PowerShell execution  
- Registry inspection  
- Event Log monitoring  

#### 🍏 macOS

- Homebrew integration  
- AppleScript execution  
- Spotlight search  

#### 🧠 Core Skills

- Web scraping  
- File operations  
- Python execution  
- SQLite queries  
- REST webhooks  

> Current warehouse size: **86+ dynamically loadable skills**

---

## 🔒 Architecture Evolution

| Feature | Beta | Nexara V1.0 |
|--------|------|-------------|
| Tool Calls | Regex JSON parsing | Native structured function calling |
| LLM Engine | Single provider | 4-tier fallback chain |
| Memory | SQLite LIKE queries | Semantic vector similarity |
| Execution | Direct `exec()` | Subprocess isolation + AST scanning |
| Autonomy | Stops on error | ReAct replanning loop |
| Scheduling | None | Natural language scheduler |

---

## ⚙️ Core Design Principles

- Host-aware execution  
- Failure tolerance  
- Minimal resource overhead  
- Expandable architecture  
- Autonomous recovery  

---

## 💬 Telegram Interface

| Command | Access | Description |
|--------|--------|-------------|
| `/start` | All | Wake agent and sync skills |
| `/memory [query]` | All | Search semantic memory |
| `/run <goal>` | Admin | Queue autonomous task |
| `/tasks` | Admin | View active tasks |
| `/monitors` | Admin | View condition monitors |
| `/llm` | Admin | Check provider health |
| `/stats` | Admin | Full host system snapshot |

---

## 📥 Installation

```bash
git clone [https://github.com/NexaraAI/nexara-agent.git](https://github.com/NexaraAI/nexara-agent.git)
cd nexara-agent
pip install -r requirements.txt
chmod +x start.sh
./start.sh

---

## 🧩 Project Structure

```text
nexara-agent/
├── agent/               # ReAct engine, semantic memory, LLM router
├── utils/               # Dynamic skill_loader and system helpers
├── main.py              # Execution logic
├── start.sh             # Automated launch script
├── requirements.txt     # Dependencies
└── .env                 # API keys & configuration
---

## 🔭 Roadmap

- Advanced multi-agent orchestration  
- Distributed skill execution  
- Expanded semantic memory indexing  
- Native dashboard interface  

---

## ⚖️ License

Licensed under **Apache License 2.0**.

See `LICENSE` for full terms.
