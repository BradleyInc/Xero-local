# xero-local

A toolkit for AI-assisted accounts payable processing against the Xero API. Three usage modes are supported:

| Mode | Entry point | AI backend |
|---|---|---|
| **MCP server** | `xero_mcp_server.py` | Claude Code (interactive) |
| **Standalone agent — Claude** | `xero_ap_agent.py` | Anthropic API (claude-opus-4-7) |
| **Standalone agent — Ollama** | `xero_ap_agent_ollama.py` | Local Ollama models |

---

## Prerequisites

- Python 3.11+
- A Xero app with OAuth 2.0 credentials ([create one here](https://developer.xero.com/app/manage))
  - Redirect URI must be set to `http://localhost:8080/callback`
- **Claude mode only**: `ANTHROPIC_API_KEY` environment variable set
- **Ollama mode only**: [Ollama](https://ollama.com) installed and running (minimum version 0.5.0)
- **MCP server + official launcher only**: Node.js

---

## Setup

**1. Install Python dependencies**

For the MCP server only:
```bash
pip install mcp[cli] requests pypdf
```

For the standalone Claude agent:
```bash
pip install anthropic pypdf requests customtkinter
```

For the standalone Ollama agent:
```bash
pip install ollama pypdf requests customtkinter
```

**2. Configure credentials**

Copy `.env.example` to `.env` and fill in your Xero app credentials:

```bash
cp .env.example .env
```

```env
XERO_CLIENT_ID=your_client_id_here
XERO_CLIENT_SECRET=your_client_secret_here
XERO_SCOPES=offline_access accounting.transactions accounting.contacts accounting.settings accounting.reports.read payroll.settings payroll.employees payroll.timesheets
```

**3. Authenticate with Xero**

On first use, either call the `xero_authenticate` MCP tool (interactive mode) or run any agent script — it will detect the missing token and open a browser for OAuth login automatically. The token is saved to `xero_token.json` and refreshed automatically. You only need to re-authenticate if the refresh token expires (typically after 60 days of inactivity).

---

## MCP Server Mode

The MCP server exposes Xero as tools that Claude Code can call during a conversation.

**Register with Claude Code** by adding to your `.mcp.json` or `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "xero-local": {
      "command": "python",
      "args": ["C:/path/to/xero-local/xero_mcp_server.py"]
    }
  }
}
```

A `xero-ap-bookkeeper` sub-agent (`.claude/agents/xero-ap-bookkeeper.md`) is also included. It is invoked automatically by Claude Code when you ask it to process draft AP bills.

### Available MCP Tools

| Tool | Description |
|---|---|
| `xero_authenticate` | Run the OAuth 2.0 flow and save the access token |
| `xero_list_tenants` | List all connected Xero organisations |
| `xero_list_draft_accpay_invoices` | List all DRAFT AP bills for a tenant |
| `xero_get_invoice` | Get full details of a specific bill |
| `xero_get_invoice_attachments` | Retrieve attachments (PDFs extracted as text, images returned inline) |
| `xero_get_contacts` | Search for suppliers/customers by name |
| `xero_create_contact` | Create a new supplier contact |
| `xero_get_accounts` | Return the full chart of accounts |
| `xero_get_tax_rates` | Return all active tax rates |
| `xero_update_invoice` | Update fields and line items on a DRAFT bill |

---

## Standalone Agent — Claude (`xero_ap_agent.py`)

A hybrid Python/AI agent that processes all draft AP bills autonomously.

**Architecture**: Python handles all orchestration and Xero API calls (zero AI tokens for data retrieval). Claude is called **once per bill** for reasoning only — reading the attachment, extracting fields, assigning account codes, and applying GST treatment.

**Features**:
- GUI for selecting which organisations to process, showing live draft bill counts
- Prompt caching: chart of accounts and tax rates are cached across bills for the same organisation
- Adaptive thinking via `claude-opus-4-7`
- Structured JSON output (validated against a Pydantic schema)
- Automatic contact creation when a supplier does not exist in Xero
- PDF text extraction and image attachments passed directly to the model
- Per-bill error handling — failures skip to the next bill without stopping the run

```bash
python xero_ap_agent.py
```

Requires `ANTHROPIC_API_KEY` to be set in your environment.

---

## Standalone Agent — Ollama (`xero_ap_agent_ollama.py`)

Same hybrid architecture as the Claude agent, but uses a locally-running Ollama model — bringing API token costs to zero.

**Features**:
- GUI for selecting both the Ollama model and the organisations to process
- Live model list from Ollama, showing parameter size and disk size
- Vision support detection: image attachments are passed to the model only when the selected model supports vision (e.g. `llava`, `llama3.2-vision`, `qwen2.5-vl`, `gemma3`)
- Two-phase reasoning: tool loop (contact search/creation) followed by a separate structured JSON call

**Recommended models** (pull with `ollama pull <model>`):

| Model | Notes |
|---|---|
| `llama3.1` | 8B — fast, good balance |
| `qwen2.5:14b` | 14B — excellent for structured tasks |
| `mistral` | 7B — lightweight, reliable tool use |
| `llama3.1:70b` | 70B — best quality, needs ~48 GB RAM |

```bash
python xero_ap_agent_ollama.py
```

Ollama must be installed and running before launching.

---

## Official Xero MCP Launcher (optional)

`start-xero-mcp.js` and `start-xero-mcp.bat` launch the official [`@xeroapi/xero-mcp-server`](https://www.npmjs.com/package/@xeroapi/xero-mcp-server) npm package with credentials loaded from the local `.env` file. These are an alternative to `xero_mcp_server.py` if you prefer the official package.

---

## File Reference

| File | Purpose |
|---|---|
| `xero_mcp_server.py` | Custom MCP server — main library used by all modes |
| `xero_ap_agent.py` | Standalone agent using the Anthropic Claude API |
| `xero_ap_agent_ollama.py` | Standalone agent using local Ollama models |
| `.claude/agents/xero-ap-bookkeeper.md` | Claude Code sub-agent definition for interactive AP processing |
| `start-xero-mcp.js` | Node.js launcher for the official `@xeroapi/xero-mcp-server` package |
| `start-xero-mcp.bat` | Windows batch launcher for the same |
| `.env` | Your credentials — **not committed, do not share** |
| `.env.example` | Credential template safe to commit |
| `xero_token.json` | OAuth token cache — **not committed, regenerated on auth** |

---

## Security

- `.env` and `xero_token.json` are listed in `.gitignore` and will never be committed.
- Credentials are loaded at startup from the `.env` file in the project directory.
- The OAuth token is stored locally only and refreshed automatically.
- Neither agent approves bills — all processing leaves bills in DRAFT status for client review.
