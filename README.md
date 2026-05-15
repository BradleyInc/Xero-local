# xero-local

A local MCP (Model Context Protocol) server that connects Claude to the Xero accounting API. It exposes tools for reading and updating draft accounts-payable bills, attachments, contacts, accounts, and tax rates — enabling Claude to process AP bills directly inside a conversation.

Also includes launcher scripts (`start-xero-mcp.js` / `start-xero-mcp.bat`) for running the official [`@xeroapi/xero-mcp-server`](https://www.npmjs.com/package/@xeroapi/xero-mcp-server) npm package with credentials loaded from the local `.env` file.

---

## Prerequisites

- Python 3.11+
- Node.js (for the official Xero MCP launcher scripts)
- A Xero app with OAuth 2.0 credentials ([create one here](https://developer.xero.com/app/manage))
  - Redirect URI must be set to `http://localhost:8080/callback`

---

## Setup

**1. Install Python dependencies**

```bash
pip install mcp[cli] requests pypdf
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

**3. Register the server with Claude Code**

Add the following to your Claude Code MCP settings (e.g. `claude_desktop_config.json` or `.mcp.json`):

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

---

## Authentication

On first use, call the `xero_authenticate` tool. This opens a browser window for Xero OAuth login and saves a token to `xero_token.json` in the project directory. The token is refreshed automatically on subsequent calls — you only need to authenticate again if the refresh token expires (typically after 60 days of inactivity).

---

## Available Tools

| Tool | Description |
|---|---|
| `xero_authenticate` | Run the OAuth 2.0 flow and save the access token |
| `xero_list_tenants` | List all connected Xero organisations |
| `xero_list_draft_accpay_invoices` | List all DRAFT AP bills for a tenant |
| `xero_get_invoice` | Get full details of a specific bill |
| `xero_get_invoice_attachments` | Retrieve attachments (PDFs extracted as text, images returned inline) |
| `xero_get_contacts` | Search for suppliers/customers by name |
| `xero_get_accounts` | Return the full chart of accounts |
| `xero_get_tax_rates` | Return all active tax rates |
| `xero_update_invoice` | Update fields and line items on a DRAFT bill |

---

## File Reference

| File | Purpose |
|---|---|
| `xero_mcp_server.py` | Custom MCP server (main entry point) |
| `start-xero-mcp.js` | Launches the official `@xeroapi/xero-mcp-server` npm package |
| `start-xero-mcp.bat` | Windows batch launcher for the npm package |
| `.env` | Your credentials — **not committed, do not share** |
| `.env.example` | Credential template safe to commit |
| `xero_token.json` | OAuth token cache — **not committed, regenerated on auth** |

---

## Security

- `.env` and `xero_token.json` are listed in `.gitignore` and will never be committed.
- Credentials are loaded at startup from the `.env` file in the project directory.
- The OAuth token is stored locally only and refreshed automatically.
