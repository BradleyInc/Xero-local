"""
Xero AP/GST Processing Agent — Ollama Mode

Same hybrid Python/Xero orchestration as xero_ap_agent.py, but uses the
Ollama Python library for reasoning instead of the Anthropic Claude API.

Supports any model available via Ollama — local models (llama3.1, qwen2.5,
mistral, etc.) or models served by a remote Ollama host (cloud GPU, shared
server, etc.) — with no per-token API costs.

Usage:
    python xero_ap_agent_ollama.py

Requirements:
    pip install ollama pypdf requests customtkinter
    Ollama must be installed and running  →  https://ollama.com
    Minimum Ollama version: 0.5.0  (structured JSON output support)

Recommended models (tool calling + good instruction following):
    ollama pull llama3.1          # 8B — fast, good balance
    ollama pull qwen2.5:14b       # 14B — excellent for structured tasks
    ollama pull mistral           # 7B — lightweight, reliable tool use
    ollama pull llama3.1:70b      # 70B — best quality, needs 48GB+ RAM
"""

import json
import os

import customtkinter as ctk
import ollama
from pydantic import BaseModel
from typing import Literal

from xero_mcp_server import (
    _get_token,
    _log,
    xero_authenticate,
    xero_create_contact,
    xero_get_accounts,
    xero_get_contacts,
    xero_get_invoice,
    xero_get_invoice_attachments,
    xero_get_tax_rates,
    xero_list_draft_accpay_invoices,
    xero_list_tenants,
    xero_update_invoice,
)


# ── Structured output schema ──────────────────────────────────────────────────
# TODO: LineItem, BillDecision, and _DECISION_SCHEMA are duplicated in
#       xero_ap_agent.py — extract to a shared xero_ap_shared.py module.

class LineItem(BaseModel):
    Description: str
    AccountCode: str
    UnitAmount: float
    TaxType: Literal["INPUT2", "EXEMPTEXPENSES", "BASEXCLUDED"]
    Quantity: float = 1.0


class BillDecision(BaseModel):
    action: Literal["update", "query"]
    contact_name: str | None = None
    date: str | None = None
    due_date: str | None = None
    reference: str | None = None
    line_items: list[LineItem] | None = None
    query: str | None = None


_DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["update", "query"]},
        "contact_name": {"type": "string"},
        "date": {"type": "string", "description": "Invoice date YYYY-MM-DD"},
        "due_date": {"type": "string", "description": "Due date YYYY-MM-DD"},
        "reference": {"type": "string", "description": "Supplier invoice number"},
        "line_items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "Description": {"type": "string"},
                    "AccountCode": {"type": "string"},
                    "UnitAmount": {"type": "number"},
                    "TaxType": {
                        "type": "string",
                        "enum": ["INPUT2", "EXEMPTEXPENSES", "BASEXCLUDED"],
                    },
                    "Quantity": {"type": "number"},
                },
                "required": [
                    "Description", "AccountCode", "UnitAmount", "TaxType", "Quantity",
                ],
                "additionalProperties": False,
            },
        },
        "query": {"type": "string"},
    },
    "required": ["action"],
    "additionalProperties": False,
}


# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are an expert Australian accounts payable bookkeeper.

You will receive:
- A Xero draft bill (JSON)
- The bill's attachments (supplier invoice documents as text or images)
- The tenant's chart of accounts
- The tenant's active tax rates

## Your task

Use the provided tools to gather information, then return a structured JSON decision.

### Step 1 — Verify the supplier contact
Always call search_contacts first to find the supplier's exact name in Xero.
If search_contacts returns no results, call create_contact to create the supplier.

### Step 2 — Return your final JSON decision

Return one of these two structures:

**action = "update"** — when you can determine all details:
{
  "action": "update",
  "contact_name": "Exact name returned by Xero",
  "date": "YYYY-MM-DD",
  "due_date": "YYYY-MM-DD",
  "reference": "Supplier invoice number from the document",
  "line_items": [
    {
      "Description": "Brief description of what was purchased",
      "AccountCode": "code from chart of accounts",
      "UnitAmount": 110.00,
      "TaxType": "INPUT2",
      "Quantity": 1.0
    }
  ]
}

**action = "query"** — when you cannot determine the bill:
{
  "action": "query",
  "query": "Clear description of what is missing or ambiguous"
}

## Rules
- Always call search_contacts before returning a contact_name — never guess.
- If search_contacts returns no results, call create_contact before returning.
- Use only account codes from the provided chart of accounts.
- UnitAmount must match the bill's LineAmountTypes:
    * INCLUSIVE → amount includes GST (e.g. $110.00 for $100 + $10 GST)
    * EXCLUSIVE → amount excludes GST (e.g. $100.00 net)
- TaxType must be exactly one of:
    * INPUT2          — GST on Expenses: standard taxable business expenses
    * EXEMPTEXPENSES  — GST Free Expenses: medical, food, exported services
    * BASEXCLUDED     — BAS Excluded: wages, super, ATO payments, bank fees
- Round all amounts to 2 decimal places.
- Use the invoice date from the document — never use today's date.
- If no due date is stated, add 30 days to the invoice date.
- Do not split a single charge across multiple lines unless the document itemises it separately.
"""


# ── Ollama tool definitions (OpenAI / Ollama format) ─────────────────────────

_OLLAMA_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_contacts",
            "description": "Search for a supplier contact in Xero by name or partial name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_term": {
                        "type": "string",
                        "description": "Supplier name or partial name to search for.",
                    }
                },
                "required": ["search_term"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_contact",
            "description": (
                "Create a new supplier contact in Xero. "
                "Only call this after search_contacts returns no results."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Full supplier name as it appears on the invoice.",
                    },
                    "email": {
                        "type": "string",
                        "description": "Supplier email address (optional).",
                    },
                },
                "required": ["name"],
                "additionalProperties": False,
            },
        },
    },
]


# ── Within-run contact cache ──────────────────────────────────────────────────

_contacts_created_this_run: dict[tuple[str, str], str] = {}


# ── Vision model detection ────────────────────────────────────────────────────

_VISION_KEYWORDS = {
    "llava", "llama3.2-vision", "llama3.2:11b-vision", "llama3.2:90b-vision",
    "moondream", "minicpm-v", "cogvlm", "qwen2-vl", "qwen2.5-vl",
    "gemma3", "mistral-small3.1",
}


def _supports_vision(model_name: str) -> bool:
    name_lower = model_name.lower()
    return any(kw in name_lower for kw in _VISION_KEYWORDS)


# ── Ollama host default (overridable via OLLAMA_HOST env var) ─────────────────

_DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


# ── Attachment size limits ────────────────────────────────────────────────────

_MAX_TEXT_CHARS = 15_000
_MAX_IMAGE_B64_LEN = 1_500_000
_MAX_TOTAL_CHARS = 80_000


# ── Attachment fetcher (Ollama format) ────────────────────────────────────────

def _fetch_attachment_content(tenant_id: str, invoice_id: str) -> tuple[str, list[str]]:
    """
    Fetch attachments and return (combined_text, [base64_images]) for Ollama.
    Images are only included when the selected model supports vision.
    """
    mcp_results = xero_get_invoice_attachments(tenant_id, invoice_id)
    text_parts: list[str] = []
    images: list[str] = []
    total_chars = 0

    for item in mcp_results:
        if total_chars >= _MAX_TOTAL_CHARS:
            _log("  Attachment total limit reached — skipping remaining")
            break

        if item.type == "text":
            text = item.text
            if len(text) > _MAX_TEXT_CHARS:
                print(f"  WARNING: text attachment truncated ({len(text)} chars) — review manually")
                text = text[:_MAX_TEXT_CHARS] + "\n[... attachment truncated due to size ...]"
            remaining = _MAX_TOTAL_CHARS - total_chars
            if len(text) > remaining:
                print(f"  WARNING: attachment cut at total limit — review manually")
                text = text[:remaining] + "\n[... attachment truncated at total limit ...]"
            text_parts.append(text)
            total_chars += len(text)

        elif item.type == "image":
            if len(item.data) > _MAX_IMAGE_B64_LEN:
                print(f"  WARNING: image attachment skipped (too large) — review manually")
                text_parts.append("[Image attachment skipped — file too large to include]")
                total_chars += 60
            else:
                images.append(item.data)
                total_chars += len(item.data)

    combined_text = "\n\n".join(text_parts) if text_parts else "No attachments found on this bill."
    return combined_text, images


# ── Tool dispatcher ───────────────────────────────────────────────────────────

def _dispatch_tool(tenant_id: str, tool_call) -> str:
    """Execute a tool call from the model and return the result as a string."""
    name = tool_call.function.name
    args = tool_call.function.arguments  # dict in Ollama Python library

    if name == "search_contacts":
        search_term = args.get("search_term", "")
        _log(f"  search_contacts({search_term!r})")
        return xero_get_contacts(tenant_id, search_term)

    if name == "create_contact":
        contact_name = args.get("name", "")
        email = args.get("email")
        cache_key = (tenant_id, contact_name.lower().strip())

        if cache_key in _contacts_created_this_run:
            _log(f"  create_contact({contact_name!r}): using cached result from this run")
            return _contacts_created_this_run[cache_key]

        _log(f"  create_contact({contact_name!r}, email={email!r})")
        result = xero_create_contact(tenant_id, contact_name, email)
        _contacts_created_this_run[cache_key] = result

        # Verify the contact round-trips in Xero search before proceeding.
        verify_raw = xero_get_contacts(tenant_id, contact_name)
        try:
            if json.loads(verify_raw):
                _log("  create_contact: round-trip search verified OK")
            else:
                _log("  create_contact: round-trip search empty (Xero indexing lag) — proceeding")
        except Exception:
            pass

        return result

    return json.dumps({"error": f"Unknown tool: {name}"})


# ── Reasoning call (Ollama, two-phase) ────────────────────────────────────────

def _reason_about_bill(
    model: str,
    client: ollama.Client,
    tenant_id: str,
    invoice: dict,
    attachment_text: str,
    attachment_images: list[str],
    accounts_json: str,
    tax_rates_json: str,
) -> BillDecision:
    """
    Two-phase Ollama reasoning per bill:

    Phase 1 — Tool loop: model calls search_contacts / create_contact as needed.
               Keeps iterating until the model stops requesting tools (max 8 turns).

    Phase 2 — Structured decision: a single final call with format=_DECISION_SCHEMA
               produces the validated JSON BillDecision.
    """
    user_text = (
        f"## Chart of accounts\n{accounts_json}\n\n"
        f"## Tax rates\n{tax_rates_json}\n\n"
        f"## Draft bill (current state in Xero)\n{json.dumps(invoice, indent=2)}\n\n"
        f"## Supplier document / attachment\n{attachment_text}"
    )

    # Images are only passed if the selected model supports vision.
    use_images = bool(attachment_images) and _supports_vision(model)
    first_msg: dict = {"role": "user", "content": user_text}
    if use_images:
        first_msg["images"] = attachment_images
    elif attachment_images:
        _log(f"  {len(attachment_images)} image(s) skipped — {model} does not support vision")

    messages: list[dict] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        first_msg,
    ]

    # ── Phase 1: tool loop ────────────────────────────────────────────────────
    for _attempt in range(8):
        response = client.chat(
            model=model,
            messages=messages,
            tools=_OLLAMA_TOOLS,
        )
        tool_calls = response.message.tool_calls or []
        _log(
            f"  Ollama [{model}]: tool_calls={len(tool_calls)} "
            f"content_len={len(response.message.content or '')}"
        )

        if not tool_calls:
            break

        # Append assistant message then one tool-result message per call.
        messages.append(response.message)
        for tc in tool_calls:
            result = _dispatch_tool(tenant_id, tc)
            messages.append({"role": "tool", "content": result})

    # ── Phase 2: structured decision ─────────────────────────────────────────
    messages.append({
        "role": "user",
        "content": "Now return your final structured JSON decision for this bill.",
    })

    final = client.chat(
        model=model,
        messages=messages,
        format=_DECISION_SCHEMA,
    )

    content = final.message.content
    if not content:
        raise RuntimeError(f"Ollama [{model}] returned an empty final response")
    return BillDecision(**json.loads(content))


# ── Combined model + org selection GUI ───────────────────────────────────────

def _select_model_and_tenants_gui(
    tenants: list[dict],
) -> tuple[str | None, list[dict], "ollama.Client | None"]:
    """
    Show a single GUI window for choosing:
      - Ollama host URL (local or remote)
      - Which Ollama model to use for reasoning
      - Which Xero organisations to process

    Returns (model_name, selected_tenants, ollama_client) or (None, [], None) if cancelled.
    """
    # ── Pre-fetch: draft bill counts ──────────────────────────────────────────
    print("Checking draft bill counts...")
    bill_counts: dict[str, int] = {}
    for t in tenants:
        try:
            bill_counts[t["tenantId"]] = len(
                json.loads(xero_list_draft_accpay_invoices(t["tenantId"]))
            )
        except Exception:
            bill_counts[t["tenantId"]] = -1

    # ── Build GUI ─────────────────────────────────────────────────────────────
    result_model: list[str] = []
    result_tenants: list[dict] = []
    result_client: list[ollama.Client] = []
    connected_client: list[ollama.Client] = []  # client from the last successful Connect

    ctk.set_appearance_mode("system")
    ctk.set_default_color_theme("blue")

    root = ctk.CTk()
    root.title("Xero AP Processing — Ollama")
    root.resizable(False, False)

    win_w = 580
    model_list_h = 220
    org_list_h = min(len(tenants) * 48, 200)
    win_h = max(500, min(100 + 70 + 60 + model_list_h + 50 + org_list_h + 120, 860))

    root.update_idletasks()
    sx = (root.winfo_screenwidth() - win_w) // 2
    sy = (root.winfo_screenheight() - win_h) // 2
    root.geometry(f"{win_w}x{win_h}+{sx}+{sy}")

    # ── Header ────────────────────────────────────────────────────────────────
    ctk.CTkLabel(
        root,
        text="Xero AP Processing — Ollama",
        font=ctk.CTkFont(size=20, weight="bold"),
    ).pack(pady=(24, 4))

    ctk.CTkLabel(
        root,
        text="Select a host, model, and the organisations to process:",
        font=ctk.CTkFont(size=13),
        text_color=("gray40", "gray60"),
    ).pack(pady=(0, 14))

    # ── Host section ──────────────────────────────────────────────────────────
    ctk.CTkLabel(
        root,
        text="Ollama Host",
        font=ctk.CTkFont(size=14, weight="bold"),
        anchor="w",
    ).pack(fill="x", padx=28, pady=(0, 4))

    host_row = ctk.CTkFrame(root, fg_color="transparent")
    host_row.pack(fill="x", padx=28, pady=(0, 10))

    host_entry = ctk.CTkEntry(host_row, font=ctk.CTkFont(size=13))
    host_entry.insert(0, _DEFAULT_HOST)
    host_entry.pack(side="left", fill="x", expand=True, padx=(0, 8))

    # connect_btn wired up after callbacks are defined
    connect_btn = ctk.CTkButton(host_row, text="Connect", width=100, height=32,
                                font=ctk.CTkFont(size=13))
    connect_btn.pack(side="right")

    # ── Model section ─────────────────────────────────────────────────────────
    ctk.CTkLabel(
        root,
        text="Model",
        font=ctk.CTkFont(size=14, weight="bold"),
        anchor="w",
    ).pack(fill="x", padx=28, pady=(0, 4))

    model_scroll = ctk.CTkScrollableFrame(root, width=520, height=model_list_h)
    model_scroll.pack(padx=28, pady=(0, 12), fill="x")

    model_var = ctk.StringVar(value="")
    ctk.CTkLabel(
        model_scroll,
        text="Enter a host URL above and click Connect to load models.",
        font=ctk.CTkFont(size=12),
        text_color=("gray50", "gray50"),
    ).pack(pady=8)

    # ── Org section ───────────────────────────────────────────────────────────
    # TODO: The bill-count pre-fetch, tenant checkbox rows, badge formatting, and
    #       Select All / Deselect All buttons here are substantially duplicated in
    #       xero_ap_agent.py — extract to a shared _build_org_section() helper.
    ctk.CTkLabel(
        root,
        text="Organisations",
        font=ctk.CTkFont(size=14, weight="bold"),
        anchor="w",
    ).pack(fill="x", padx=28, pady=(0, 4))

    org_scroll = ctk.CTkScrollableFrame(root, width=520, height=org_list_h)
    org_scroll.pack(padx=28, pady=(0, 8), fill="x")

    check_vars: list[tuple[dict, ctk.BooleanVar]] = []
    single_tenant = len(tenants) == 1

    for tenant in tenants:
        count = bill_counts.get(tenant["tenantId"], -1)
        if count == -1:
            badge = "unknown draft bills"
        elif count == 0:
            badge = "no draft bills"
        elif count == 1:
            badge = "1 draft bill"
        else:
            badge = f"{count} draft bills"

        row = ctk.CTkFrame(org_scroll, fg_color="transparent")
        row.pack(fill="x", padx=8, pady=4)

        # Pre-select and lock the checkbox when there is only one tenant.
        var = ctk.BooleanVar(value=single_tenant)
        cb = ctk.CTkCheckBox(
            row,
            text=tenant["name"],
            variable=var,
            font=ctk.CTkFont(size=13, weight="bold"),
            command=lambda: _refresh_run_btn(),
            width=300,
            state="disabled" if single_tenant else "normal",
        )
        cb.pack(side="left")

        label_color = ("gray50", "gray50") if count == 0 else ("gray30", "gray70")
        ctk.CTkLabel(
            row,
            text=badge,
            font=ctk.CTkFont(size=12),
            text_color=label_color,
        ).pack(side="right", padx=8)

        check_vars.append((tenant, var))

    # Select All / Deselect All (only shown when multiple tenants).
    if not single_tenant:
        link_row = ctk.CTkFrame(root, fg_color="transparent")
        link_row.pack(pady=(2, 0))

        def _select_all():
            for _, v in check_vars:
                v.set(True)
            _refresh_run_btn()

        def _deselect_all():
            for _, v in check_vars:
                v.set(False)
            _refresh_run_btn()

        ctk.CTkButton(
            link_row, text="Select All", width=110, height=28,
            font=ctk.CTkFont(size=12), command=_select_all,
        ).pack(side="left", padx=6)

        ctk.CTkButton(
            link_row, text="Deselect All", width=110, height=28,
            font=ctk.CTkFont(size=12),
            fg_color=("gray70", "gray30"), hover_color=("gray60", "gray40"),
            command=_deselect_all,
        ).pack(side="left", padx=6)

    # ── Run / Cancel ──────────────────────────────────────────────────────────
    action_row = ctk.CTkFrame(root, fg_color="transparent")
    action_row.pack(pady=18)

    # Disabled until both a model and at least one org are selected.
    run_btn = ctk.CTkButton(
        action_row,
        text="Run AP Processing",
        width=180,
        height=38,
        font=ctk.CTkFont(size=14, weight="bold"),
        state="disabled",
    )
    run_btn.pack(side="left", padx=10)

    ctk.CTkButton(
        action_row,
        text="Cancel",
        width=100,
        height=38,
        font=ctk.CTkFont(size=14),
        fg_color=("gray70", "gray30"),
        hover_color=("gray60", "gray40"),
        command=root.destroy,
    ).pack(side="left", padx=10)

    # ── Callbacks (defined after all widgets so closures resolve correctly) ───

    def _refresh_run_btn() -> None:
        model_ok = bool(model_var.get())
        org_ok = any(v.get() for _, v in check_vars)
        run_btn.configure(state="normal" if (model_ok and org_ok) else "disabled")

    def _load_models(host: str) -> None:
        for widget in model_scroll.winfo_children():
            widget.destroy()
        model_var.set("")
        connect_btn.configure(state="disabled")

        loading = ctk.CTkLabel(
            model_scroll, text=f"Connecting to {host} …",
            font=ctk.CTkFont(size=12), text_color=("gray50", "gray50"),
        )
        loading.pack(pady=8)
        model_scroll.update()

        try:
            client = ollama.Client(host=host)
            available = sorted(client.list().models, key=lambda m: m.size, reverse=True)
        except Exception as exc:
            for widget in model_scroll.winfo_children():
                widget.destroy()
            ctk.CTkLabel(
                model_scroll, text=f"Connection failed: {exc}",
                font=ctk.CTkFont(size=12), text_color="red",
            ).pack(pady=8)
            connect_btn.configure(state="normal")
            _refresh_run_btn()
            return

        connected_client.clear()
        connected_client.append(client)
        loading.destroy()

        if not available:
            ctk.CTkLabel(
                model_scroll,
                text="No models found. Run 'ollama pull <model>' to install one.",
                font=ctk.CTkFont(size=12), text_color=("gray50", "gray50"),
            ).pack(pady=8)
            connect_btn.configure(state="normal")
            _refresh_run_btn()
            return

        for m in available:
            size_gb = m.size / 1_000_000_000
            param_size = ""
            if m.details:
                param_size = getattr(m.details, "parameter_size", "") or ""
            vision_badge = "  [vision]" if _supports_vision(m.model) else ""
            label = f"{m.model}{vision_badge}   {param_size}   {size_gb:.1f} GB".strip()
            ctk.CTkRadioButton(
                model_scroll,
                text=label,
                variable=model_var,
                value=m.model,
                font=ctk.CTkFont(size=13),
                command=_refresh_run_btn,
            ).pack(anchor="w", padx=12, pady=5)

        connect_btn.configure(state="normal")
        _refresh_run_btn()

    def _on_connect() -> None:
        _load_models(host_entry.get().strip())

    def _on_run() -> None:
        result_model.append(model_var.get())
        result_tenants.extend(t for t, v in check_vars if v.get())
        result_client.append(connected_client[0])
        root.destroy()

    connect_btn.configure(command=_on_connect)
    run_btn.configure(command=_on_run)

    # Auto-connect on startup with the default host.
    root.after(50, _on_connect)

    root.mainloop()

    chosen_model = result_model[0] if result_model else None
    chosen_client = result_client[0] if result_client else None
    return chosen_model, result_tenants, chosen_client


# ── Main orchestration ────────────────────────────────────────────────────────

def run_ap_processing() -> None:
    """
    Process selected draft AP bills using an Ollama model (local or remote).
    All Xero API calls are made directly by Python; Ollama handles reasoning.
    """
    # ── Step 0: ensure authenticated ─────────────────────────────────────────
    if _get_token() is None:
        print("No Xero token found — starting authentication (browser will open)...")
        xero_authenticate()
        print("Authentication complete.\n")

    # ── Step 1: list tenants ──────────────────────────────────────────────────
    tenants: list[dict] = json.loads(xero_list_tenants())
    _log(f"Found {len(tenants)} tenant(s): {[t['name'] for t in tenants]}")

    # ── Step 2: select model + orgs via GUI ───────────────────────────────────
    selected_model, selected_tenants, ollama_client = _select_model_and_tenants_gui(tenants)
    if not selected_model or not selected_tenants or ollama_client is None:
        print("No model or organisations selected — exiting.")
        return

    print(f"\nModel      : {selected_model}")
    print(f"Organisations: {', '.join(t['name'] for t in selected_tenants)}\n")

    _contacts_created_this_run.clear()
    processed_ids: set[str] = set()
    total_updated = 0
    total_queries = 0

    for tenant in selected_tenants:
        tenant_id: str = tenant["tenantId"]
        tenant_name: str = tenant["name"]

        print(f"\n{'='*60}")
        print(f"Organisation: {tenant_name}")
        print(f"{'='*60}")

        # ── Step 3: fetch accounts and tax rates once per tenant ──────────────
        _log("Fetching chart of accounts...")
        accounts_json: str = xero_get_accounts(tenant_id)

        _log("Fetching tax rates...")
        tax_rates_json: str = xero_get_tax_rates(tenant_id)

        # ── Step 4: list draft bills ──────────────────────────────────────────
        bills: list[dict] = json.loads(xero_list_draft_accpay_invoices(tenant_id))
        _log(f"Found {len(bills)} draft bill(s)")

        if not bills:
            print("  No draft bills to process.")
            continue

        for i, summary in enumerate(bills, 1):
            invoice_id: str = summary["InvoiceID"]
            invoice_number: str = summary.get("InvoiceNumber") or "(no number)"
            contact: str = summary.get("Contact") or "(unknown)"
            amount = summary.get("AmountDue", "?")

            if invoice_id in processed_ids:
                _log(f"  Skipping {invoice_id} — already processed this run")
                continue

            print(f"\n  [{i}/{len(bills)}] {invoice_number} — {contact} — ${amount}")

            try:
                # ── Step 5: fetch full invoice ────────────────────────────────
                invoice: dict = json.loads(xero_get_invoice(tenant_id, invoice_id))

                # ── Step 6: fetch attachments ─────────────────────────────────
                _log("  Fetching attachments...")
                attachment_text, attachment_images = _fetch_attachment_content(
                    tenant_id, invoice_id
                )
                _log(
                    f"  Attachment: {len(attachment_text)} chars, "
                    f"{len(attachment_images)} image(s)"
                )

                # ── Step 7: ask Ollama to reason ──────────────────────────────
                _log(f"  Asking Ollama [{selected_model}]...")
                decision = _reason_about_bill(
                    selected_model,
                    ollama_client,
                    tenant_id,
                    invoice,
                    attachment_text,
                    attachment_images,
                    accounts_json,
                    tax_rates_json,
                )

                if decision.action == "query":
                    print(f"  QUERY: {decision.query}")
                    processed_ids.add(invoice_id)
                    total_queries += 1
                    continue

                # ── Step 8: validate decision fields ──────────────────────────
                if not decision.contact_name:
                    print("  QUERY: Model returned update with no contact name — skipping")
                    processed_ids.add(invoice_id)
                    total_queries += 1
                    continue

                if not decision.line_items:
                    print("  QUERY: Model returned update with no line items — skipping")
                    processed_ids.add(invoice_id)
                    total_queries += 1
                    continue

                # ── Step 9: save the update directly via Python ───────────────
                line_dicts = [li.model_dump() for li in decision.line_items]
                _log(
                    f"  Updating: contact={decision.contact_name!r} "
                    f"ref={decision.reference!r} "
                    f"lines={len(line_dicts)}"
                )
                result: dict = json.loads(
                    xero_update_invoice(
                        tenant_id=tenant_id,
                        invoice_id=invoice_id,
                        contact_name=decision.contact_name,
                        date=decision.date,
                        due_date=decision.due_date,
                        reference=decision.reference,
                        line_items=line_dicts,
                    )
                )
                print(
                    f"  Updated: {result.get('InvoiceNumber')} | "
                    f"Total=${result.get('Total')} | "
                    f"Tax=${result.get('TotalTax')} | "
                    f"Lines={result.get('LineItemCount')}"
                )
                processed_ids.add(invoice_id)
                total_updated += 1

            except Exception as exc:
                print(f"  ERROR: {exc} — skipping bill")
                _log(f"  Bill {invoice_id} failed: {exc}")

    print(f"\n{'='*60}")
    print(f"Done — updated: {total_updated}  |  queries raised: {total_queries}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run_ap_processing()
