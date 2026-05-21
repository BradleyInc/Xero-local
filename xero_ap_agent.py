"""
Xero AP/GST Processing Agent — Hybrid Mode

Python handles all orchestration and Xero API calls directly (zero Claude tokens):
  - List tenants
  - List draft bills per tenant
  - Fetch chart of accounts and tax rates (once per tenant)
  - Save updates via xero_update_invoice

Claude is called ONCE per bill for reasoning only:
  - Read and interpret the attachment (PDF text or image)
  - Verify the supplier contact exists in Xero (via search_contacts tool)
  - Extract invoice date, due date, reference, line items
  - Apply correct account codes and GST treatment
  - Raise a query if the bill cannot be determined

Usage:
    python xero_ap_agent.py

Requirements:
    pip install anthropic pypdf requests customtkinter
    ANTHROPIC_API_KEY environment variable must be set.
"""

import json

import anthropic
import customtkinter as ctk
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

client = anthropic.Anthropic()

# Contacts created during the current run — prevents duplicate creation when the
# same new supplier appears on multiple bills before Xero's search index catches up.
_contacts_created_this_run: dict[tuple[str, str], str] = {}


# ── Structured output schema ──────────────────────────────────────────────────
# TODO: LineItem, BillDecision, and _DECISION_SCHEMA are duplicated in
#       xero_ap_agent_ollama.py — extract to a shared xero_ap_shared.py module.

class LineItem(BaseModel):
    Description: str
    AccountCode: str
    UnitAmount: float
    TaxType: Literal["INPUT2", "EXEMPTEXPENSES", "BASEXCLUDED"]
    Quantity: float = 1.0


class BillDecision(BaseModel):
    action: Literal["update", "query"]
    # Populated when action == "update"
    contact_name: str | None = None
    date: str | None = None
    due_date: str | None = None
    reference: str | None = None
    line_items: list[LineItem] | None = None
    # Populated when action == "query"
    query: str | None = None


_DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["update", "query"],
        },
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
                    "Description",
                    "AccountCode",
                    "UnitAmount",
                    "TaxType",
                    "Quantity",
                ],
                "additionalProperties": False,
            },
        },
        "query": {"type": "string"},
    },
    "required": ["action"],
    "additionalProperties": False,
}


# ── System prompt (cached — identical for every bill) ────────────────────────

_SYSTEM_PROMPT = """You are an expert Australian accounts payable bookkeeper.

You will receive:
- A Xero draft bill (JSON)
- The bill's attachments (supplier invoice documents as text or images)
- The tenant's chart of accounts
- The tenant's active tax rates

## Your task

Return a structured JSON decision to either update the bill or raise a query.

### action = "update" (when you can determine all details)

Fields to populate:
- **contact_name**: Call search_contacts to find the exact supplier name in Xero. Use what Xero returns.
- **date**: Invoice date from the document (YYYY-MM-DD). Do not use today's date.
- **due_date**: Due date from the document. If not stated, add 30 days to the invoice date (YYYY-MM-DD).
- **reference**: The supplier's invoice or order number from the document.
- **line_items**: One entry per distinct expense line. For each:
  - **Description**: Brief, clear description of what was purchased.
  - **AccountCode**: Choose the best-fit code from the provided chart of accounts.
  - **UnitAmount**: The line amount. Match the bill's `LineAmountTypes`:
      * `INCLUSIVE` → amount includes GST (e.g. $110.00 for a $100 item + $10 GST)
      * `EXCLUSIVE` → amount excludes GST (e.g. $100.00 net)
  - **TaxType**: One of:
      * `INPUT2` — GST on Expenses: standard taxable business expenses (most common)
      * `EXEMPTEXPENSES` — GST Free Expenses: medical, educational, fresh food, exported services
      * `BASEXCLUDED` — BAS Excluded: wages, superannuation, ATO payments, bank fees, stamps
  - **Quantity**: 1 unless the document states a different quantity.

### action = "query" (when you cannot determine the bill)

Populate **query** with a clear description of what is missing or ambiguous.

## Rules
- Always call search_contacts before returning a contact_name — never guess.
- If search_contacts returns no results, call create_contact to create the supplier before returning.
- Use only account codes from the provided chart of accounts.
- Round all amounts to 2 decimal places.
- Do not split a single charge across multiple lines unless the document itemises it separately.
"""


# ── Tools Claude is allowed to use ───────────────────────────────────────────

_SEARCH_CONTACTS_TOOL = {
    "name": "search_contacts",
    "description": "Search for a supplier contact in Xero by name or partial name.",
    "input_schema": {
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
}

_CREATE_CONTACT_TOOL = {
    "name": "create_contact",
    "description": (
        "Create a new supplier contact in Xero. "
        "Only call this after search_contacts returns no results."
    ),
    "input_schema": {
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
}

_CLAUDE_TOOLS = [_SEARCH_CONTACTS_TOOL, _CREATE_CONTACT_TOOL]


# ── Claude reasoning call (one call per bill) ─────────────────────────────────

def _reason_about_bill(
    tenant_id: str,
    invoice: dict,
    attachment_blocks: list[dict],
    accounts_json: str,
    tax_rates_json: str,
) -> BillDecision:
    """
    Ask Claude to reason about a single bill.

    Passes accounts and tax rates as prompt-cached blocks (reused across bills
    for the same tenant). Claude may call search_contacts once or twice, then
    returns a structured JSON decision.
    """
    # Accounts and tax rates are stable for the tenant — mark them for caching.
    user_content: list[dict] = [
        {
            "type": "text",
            "text": f"## Chart of accounts\n{accounts_json}",
            "cache_control": {"type": "ephemeral"},
        },
        {
            "type": "text",
            "text": f"## Tax rates\n{tax_rates_json}",
            "cache_control": {"type": "ephemeral"},
        },
        {
            "type": "text",
            "text": f"## Draft bill (current state in Xero)\n{json.dumps(invoice, indent=2)}",
        },
    ]

    if attachment_blocks:
        user_content.append({"type": "text", "text": "## Supplier document / attachment"})
        user_content.extend(attachment_blocks)
    else:
        user_content.append({
            "type": "text",
            "text": "## Supplier document / attachment\nNo attachments found on this bill.",
        })

    user_content.append({
        "type": "text",
        "text": "Process this bill and return your structured decision.",
    })

    messages: list[dict] = [{"role": "user", "content": user_content}]

    # Bounded loop: Claude may call tools a few times before returning a decision.
    for _attempt in range(10):
        response = client.messages.create(
            model="claude-opus-4-7",
            max_tokens=8192,
            thinking={"type": "adaptive"},
            system=[
                {
                    "type": "text",
                    "text": _SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            tools=_CLAUDE_TOOLS,
            output_config={
                "format": {
                    "type": "json_schema",
                    "schema": _DECISION_SCHEMA,
                }
            },
            messages=messages,
        )

        _log(
            f"  Claude: stop={response.stop_reason} "
            f"in={response.usage.input_tokens} "
            f"out={response.usage.output_tokens} "
            f"cached={response.usage.cache_read_input_tokens}"
        )

        if response.stop_reason == "end_turn":
            text = next((b.text for b in response.content if b.type == "text"), None)
            if text is None:
                raise RuntimeError("Claude returned end_turn with no text block in response")
            return BillDecision(**json.loads(text))

        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
        if response.stop_reason != "tool_use" or not tool_use_blocks:
            raise RuntimeError(
                f"Unexpected stop_reason '{response.stop_reason}' with no tool use blocks"
            )

        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in tool_use_blocks:
            if block.name == "search_contacts":
                search_term = block.input["search_term"]
                _log(f"  search_contacts({search_term!r})")
                result = xero_get_contacts(tenant_id, search_term)
            elif block.name == "create_contact":
                name = block.input["name"]
                email = block.input.get("email")
                cache_key = (tenant_id, name.lower().strip())
                if cache_key in _contacts_created_this_run:
                    _log(f"  create_contact({name!r}): using cached result from this run")
                    result = _contacts_created_this_run[cache_key]
                else:
                    _log(f"  create_contact({name!r}, email={email!r})")
                    result = xero_create_contact(tenant_id, name, email)
                    _contacts_created_this_run[cache_key] = result
                    # Verify the contact round-trips in Xero search before proceeding.
                    verify_raw = xero_get_contacts(tenant_id, name)
                    try:
                        if json.loads(verify_raw):
                            _log("  create_contact: round-trip search verified OK")
                        else:
                            _log(
                                f"  create_contact: round-trip search returned empty "
                                f"(Xero indexing lag) — proceeding with creation result"
                            )
                    except Exception:
                        pass
            else:
                result = json.dumps({"error": f"Unknown tool: {block.name}"})
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result,
            })

        messages.append({"role": "user", "content": tool_results})

    raise RuntimeError("Claude did not reach end_turn within 10 iterations")


# ── Attachment size limits ────────────────────────────────────────────────────

_MAX_TEXT_CHARS = 15_000      # truncate individual text blocks beyond this
_MAX_IMAGE_B64_LEN = 1_500_000  # skip images whose base64 exceeds ~1.1 MB binary
_MAX_TOTAL_CHARS = 80_000     # hard stop across all blocks combined


# ── Attachment fetcher ────────────────────────────────────────────────────────

def _fetch_attachment_blocks(tenant_id: str, invoice_id: str) -> list[dict]:
    """Fetch attachments, apply size limits, and return Claude API content blocks."""
    mcp_results = xero_get_invoice_attachments(tenant_id, invoice_id)
    blocks: list[dict] = []
    total_chars = 0

    for item in mcp_results:
        if total_chars >= _MAX_TOTAL_CHARS:
            _log("  Attachment total limit reached — skipping remaining attachments")
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
            blocks.append({"type": "text", "text": text})
            total_chars += len(text)

        elif item.type == "image":
            if len(item.data) > _MAX_IMAGE_B64_LEN:
                print(f"  WARNING: image attachment skipped (too large) — review manually")
                blocks.append({
                    "type": "text",
                    "text": "[Image attachment skipped — file too large to include]",
                })
                total_chars += 60
            else:
                blocks.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": item.mimeType,
                        "data": item.data,
                    },
                })
                total_chars += len(item.data)

    return blocks


# ── Organisation selection GUI ────────────────────────────────────────────────
# TODO: The bill-count pre-fetch, tenant checkbox rows, badge formatting, and
#       Select All / Deselect All buttons here are substantially duplicated in
#       xero_ap_agent_ollama.py — extract to a shared _build_org_section() helper.

def _select_tenants_gui(tenants: list[dict]) -> list[dict]:
    """
    Show a GUI for selecting which Xero organisations to process.

    Fetches draft bill counts first so each org shows how many bills are waiting.
    Returns the selected tenant dicts, or an empty list if the user cancels.
    If only one tenant is connected the GUI is skipped and it is returned directly.
    """
    if len(tenants) == 1:
        print(f"One organisation connected: {tenants[0]['name']} — processing automatically.")
        return tenants

    # Fetch draft bill counts up front so the GUI is informative.
    print("Checking draft bill counts...")
    bill_counts: dict[str, int] = {}
    for t in tenants:
        try:
            bills = json.loads(xero_list_draft_accpay_invoices(t["tenantId"]))
            bill_counts[t["tenantId"]] = len(bills)
        except Exception:
            bill_counts[t["tenantId"]] = -1  # unknown

    selected_tenants: list[dict] = []

    ctk.set_appearance_mode("system")
    ctk.set_default_color_theme("blue")

    root = ctk.CTk()
    root.title("Xero AP Processing")
    root.resizable(False, False)

    # Size and centre on screen.
    win_w = 500
    win_h = 180 + len(tenants) * 52 + 80
    win_h = max(340, min(win_h, 680))
    root.update_idletasks()
    sx = (root.winfo_screenwidth() - win_w) // 2
    sy = (root.winfo_screenheight() - win_h) // 2
    root.geometry(f"{win_w}x{win_h}+{sx}+{sy}")

    # ── Header ────────────────────────────────────────────────────────────────
    ctk.CTkLabel(
        root,
        text="Xero AP Processing",
        font=ctk.CTkFont(size=20, weight="bold"),
    ).pack(pady=(28, 4))

    ctk.CTkLabel(
        root,
        text="Select the organisations to process draft payable bills for:",
        font=ctk.CTkFont(size=13),
        text_color=("gray40", "gray60"),
    ).pack(pady=(0, 18))

    # ── Scrollable checkbox list ──────────────────────────────────────────────
    list_height = min(len(tenants) * 52, 340)
    scroll_frame = ctk.CTkScrollableFrame(root, width=440, height=list_height)
    scroll_frame.pack(padx=28, pady=(0, 10), fill="x")

    check_vars: list[tuple[dict, ctk.BooleanVar]] = []

    def _refresh_run_btn():
        any_selected = any(v.get() for _, v in check_vars)
        run_btn.configure(state="normal" if any_selected else "disabled")

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

        row = ctk.CTkFrame(scroll_frame, fg_color="transparent")
        row.pack(fill="x", padx=8, pady=4)

        var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            row,
            text=tenant["name"],
            variable=var,
            font=ctk.CTkFont(size=14, weight="bold"),
            command=_refresh_run_btn,
            width=260,
        ).pack(side="left")

        label_color = ("gray50", "gray50") if count == 0 else ("gray30", "gray70")
        ctk.CTkLabel(
            row,
            text=badge,
            font=ctk.CTkFont(size=12),
            text_color=label_color,
        ).pack(side="right", padx=8)

        check_vars.append((tenant, var))

    # ── Select All / Deselect All ─────────────────────────────────────────────
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
        font=ctk.CTkFont(size=12), fg_color=("gray70", "gray30"),
        hover_color=("gray60", "gray40"), command=_deselect_all,
    ).pack(side="left", padx=6)

    # ── Run / Cancel ──────────────────────────────────────────────────────────
    action_row = ctk.CTkFrame(root, fg_color="transparent")
    action_row.pack(pady=18)

    def _on_run():
        nonlocal selected_tenants
        selected_tenants = [t for t, v in check_vars if v.get()]
        root.destroy()

    run_btn = ctk.CTkButton(
        action_row,
        text="Run AP Processing",
        width=180,
        height=38,
        font=ctk.CTkFont(size=14, weight="bold"),
        state="disabled",
        command=_on_run,
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

    root.mainloop()
    return selected_tenants


# ── Main orchestration — pure Python, no Claude tokens ───────────────────────

def run_ap_processing() -> None:
    """
    Process all draft AP bills across all connected Xero organisations.

    All Xero API calls (list, fetch, update) are made directly by Python.
    Claude is called once per bill to reason about it.
    """
    # ── Step 0: ensure authenticated ─────────────────────────────────────────
    if _get_token() is None:
        print("No Xero token found — starting authentication (browser will open)...")
        xero_authenticate()
        print("Authentication complete.\n")

    # ── Step 1: list tenants ──────────────────────────────────────────────────
    tenants: list[dict] = json.loads(xero_list_tenants())
    _log(f"Found {len(tenants)} tenant(s): {[t['name'] for t in tenants]}")

    selected_tenants = _select_tenants_gui(tenants)
    if not selected_tenants:
        print("No organisations selected — exiting.")
        return

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

        # ── Step 2: fetch accounts and tax rates once per tenant ──────────────
        _log("Fetching chart of accounts...")
        accounts_json: str = xero_get_accounts(tenant_id)

        _log("Fetching tax rates...")
        tax_rates_json: str = xero_get_tax_rates(tenant_id)

        # ── Step 3: list draft bills ──────────────────────────────────────────
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
                # ── Step 4: fetch full invoice ────────────────────────────────
                invoice: dict = json.loads(xero_get_invoice(tenant_id, invoice_id))

                # ── Step 5: fetch attachments ─────────────────────────────────
                _log("  Fetching attachments...")
                attachment_blocks = _fetch_attachment_blocks(tenant_id, invoice_id)
                _log(f"  {len(attachment_blocks)} content block(s) from attachments")

                # ── Step 6: ask Claude to reason ──────────────────────────────
                _log("  Asking Claude...")
                decision = _reason_about_bill(
                    tenant_id,
                    invoice,
                    attachment_blocks,
                    accounts_json,
                    tax_rates_json,
                )

                if decision.action == "query":
                    print(f"  QUERY: {decision.query}")
                    processed_ids.add(invoice_id)
                    total_queries += 1
                    continue

                # ── Step 7: save the update directly via Python ───────────────
                if not decision.contact_name:
                    print(f"  QUERY: Claude returned an update decision with no contact name — skipping")
                    processed_ids.add(invoice_id)
                    total_queries += 1
                    continue

                if not decision.line_items:
                    print(f"  QUERY: Claude returned an update decision with no line items — skipping")
                    processed_ids.add(invoice_id)
                    total_queries += 1
                    continue

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
