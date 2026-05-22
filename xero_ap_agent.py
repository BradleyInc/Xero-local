"""
Xero AP/GST Processing Agent — Hybrid Mode (self-contained)

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
    XERO_CLIENT_ID and XERO_CLIENT_SECRET environment variables (or .env file) must be set.
"""

import base64
import io
import json
import os
import re
import sys
import threading
import time
import traceback
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Literal
from urllib.parse import parse_qs, urlparse

import anthropic
import customtkinter as ctk
import pypdf
import requests
from pydantic import BaseModel

# ── .env loader ───────────────────────────────────────────────────────────────

_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(_env_path):
    with open(_env_path, "r") as _f:
        for _line in _f:
            _match = re.match(r"^([^#=][^=]*)=(.*)$", _line.strip())
            if _match:
                os.environ.setdefault(
                    _match.group(1).strip(),
                    _match.group(2).strip().strip("\"'"),
                )

# ── Xero constants ────────────────────────────────────────────────────────────

CLIENT_ID = os.environ["XERO_CLIENT_ID"]
CLIENT_SECRET = os.environ["XERO_CLIENT_SECRET"]
REDIRECT_URI = "http://localhost:8080/callback"
SCOPES = "offline_access accounting.transactions accounting.contacts accounting.settings accounting.reports.read"
TOKEN_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "xero_token.json")
XERO_API_BASE = "https://api.xero.com/api.xro/2.0"
REQUEST_TIMEOUT = 30

_cached_token: dict | None = None

# ── Logging ───────────────────────────────────────────────────────────────────

def _log(msg: str) -> None:
    print(f"[xero-ap {datetime.now().strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)

# ── Token helpers ─────────────────────────────────────────────────────────────

def _b64(s: str) -> str:
    return base64.b64encode(s.encode()).decode()


def _basic_auth_header() -> str:
    return f"Basic {_b64(f'{CLIENT_ID}:{CLIENT_SECRET}')}"


def _load_token() -> dict | None:
    try:
        with open(TOKEN_FILE) as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def _save_token(data: dict) -> None:
    global _cached_token
    _cached_token = {
        **data,
        "expires_at": (
            datetime.now() + timedelta(seconds=data.get("expires_in", 1800))
        ).timestamp(),
    }
    with open(TOKEN_FILE, "w") as f:
        json.dump(_cached_token, f)


def _raise_for_status(resp: requests.Response) -> None:
    if not resp.ok:
        raw = resp.text[:4000]
        detail = raw
        try:
            data = resp.json()
            validation_msgs = [
                ve.get("Message", "")
                for el in data.get("Elements", [])
                for ve in el.get("ValidationErrors", [])
                if ve.get("Message")
            ]
            top_msg = data.get("Message", "")
            if validation_msgs:
                detail = f"{top_msg} — ValidationErrors: {'; '.join(validation_msgs)}"
            elif top_msg:
                detail = top_msg
        except Exception:
            pass
        _log(f"HTTP {resp.status_code} from {resp.url}: {detail}")
        raise RuntimeError(f"HTTP {resp.status_code}: {detail}\nFull body: {raw}")


def _refresh(refresh_token: str) -> None:
    _log("Refreshing access token...")
    resp = requests.post(
        "https://identity.xero.com/connect/token",
        headers={
            "Authorization": _basic_auth_header(),
            "Content-Type": "application/x-www-form-urlencoded",
        },
        data={"grant_type": "refresh_token", "refresh_token": refresh_token},
        timeout=REQUEST_TIMEOUT,
    )
    _raise_for_status(resp)
    _save_token(resp.json())
    _log("Token refreshed successfully.")


def _get_token() -> dict | None:
    global _cached_token
    if _cached_token is None:
        _cached_token = _load_token()
    if not _cached_token:
        _log("No token file found — authentication required.")
        return None
    now_ts = datetime.now().timestamp()
    if now_ts > _cached_token.get("expires_at", 0) - 60:
        secs_ago = int(now_ts - _cached_token.get("expires_at", 0))
        _log(f"Token expired {secs_ago}s ago — refreshing.")
        _refresh(_cached_token["refresh_token"])
    return _cached_token


def _auth_headers(tenant_id: str | None = None) -> dict:
    token = _get_token()
    if not token:
        raise RuntimeError("Not authenticated. Call xero_authenticate first.")
    headers = {
        "Authorization": f"Bearer {token['access_token']}",
        "Accept": "application/json",
    }
    if tenant_id:
        headers["Xero-tenant-id"] = tenant_id
    return headers


# ── OAuth flow ────────────────────────────────────────────────────────────────

class _CallbackHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(b"<h1>Authentication complete</h1><p>You may close this tab.</p>")
        params = parse_qs(urlparse(self.path).query)
        self.server.auth_code = params.get("code", [None])[0]

    def log_message(self, *_):
        pass


def _run_oauth_flow() -> None:
    server = HTTPServer(("localhost", 8080), _CallbackHandler)
    server.auth_code = None
    threading.Thread(target=server.serve_forever, daemon=True).start()

    auth_url = (
        "https://login.xero.com/identity/connect/authorize"
        "?response_type=code"
        f"&client_id={CLIENT_ID}"
        f"&redirect_uri={REDIRECT_URI}"
        f"&scope={SCOPES.replace(' ', '%20')}"
    )
    webbrowser.open(auth_url)

    while server.auth_code is None:
        time.sleep(0.1)
    server.shutdown()

    resp = requests.post(
        "https://identity.xero.com/connect/token",
        headers={
            "Authorization": _basic_auth_header(),
            "Content-Type": "application/x-www-form-urlencoded",
        },
        data={
            "grant_type": "authorization_code",
            "code": server.auth_code,
            "redirect_uri": REDIRECT_URI,
        },
        timeout=REQUEST_TIMEOUT,
    )
    _raise_for_status(resp)
    _save_token(resp.json())


# ── Xero API functions ────────────────────────────────────────────────────────

def xero_authenticate() -> str:
    _log("xero_authenticate: starting OAuth flow")
    try:
        _run_oauth_flow()
        _log("xero_authenticate: OAuth flow completed successfully")
        return "Authentication successful. Token saved."
    except Exception as exc:
        _log(f"xero_authenticate ERROR: {exc}\n{traceback.format_exc()}")
        raise RuntimeError(f"Authentication failed: {exc}") from exc


def xero_list_tenants() -> str:
    _log("xero_list_tenants: fetching connected organisations")
    try:
        resp = requests.get(
            "https://api.xero.com/connections",
            headers=_auth_headers(),
            timeout=REQUEST_TIMEOUT,
        )
        _raise_for_status(resp)
        tenants = resp.json()
        _log(f"xero_list_tenants: found {len(tenants)} tenant(s): {[t.get('tenantName') for t in tenants]}")
        return json.dumps(
            [{"name": t["tenantName"], "tenantId": t["tenantId"], "type": t["tenantType"]}
             for t in tenants],
        )
    except Exception as exc:
        _log(f"xero_list_tenants ERROR: {exc}\n{traceback.format_exc()}")
        raise


def xero_list_draft_accpay_invoices(tenant_id: str) -> str:
    _log(f"xero_list_draft_accpay_invoices: tenant_id={tenant_id}")
    try:
        resp = requests.get(
            f"{XERO_API_BASE}/Invoices",
            headers=_auth_headers(tenant_id),
            params={"where": 'Type=="ACCPAY" AND Status=="DRAFT"'},
            timeout=REQUEST_TIMEOUT,
        )
        _raise_for_status(resp)
        invoices = resp.json().get("Invoices", [])
        _log(f"xero_list_draft_accpay_invoices: found {len(invoices)} draft bill(s)")
        summary = [
            {
                "InvoiceID": inv.get("InvoiceID"),
                "InvoiceNumber": inv.get("InvoiceNumber"),
                "Contact": inv.get("Contact", {}).get("Name"),
                "Date": inv.get("DateString"),
                "DueDate": inv.get("DueDateString"),
                "AmountDue": inv.get("AmountDue"),
                "CurrencyCode": inv.get("CurrencyCode"),
                "Reference": inv.get("Reference"),
                "HasAttachments": inv.get("HasAttachments", False),
            }
            for inv in invoices
        ]
        return json.dumps(summary)
    except Exception as exc:
        _log(f"xero_list_draft_accpay_invoices ERROR: {exc}\n{traceback.format_exc()}")
        raise


@dataclass
class _TextContent:
    type: str = "text"
    text: str = ""


@dataclass
class _ImageContent:
    type: str = "image"
    data: str = ""
    mimeType: str = ""


def xero_get_invoice_attachments(
    tenant_id: str, invoice_id: str
) -> list[_TextContent | _ImageContent]:
    _log(f"xero_get_invoice_attachments: tenant_id={tenant_id} invoice_id={invoice_id}")
    try:
        headers = _auth_headers(tenant_id)
        resp = requests.get(
            f"{XERO_API_BASE}/Invoices/{invoice_id}/Attachments",
            headers=headers,
            timeout=REQUEST_TIMEOUT,
        )
        _raise_for_status(resp)
        attachments = resp.json().get("Attachments", [])
        _log(f"xero_get_invoice_attachments: {len(attachments)} attachment(s) found for invoice {invoice_id}")

        result: list[_TextContent | _ImageContent] = []
        for a in attachments:
            filename = a.get("FileName", "")
            mime_type = a.get("MimeType", "")
            _log(f"xero_get_invoice_attachments: fetching '{filename}' ({mime_type}, {a.get('ContentLength')} bytes)")

            content_headers = dict(headers)
            content_headers["Accept"] = mime_type or "*/*"
            try:
                content_resp = requests.get(
                    f"{XERO_API_BASE}/Invoices/{invoice_id}/Attachments/{filename}",
                    headers=content_headers,
                    timeout=REQUEST_TIMEOUT,
                )
                _raise_for_status(content_resp)
            except Exception as exc:
                _log(f"xero_get_invoice_attachments: failed to fetch '{filename}': {exc}")
                result.append(_TextContent(text=json.dumps({"FileName": filename, "error": str(exc)})))
                continue

            metadata = {
                "AttachmentID": a.get("AttachmentID"),
                "FileName": filename,
                "MimeType": mime_type,
                "ContentLength": a.get("ContentLength"),
            }

            is_text = mime_type.startswith("text/") or mime_type in (
                "application/json", "application/xml"
            )
            is_image = mime_type.startswith("image/")

            if is_text:
                metadata["Content"] = content_resp.text
                result.append(_TextContent(text=json.dumps(metadata, indent=2)))
            elif mime_type == "application/pdf":
                try:
                    reader = pypdf.PdfReader(io.BytesIO(content_resp.content))
                    pages = [page.extract_text() or "" for page in reader.pages]
                    text = "\n".join(pages).strip()
                    if len(text) > 15_000:
                        text = text[:15_000] + "\n[truncated]"
                    metadata["Content"] = text
                    _log(f"xero_get_invoice_attachments: extracted {len(text)} chars from PDF '{filename}'")
                except Exception as pdf_exc:
                    _log(f"xero_get_invoice_attachments: PDF extraction failed for '{filename}': {pdf_exc}")
                    metadata["Content"] = f"[PDF extraction error: {pdf_exc}]"
                result.append(_TextContent(text=json.dumps(metadata, indent=2)))
            elif is_image:
                result.append(_TextContent(text=json.dumps(metadata, indent=2)))
                result.append(_ImageContent(
                    data=base64.b64encode(content_resp.content).decode(),
                    mimeType=mime_type,
                ))
            else:
                metadata["Content"] = base64.b64encode(content_resp.content).decode()
                metadata["ContentEncoding"] = "base64"
                result.append(_TextContent(text=json.dumps(metadata, indent=2)))

        return result
    except Exception as exc:
        _log(f"xero_get_invoice_attachments ERROR: {exc}\n{traceback.format_exc()}")
        raise


def xero_get_contacts(tenant_id: str, search: str = "") -> str:
    _log(f"xero_get_contacts: tenant_id={tenant_id} search={repr(search)}")
    try:
        params: dict = {"summaryOnly": "true", "where": 'ContactStatus=="ACTIVE"'}
        if search:
            params["searchTerm"] = search
        resp = requests.get(
            f"{XERO_API_BASE}/Contacts",
            headers=_auth_headers(tenant_id),
            params=params,
            timeout=REQUEST_TIMEOUT,
        )
        _raise_for_status(resp)
        contacts = resp.json().get("Contacts", [])
        _log(f"xero_get_contacts: returned {len(contacts)} contact(s)")
        return json.dumps(
            [
                {
                    "ContactID": c.get("ContactID"),
                    "Name": c.get("Name"),
                    "EmailAddress": c.get("EmailAddress"),
                }
                for c in contacts
            ],
        )
    except Exception as exc:
        _log(f"xero_get_contacts ERROR: {exc}\n{traceback.format_exc()}")
        raise


def xero_create_contact(tenant_id: str, name: str, email: str | None = None) -> str:
    _log(f"xero_create_contact: tenant_id={tenant_id} name={repr(name)} email={repr(email)}")
    try:
        contact_data: dict = {"Name": name}
        if email:
            contact_data["EmailAddress"] = email

        headers = _auth_headers(tenant_id)
        headers["Content-Type"] = "application/json"

        resp = requests.post(
            f"{XERO_API_BASE}/Contacts",
            headers=headers,
            json={"Contacts": [contact_data]},
            timeout=REQUEST_TIMEOUT,
        )
        _raise_for_status(resp)
        contact = resp.json().get("Contacts", [{}])[0]
        result = {
            "ContactID": contact.get("ContactID"),
            "Name": contact.get("Name"),
            "EmailAddress": contact.get("EmailAddress"),
            "ContactStatus": contact.get("ContactStatus"),
        }
        _log(f"xero_create_contact: created contact Name={result['Name']} ID={result['ContactID']}")
        return json.dumps(result)
    except Exception as exc:
        _log(f"xero_create_contact ERROR: {exc}\n{traceback.format_exc()}")
        raise


def xero_get_invoice(tenant_id: str, invoice_id: str) -> str:
    _log(f"xero_get_invoice: tenant_id={tenant_id} invoice_id={invoice_id}")
    try:
        resp = requests.get(
            f"{XERO_API_BASE}/Invoices/{invoice_id}",
            headers=_auth_headers(tenant_id),
            timeout=REQUEST_TIMEOUT,
        )
        _raise_for_status(resp)
        invoices = resp.json().get("Invoices", [])
        if not invoices:
            _log(f"xero_get_invoice: no invoice found for id={invoice_id}")
            return json.dumps({"error": "Invoice not found"})
        inv = invoices[0]
        _log(f"xero_get_invoice: retrieved invoice {inv.get('InvoiceNumber')} status={inv.get('Status')} total={inv.get('Total')} lines={len(inv.get('LineItems', []))}")
        return json.dumps(
            {
                "InvoiceID": inv.get("InvoiceID"),
                "InvoiceNumber": inv.get("InvoiceNumber"),
                "Type": inv.get("Type"),
                "Status": inv.get("Status"),
                "Contact": inv.get("Contact", {}).get("Name"),
                "Date": inv.get("DateString"),
                "DueDate": inv.get("DueDateString"),
                "Reference": inv.get("Reference"),
                "LineAmountTypes": inv.get("LineAmountTypes"),
                "LineItems": inv.get("LineItems", []),
                "SubTotal": inv.get("SubTotal"),
                "TotalTax": inv.get("TotalTax"),
                "Total": inv.get("Total"),
                "AmountDue": inv.get("AmountDue"),
                "CurrencyCode": inv.get("CurrencyCode"),
                "HasAttachments": inv.get("HasAttachments", False),
            },
        )
    except Exception as exc:
        _log(f"xero_get_invoice ERROR: {exc}\n{traceback.format_exc()}")
        raise


_AP_ACCOUNT_TYPES = {
    "EXPENSE", "OVERHEADS", "DIRECTCOSTS",
    "CURRENT", "CURRLIAB", "LIABILITY", "FIXED", "NONCURRENT",
}


def xero_get_accounts(tenant_id: str, include_all: bool = False) -> str:
    _log(f"xero_get_accounts: tenant_id={tenant_id} include_all={include_all}")
    try:
        resp = requests.get(
            f"{XERO_API_BASE}/Accounts",
            headers=_auth_headers(tenant_id),
            params={"where": 'Status=="ACTIVE"'},
            timeout=REQUEST_TIMEOUT,
        )
        _raise_for_status(resp)
        accounts = resp.json().get("Accounts", [])
        if not include_all:
            accounts = [a for a in accounts if a.get("Type") in _AP_ACCOUNT_TYPES]
        _log(f"xero_get_accounts: returning {len(accounts)} accounts (include_all={include_all})")
        return json.dumps(
            [
                {
                    "Code": a.get("Code"),
                    "Name": a.get("Name"),
                    "Type": a.get("Type"),
                    "TaxType": a.get("TaxType"),
                }
                for a in accounts
            ]
        )
    except Exception as exc:
        _log(f"xero_get_accounts ERROR: {exc}\n{traceback.format_exc()}")
        raise


def xero_get_tax_rates(tenant_id: str) -> str:
    _log(f"xero_get_tax_rates: tenant_id={tenant_id}")
    try:
        resp = requests.get(
            f"{XERO_API_BASE}/TaxRates",
            headers=_auth_headers(tenant_id),
            params={"where": 'Status=="ACTIVE"'},
            timeout=REQUEST_TIMEOUT,
        )
        _raise_for_status(resp)
        rates = resp.json().get("TaxRates", [])
        _log(f"xero_get_tax_rates: returned {len(rates)} tax rates: {[r.get('TaxType') for r in rates]}")
        return json.dumps(
            [
                {
                    "Name": r.get("Name"),
                    "TaxType": r.get("TaxType"),
                    "EffectiveRate": r.get("EffectiveRate"),
                    "Status": r.get("Status"),
                }
                for r in rates
            ]
        )
    except Exception as exc:
        _log(f"xero_get_tax_rates ERROR: {exc}\n{traceback.format_exc()}")
        raise


def xero_update_invoice(
    tenant_id: str,
    invoice_id: str,
    contact_name: str | None = None,
    date: str | None = None,
    due_date: str | None = None,
    reference: str | None = None,
    line_items: list[dict] | None = None,
) -> str:
    _log(f"xero_update_invoice: tenant_id={tenant_id} invoice_id={invoice_id}")
    try:
        payload: dict = {}

        if contact_name is not None:
            payload["Contact"] = {"Name": contact_name}
        if date is not None:
            payload["Date"] = date
        if due_date is not None:
            payload["DueDate"] = due_date
        if reference is not None:
            payload["Reference"] = reference
        if line_items is not None:
            payload["LineItems"] = line_items

        _log(f"xero_update_invoice: payload fields={list(payload.keys())} line_item_count={len(line_items) if line_items else 'n/a'}")
        _log(f"xero_update_invoice: full payload={json.dumps(payload)}")

        headers = _auth_headers(tenant_id)
        headers["Content-Type"] = "application/json"

        resp = requests.post(
            f"{XERO_API_BASE}/Invoices/{invoice_id}",
            headers=headers,
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )

        _log(f"xero_update_invoice: response status={resp.status_code}")
        if not resp.ok:
            _log(f"xero_update_invoice: error response body={resp.text[:3000]}")
        _raise_for_status(resp)

        updated = resp.json().get("Invoices", [{}])[0]
        result = {
            "InvoiceID": updated.get("InvoiceID"),
            "InvoiceNumber": updated.get("InvoiceNumber"),
            "Status": updated.get("Status"),
            "Contact": updated.get("Contact", {}).get("Name"),
            "Date": updated.get("DateString"),
            "DueDate": updated.get("DueDateString"),
            "Reference": updated.get("Reference"),
            "SubTotal": updated.get("SubTotal"),
            "TotalTax": updated.get("TotalTax"),
            "Total": updated.get("Total"),
            "AmountDue": updated.get("AmountDue"),
            "LineItemCount": len(updated.get("LineItems", [])),
        }
        _log(f"xero_update_invoice: SUCCESS invoice={result['InvoiceNumber']} status={result['Status']} total={result['Total']} subtotal={result['SubTotal']} tax={result['TotalTax']}")
        return json.dumps(result)
    except Exception as exc:
        _log(f"xero_update_invoice ERROR: {exc}\n{traceback.format_exc()}")
        raise


# ── Claude API client ─────────────────────────────────────────────────────────

client = anthropic.Anthropic()

# Contacts created during the current run — prevents duplicate creation when the
# same new supplier appears on multiple bills before Xero's search index catches up.
_contacts_created_this_run: dict[tuple[str, str], str] = {}


# ── Structured output schema ──────────────────────────────────────────────────

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
                _log(f"  WARNING: text attachment truncated ({len(text)} chars) — review manually")
                text = text[:_MAX_TEXT_CHARS] + "\n[... attachment truncated due to size ...]"
            remaining = _MAX_TOTAL_CHARS - total_chars
            if len(text) > remaining:
                _log("  WARNING: attachment cut at total limit — review manually")
                text = text[:remaining] + "\n[... attachment truncated at total limit ...]"
            blocks.append({"type": "text", "text": text})
            total_chars += len(text)

        elif item.type == "image":
            if len(item.data) > _MAX_IMAGE_B64_LEN:
                _log("  WARNING: image attachment skipped (too large) — review manually")
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

    print("Checking draft bill counts...")
    def _fetch_count(tenant_id: str) -> int:
        try:
            return len(json.loads(xero_list_draft_accpay_invoices(tenant_id)))
        except Exception:
            return -1

    with ThreadPoolExecutor(max_workers=min(len(tenants), 8)) as ex:
        futures = {t["tenantId"]: ex.submit(_fetch_count, t["tenantId"]) for t in tenants}
    bill_counts: dict[str, int] = {tid: f.result() for tid, f in futures.items()}

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

        _log("Fetching chart of accounts and tax rates...")
        with ThreadPoolExecutor(max_workers=2) as ex:
            f_acc = ex.submit(xero_get_accounts, tenant_id)
            f_tax = ex.submit(xero_get_tax_rates, tenant_id)
        accounts_json: str = f_acc.result()
        tax_rates_json: str = f_tax.result()

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
                    query_msg = decision.query
                elif not decision.contact_name:
                    query_msg = "Claude returned an update decision with no contact name — skipping"
                elif not decision.line_items:
                    query_msg = "Claude returned an update decision with no line items — skipping"
                else:
                    query_msg = None

                if query_msg is not None:
                    print(f"  QUERY: {query_msg}")
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
