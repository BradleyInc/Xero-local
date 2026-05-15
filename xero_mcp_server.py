import base64
import io
import json
import os
import re
import threading
import time
import webbrowser
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

import pypdf
import requests
from mcp.server.fastmcp import FastMCP
from mcp.types import ImageContent, TextContent

# Load .env from same directory as this script
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

CLIENT_ID = os.environ["XERO_CLIENT_ID"]
CLIENT_SECRET = os.environ["XERO_CLIENT_SECRET"]
REDIRECT_URI = "http://localhost:8080/callback"
SCOPES = "offline_access accounting.transactions accounting.contacts accounting.settings accounting.reports.read"
TOKEN_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "xero_token.json")
XERO_API_BASE = "https://api.xero.com/api.xro/2.0"
REQUEST_TIMEOUT = 30

mcp = FastMCP("Xero")


# ── Token helpers ────────────────────────────────────────────────────────────

def _b64(s: str) -> str:
    return base64.b64encode(s.encode()).decode()


def _basic_auth_header() -> str:
    return f"Basic {_b64(f'{CLIENT_ID}:{CLIENT_SECRET}')}"


def _load_token() -> dict | None:
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE) as f:
            return json.load(f)
    return None


def _save_token(data: dict) -> None:
    data["expires_at"] = (
        datetime.now() + timedelta(seconds=data.get("expires_in", 1800))
    ).timestamp()
    with open(TOKEN_FILE, "w") as f:
        json.dump(data, f)


def _raise_for_status(resp: requests.Response) -> None:
    if not resp.ok:
        raise RuntimeError(f"{resp.status_code} error: {resp.text[:2000]}")


def _refresh(refresh_token: str) -> dict:
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
    data = resp.json()
    _save_token(data)
    return data


def _get_token() -> dict | None:
    """Return a valid token, refreshing silently if expired."""
    token = _load_token()
    if not token:
        return None
    if datetime.now().timestamp() > token.get("expires_at", 0) - 60:
        token = _refresh(token["refresh_token"])
    return token


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


# ── OAuth flow ───────────────────────────────────────────────────────────────

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


def _run_oauth_flow() -> dict:
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
    data = resp.json()
    _save_token(data)
    return data


# ── MCP tools ────────────────────────────────────────────────────────────────

@mcp.tool()
def xero_authenticate() -> str:
    """
    Authenticate with Xero via OAuth 2.0 (web app / authorization code flow).
    Opens a browser window for login. Only needed once — the token is saved and
    auto-refreshed for subsequent calls.
    """
    _run_oauth_flow()
    return "Authentication successful. Token saved."


@mcp.tool()
def xero_list_tenants() -> str:
    """
    List all Xero organisations (tenants) connected to this app, returning each
    tenant's name, tenantId, and tenantType.
    """
    resp = requests.get(
        "https://api.xero.com/connections",
        headers=_auth_headers(),
        timeout=REQUEST_TIMEOUT,
    )
    _raise_for_status(resp)
    tenants = resp.json()
    return json.dumps(
        [{"name": t["tenantName"], "tenantId": t["tenantId"], "type": t["tenantType"]}
         for t in tenants],
        indent=2,
    )


@mcp.tool()
def xero_list_draft_accpay_invoices(tenant_id: str) -> str:
    """
    List all DRAFT accounts-payable (ACCPAY) bills for the given Xero tenant.

    Args:
        tenant_id: The tenantId GUID from xero_list_tenants.
    """
    resp = requests.get(
        f"{XERO_API_BASE}/Invoices",
        headers=_auth_headers(tenant_id),
        params={"where": 'Type=="ACCPAY" AND Status=="DRAFT"'},
        timeout=REQUEST_TIMEOUT,
    )
    _raise_for_status(resp)
    invoices = resp.json().get("Invoices", [])
    # Return a concise summary to keep response size manageable
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
    return json.dumps(summary, indent=2)


@mcp.tool()
def xero_get_invoice_attachments(
    tenant_id: str, invoice_id: str
) -> list[TextContent | ImageContent]:
    """
    List all attachments on a specific invoice and retrieve their content.
    PDFs and text files are returned as extracted text. Images (PNG, JPEG,
    etc.) are returned as MCP image content so they can be viewed directly.

    Args:
        tenant_id:  The tenantId GUID from xero_list_tenants.
        invoice_id: The InvoiceID GUID from xero_list_draft_accpay_invoices.
    """
    headers = _auth_headers(tenant_id)
    resp = requests.get(
        f"{XERO_API_BASE}/Invoices/{invoice_id}/Attachments",
        headers=headers,
        timeout=REQUEST_TIMEOUT,
    )
    _raise_for_status(resp)
    attachments = resp.json().get("Attachments", [])

    result: list[TextContent | ImageContent] = []
    for a in attachments:
        filename = a.get("FileName", "")
        mime_type = a.get("MimeType", "")

        content_headers = dict(headers)
        content_headers["Accept"] = mime_type or "*/*"
        content_resp = requests.get(
            f"{XERO_API_BASE}/Invoices/{invoice_id}/Attachments/{filename}",
            headers=content_headers,
            timeout=REQUEST_TIMEOUT,
        )
        _raise_for_status(content_resp)

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
            result.append(TextContent(type="text", text=json.dumps(metadata, indent=2)))
        elif mime_type == "application/pdf":
            reader = pypdf.PdfReader(io.BytesIO(content_resp.content))
            pages = [page.extract_text() or "" for page in reader.pages]
            metadata["Content"] = "\n".join(pages).strip()
            result.append(TextContent(type="text", text=json.dumps(metadata, indent=2)))
        elif is_image:
            result.append(TextContent(type="text", text=json.dumps(metadata, indent=2)))
            result.append(ImageContent(
                type="image",
                data=base64.b64encode(content_resp.content).decode(),
                mimeType=mime_type,
            ))
        else:
            metadata["Content"] = base64.b64encode(content_resp.content).decode()
            metadata["ContentEncoding"] = "base64"
            result.append(TextContent(type="text", text=json.dumps(metadata, indent=2)))

    return result


@mcp.tool()
def xero_get_contacts(tenant_id: str, search: str = "") -> str:
    """
    Search for contacts (suppliers/customers) in a Xero tenant.
    Use this to confirm a supplier's exact name before updating a bill.

    Args:
        tenant_id: The tenantId GUID from xero_list_tenants.
        search:    Optional name/email search term (partial match). Leave blank
                   to list all active contacts (may be large).
    """
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
    return json.dumps(
        [
            {
                "ContactID": c.get("ContactID"),
                "Name": c.get("Name"),
                "EmailAddress": c.get("EmailAddress"),
            }
            for c in contacts
        ],
        indent=2,
    )


@mcp.tool()
def xero_get_invoice(tenant_id: str, invoice_id: str) -> str:
    """
    Get full details of a specific invoice/bill, including all current line
    items, totals, and contact. Use this to inspect a bill's current state
    before deciding what to change.

    Args:
        tenant_id:  The tenantId GUID from xero_list_tenants.
        invoice_id: The InvoiceID GUID from xero_list_draft_accpay_invoices.
    """
    resp = requests.get(
        f"{XERO_API_BASE}/Invoices/{invoice_id}",
        headers=_auth_headers(tenant_id),
        timeout=REQUEST_TIMEOUT,
    )
    _raise_for_status(resp)
    invoices = resp.json().get("Invoices", [])
    if not invoices:
        return json.dumps({"error": "Invoice not found"})
    inv = invoices[0]
    return json.dumps(
        {
            "InvoiceID": inv.get("InvoiceID"),
            "InvoiceNumber": inv.get("InvoiceNumber"),
            "Type": inv.get("Type"),
            "Status": inv.get("Status"),
            "Contact": inv.get("Contact", {}),
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
        indent=2,
    )


@mcp.tool()
def xero_get_accounts(tenant_id: str) -> str:
    """
    Return the full chart of accounts for a Xero tenant (all active accounts).

    Args:
        tenant_id: The tenantId GUID from xero_list_tenants.
    """
    resp = requests.get(
        f"{XERO_API_BASE}/Accounts",
        headers=_auth_headers(tenant_id),
        params={"where": 'Status=="ACTIVE"'},
        timeout=REQUEST_TIMEOUT,
    )
    _raise_for_status(resp)
    accounts = resp.json().get("Accounts", [])
    return json.dumps(
        [
            {
                "AccountID": a.get("AccountID"),
                "Code": a.get("Code"),
                "Name": a.get("Name"),
                "Type": a.get("Type"),
                "TaxType": a.get("TaxType"),
            }
            for a in accounts
        ],
        indent=2,
    )


@mcp.tool()
def xero_get_tax_rates(tenant_id: str) -> str:
    """
    Return all tax rates for a Xero tenant (e.g. GST on Expenses, GST Free
    Expenses, BAS Excluded).

    Args:
        tenant_id: The tenantId GUID from xero_list_tenants.
    """
    resp = requests.get(
        f"{XERO_API_BASE}/TaxRates",
        headers=_auth_headers(tenant_id),
        params={"where": 'Status=="ACTIVE"'},
        timeout=REQUEST_TIMEOUT,
    )
    _raise_for_status(resp)
    rates = resp.json().get("TaxRates", [])
    return json.dumps(
        [
            {
                "Name": r.get("Name"),
                "TaxType": r.get("TaxType"),
                "EffectiveRate": r.get("EffectiveRate"),
                "Status": r.get("Status"),
            }
            for r in rates
        ],
        indent=2,
    )


@mcp.tool()
def xero_update_invoice(
    tenant_id: str,
    invoice_id: str,
    contact_name: str | None = None,
    date: str | None = None,
    due_date: str | None = None,
    reference: str | None = None,
    line_items: list[dict] | None = None,
) -> str:
    """
    Update fields on a DRAFT accounts-payable bill in Xero.

    Invoice Total and Total GST are computed by Xero from line items — they
    cannot be set independently. To control them, pass line_items where each
    item carries:
      - UnitAmount: the GST-inclusive dollar amount for that line. Bills in
        Xero default to Inclusive mode, so amounts are interpreted as totals
        and Xero back-calculates the tax component. For INPUT lines: pass the
        invoice total including GST (GST = UnitAmount / 11). For
        EXEMPTEXPENSES / BASEXCLUDED lines: pass the full amount (no tax).
      - TaxType: controls GST treatment and therefore the GST component:
          "INPUT"          → 10% GST on Expenses (GST = UnitAmount / 11)
          "EXEMPTEXPENSES" → GST Free Expenses    (TotalTax = 0)
          "BASEXCLUDED"    → BAS Excluded          (TotalTax = 0, not on BAS)
      - AccountCode: the account to post to (from xero_get_accounts)
      - Description: narrative for this line
      - Quantity: number of units (omit or set 1 for invoices)

    LineAmountTypes is intentionally omitted from the update payload so Xero
    retains the bill's existing setting (Inclusive). Do not add it — both
    "EXCLUSIVE" and "INCLUSIVE" are rejected by the Xero API on updates.
    Passing line_items replaces ALL existing line items on the bill.

    Args:
        tenant_id:    The tenantId GUID from xero_list_tenants.
        invoice_id:   The InvoiceID GUID of the bill to update.
        contact_name: Supplier name — must match an existing Xero contact
                      exactly (use xero_get_contacts to find the right name).
        date:         Invoice date in YYYY-MM-DD format.
        due_date:     Payment due date in YYYY-MM-DD format.
        reference:    Supplier invoice number / reference from the document.
        line_items:   List of line item dicts (replaces all existing lines).
    """
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

    headers = _auth_headers(tenant_id)
    headers["Content-Type"] = "application/json"

    resp = requests.post(
        f"{XERO_API_BASE}/Invoices/{invoice_id}",
        headers=headers,
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )
    _raise_for_status(resp)
    updated = resp.json().get("Invoices", [{}])[0]
    return json.dumps(
        {
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
            "LineItems": updated.get("LineItems", []),
        },
        indent=2,
    )


if __name__ == "__main__":
    mcp.run()
