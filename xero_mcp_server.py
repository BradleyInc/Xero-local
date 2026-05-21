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


def _log(msg: str) -> None:
    """Write a timestamped diagnostic line to stderr (never stdout — that carries MCP protocol)."""
    print(f"[xero-mcp {datetime.now().strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


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
        raw = resp.text[:4000]
        detail = raw
        try:
            data = resp.json()
            # Xero wraps validation failures inside Elements[n].ValidationErrors
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


def _refresh(refresh_token: str) -> dict:
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
    data = resp.json()
    _save_token(data)
    _log("Token refreshed successfully.")
    return data


def _get_token() -> dict | None:
    """Return a valid token, refreshing silently if expired."""
    token = _load_token()
    if not token:
        _log("No token file found — authentication required.")
        return None
    expires_at = token.get("expires_at", 0)
    if datetime.now().timestamp() > expires_at - 60:
        secs_ago = int(datetime.now().timestamp() - expires_at)
        _log(f"Token expired {secs_ago}s ago — refreshing.")
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
    _log("xero_authenticate: starting OAuth flow")
    try:
        _run_oauth_flow()
        _log("xero_authenticate: OAuth flow completed successfully")
        return "Authentication successful. Token saved."
    except Exception as exc:
        _log(f"xero_authenticate ERROR: {exc}\n{traceback.format_exc()}")
        raise RuntimeError(f"Authentication failed: {exc}") from exc


@mcp.tool()
def xero_list_tenants() -> str:
    """
    List all Xero organisations (tenants) connected to this app, returning each
    tenant's name, tenantId, and tenantType.
    """
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


@mcp.tool()
def xero_list_draft_accpay_invoices(tenant_id: str) -> str:
    """
    List all DRAFT accounts-payable (ACCPAY) bills for the given Xero tenant.

    Args:
        tenant_id: The tenantId GUID from xero_list_tenants.
    """
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

        result: list[TextContent | ImageContent] = []
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
                result.append(TextContent(type="text", text=json.dumps({
                    "FileName": filename, "error": str(exc)
                })))
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
                result.append(TextContent(type="text", text=json.dumps(metadata, indent=2)))
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
    except Exception as exc:
        _log(f"xero_get_invoice_attachments ERROR: {exc}\n{traceback.format_exc()}")
        raise


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


@mcp.tool()
def xero_create_contact(tenant_id: str, name: str, email: str | None = None) -> str:
    """
    Create a new contact (supplier/customer) in a Xero tenant.

    Args:
        tenant_id: The tenantId GUID from xero_list_tenants.
        name:      The contact's display name. Must be unique within the tenant.
        email:     Optional email address.
    """
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

@mcp.tool()
def xero_get_accounts(tenant_id: str, include_all: bool = False) -> str:
    """
    Return the chart of accounts for a Xero tenant. By default returns only
    AP-relevant account types (expenses, liabilities, current assets) to keep
    the response compact. Pass include_all=true to return every active account.

    Args:
        tenant_id:   The tenantId GUID from xero_list_tenants.
        include_all: If true, return all active accounts regardless of type.
    """
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


@mcp.tool()
def xero_get_tax_rates(tenant_id: str) -> str:
    """
    Return all tax rates for a Xero tenant (e.g. GST on Expenses, GST Free
    Expenses, BAS Excluded).

    Args:
        tenant_id: The tenantId GUID from xero_list_tenants.
    """
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
    Update fields on a DRAFT ACCPAY bill. Passing line_items replaces ALL
    existing lines. Each line item: Description, AccountCode, UnitAmount,
    TaxType ("INPUT2"/"EXEMPTEXPENSES"/"BASEXCLUDED"), Quantity.
    UnitAmount is GST-inclusive when the bill's LineAmountTypes is
    INCLUSIVE, or GST-exclusive when EXCLUSIVE — match the bill's existing
    setting (check via xero_get_invoice). LineAmountTypes is omitted to
    preserve the bill's existing setting. contact_name must exactly match
    a Xero contact.

    Args:
        tenant_id:    tenantId GUID from xero_list_tenants.
        invoice_id:   InvoiceID GUID of the bill to update.
        contact_name: Exact supplier name from xero_get_contacts.
        date:         Invoice date (YYYY-MM-DD).
        due_date:     Due date (YYYY-MM-DD).
        reference:    Supplier invoice number/reference.
        line_items:   Replaces all existing line items.
    """
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


if __name__ == "__main__":
    mcp.run()
