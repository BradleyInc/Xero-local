@echo off
for /f "usebackq tokens=1,* delims==" %%a in ("%~dp0.env") do (
    if not "%%a"=="" if not "%%a:~0,1%"=="#" set "%%a=%%b"
)
"C:\Program Files\nodejs\npx.cmd" -y @xeroapi/xero-mcp-server@latest
