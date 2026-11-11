# Security policy

This is a research codebase for portfolio research. It contains no production secrets,
no live trading credentials, and no execution connectivity to a broker.

## Reporting a vulnerability

Please report security issues privately to **mrpandanarang@gmail.com**. Do not open a
public issue for a security report. Include the affected version, a minimal
reproduction, and the impact you believe it has.

## Supported versions

| Version | Supported |
|:--|:--|
| 2.x | yes |
| 1.x | no (superseded by the Stillwater rewrite) |

## Scope

The most relevant security surface is the optional `download` extra (`yfinance`) and
any local artifact paths passed to the CLI. The default workflow uses the archived
snapshot in `data/raw/` and does not require network access.
