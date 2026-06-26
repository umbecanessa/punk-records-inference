# Security policy

## Supported versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |

## Reporting a vulnerability

Please **do not** open a public GitHub issue for security-sensitive reports.

Instead, use [GitHub private vulnerability reporting](https://github.com/umbecanessa/punk-records-inference/security/advisories/new) on this repository, or email the maintainers through a GitHub issue with the title `[security]` if private reporting is unavailable.

Include:

- Affected component (connector, agent shim, admin API, Docker image, etc.)
- Steps to reproduce
- Impact assessment (data exposure, privilege escalation, denial of service)

We aim to acknowledge reports within **5 business days** and publish fixes or mitigations as soon as practical.

## Scope notes

- PRI runs **locally** with your BYOC checkpoint — treat `/data/pri` captures as sensitive model state.
- The admin API (`NLS_ADMIN_*`) should not be exposed to untrusted networks without authentication.
- Do not commit `.env`, API keys, or capture artifacts to the repository.
