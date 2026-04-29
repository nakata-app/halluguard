# Security policy

## Reporting a vulnerability

Please **do not** open a public GitHub issue for security-sensitive
findings. Instead, email the maintainer at
**hey@nakata.app** with:

- A description of the issue.
- Steps to reproduce (a minimal repro is enough).
- The version / commit you tested against.
- Optionally, your proposed fix.

We aim to acknowledge a report within 72 hours and to ship a fix in
the next minor release where applicable.

## Scope

In scope: the public Python API (`Guard`, `NLIVerifier`,
`SupportReport`, `DaemonEncoder`), the CLI, the benchmark harnesses,
and our own tests.

Out of scope:
- Bugs in third-party NLI / encoder models we depend on. Report
  those to the upstream library (`sentence-transformers`,
  `transformers`, `torch`).
- Performance issues without a security impact (file regular issues
  instead).
- Adversarial inputs that *fool* the NLI verifier into a wrong call, 
  these are accuracy issues, not security ones. Open a regular issue
  with a reproduction.

## Threat model, daemon mode

`Guard.from_daemon` calls a local `adaptmem serve` process. The
daemon is **localhost-only, single-user** by default and does not
authenticate. Do not point `daemon_url` at a public-internet host
unless you've put an auth proxy in front.
