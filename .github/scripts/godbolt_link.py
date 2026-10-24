#!/usr/bin/env python3
"""Point the README's Compiler Explorer links at a given FKL release.

Each backend in the links config (CUDA, CPU, ROCm, ...) has a README link of the form

    [Try FKL v<version> (<label>)](https://godbolt.org/z/<id>)

or, before its first refresh, the placeholder [Try FKL (<label>)](https://godbolt.org/).

For every link that is not on the requested release yet, the example is compiled on
godbolt.org with that backend's compiler and the new FKL version, a new short link is saved
and the README link is rewritten. A backend whose compile fails keeps its old link and is
reported. Nothing happens while the release is not on Compiler Explorer yet.

Writes `changed`, `failed` and a markdown `summary` to $GITHUB_OUTPUT when run in GitHub
Actions, and exits non-zero when any backend failed.
"""

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request

API = "https://godbolt.org/api"
LIB_ID = "fusedkernellibrary"
LINK_RE = re.compile(
    r"\[Try FKL(?: v(?P<version>[^\]\s]+))? \((?P<label>[^)]+)\)\]"
    r"\((?P<url>https://godbolt\.org/[^)\s]*)\)"
)


def request(path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        f"{API}/{path}",
        data=data,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "FusedKernelLibrary-godbolt-link",
        },
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.load(resp)


def set_output(**values):
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a", encoding="utf-8") as f:
            for key, value in values.items():
                f.write(f"{key}<<__EOF__\n{value}\n__EOF__\n")


class CompilerExplorer:
    """Caches the per-language library and compiler listings."""

    def __init__(self):
        self._libs = {}
        self._compilers = {}

    def version_id(self, language, version):
        if language not in self._libs:
            self._libs[language] = request(f"libraries/{language}")
        for lib in self._libs[language]:
            if lib["id"] == LIB_ID:
                for v in lib["versions"]:
                    if v["version"] == version:
                        return v["id"]
        return None

    def has_compiler(self, language, compiler):
        if language not in self._compilers:
            self._compilers[language] = {
                c["id"] for c in request(f"compilers/{language}?fields=id")
            }
        return compiler in self._compilers[language]


def compile_errors(backend, source, ver_id):
    """Return None when the example compiles, else the compiler's error output."""
    result = request(
        f"compiler/{backend['compiler']}/compile",
        {
            "source": source,
            "lang": backend["language"],
            "options": {
                "userArguments": backend.get("options", ""),
                "libraries": [{"id": LIB_ID, "version": ver_id}],
                "filters": {"binary": False, "execute": False},
            },
        },
    )
    if result.get("code", 1) == 0:
        return None
    lines = [re.sub(r"\x1b\[[0-9;]*m", "", line.get("text", "")) for line in result.get("stderr", [])]
    return "\n".join(line for line in lines if "error" in line.lower()) or "\n".join(lines)


def client_state(backend, source, ver_id):
    return {
        "sessions": [
            {
                "id": 1,
                "language": backend["language"],
                "source": source,
                "compilers": [
                    {
                        "id": backend["compiler"],
                        "options": backend.get("options", ""),
                        "libs": [{"name": LIB_ID, "ver": ver_id}],
                    }
                ],
            }
        ]
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readme", default="README.md")
    parser.add_argument("--config", default=".github/godbolt/links.json")
    parser.add_argument("--source", default=".github/godbolt/playground.cpp")
    parser.add_argument("--version", required=True, help="Release tag, e.g. Beta-0.2.0")
    parser.add_argument("--dry-run", action="store_true", help="Check everything, save nothing")
    args = parser.parse_args()

    set_output(changed="false", failed="", summary="")
    with open(args.readme, encoding="utf-8") as f:
        readme = f.read()
    with open(args.config, encoding="utf-8") as f:
        backends = json.load(f)
    links = {m["label"]: m for m in LINK_RE.finditer(readme)}

    missing = [b["label"] for b in backends if b["label"] not in links]
    if missing:
        sys.exit(
            f"No README link for {', '.join(missing)}. Add e.g. "
            f"'[Try FKL ({missing[0]})](https://godbolt.org/)' to {args.readme}"
        )

    version = args.version.removeprefix("Beta-")
    ce = CompilerExplorer()
    replacements = {}  # label -> new markdown link
    updated, failed = [], []

    for backend in backends:
        label = backend["label"]
        if links[label]["version"] == version:
            print(f"{label}: already links to {version}")
            continue

        ver_id = ce.version_id(backend["language"], args.version)
        if ver_id is None:
            print(f"{args.version} is not on Compiler Explorer ({backend['language']}) yet")
            return

        source_path = backend.get("source", args.source)
        with open(source_path, encoding="utf-8") as f:
            source = f.read()

        if not ce.has_compiler(backend["language"], backend["compiler"]):
            error = f"compiler {backend['compiler']} is not on Compiler Explorer any more"
        else:
            try:
                error = compile_errors(backend, source, ver_id)
            except urllib.error.URLError as e:
                error = f"compile request failed: {e}"
        if error:
            first_lines = "\n".join(error.splitlines()[:10])
            print(f"{label}: FAILED with {backend['compiler']}\n{first_lines}")
            failed.append((label, backend["compiler"], error))
            continue
        print(f"{label}: example compiles with {backend['compiler']} and {LIB_ID} {ver_id}")

        if args.dry_run:
            continue
        url = request("shortener", client_state(backend, source, ver_id))["url"]
        replacements[label] = f"[Try FKL v{version} ({label})]({url})"
        updated.append((label, backend["compiler"], url))

    if replacements:
        readme = LINK_RE.sub(lambda m: replacements.get(m["label"], m[0]), readme)
        with open(args.readme, "w", encoding="utf-8", newline="") as f:
            f.write(readme)

    summary = [f"- ✅ **{label}** (`{compiler}`): {url}" for label, compiler, url in updated]
    summary += [
        f"- ❌ **{label}** (`{compiler}`): link not updated, the example does not compile:\n"
        f"  ```\n  {error.splitlines()[0] if error else ''}\n  ```"
        for label, compiler, error in failed
    ]
    set_output(
        changed="true" if replacements else "false",
        failed=", ".join(label for label, _, _ in failed),
        summary="\n".join(summary),
    )
    if failed:
        sys.exit(f"Example failed to compile for: {', '.join(label for label, _, _ in failed)}")


if __name__ == "__main__":
    main()
