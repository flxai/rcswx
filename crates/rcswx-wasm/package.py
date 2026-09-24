"""Complete wasm-pack output using tracked declarations and license sources."""

import json
import shutil
import subprocess
from pathlib import Path

crate = Path(__file__).resolve().parent
root = crate.parent.parent
package = crate / "pkg"
for source in [
    crate / "browser.d.ts",
    root / "LICENSE.einsearch",
    root / "LICENSE.rust-dependencies",
]:
    shutil.copyfile(source, package / source.name)
metadata = json.loads((package / "package.json").read_text())
metadata["files"] = sorted(
    set(metadata.get("files", []))
    | {"browser.d.ts", "LICENSE.einsearch", "LICENSE.rust-dependencies", "THIRD_PARTY_NOTICES"}
)
metadata["exports"] = {
    ".": {"types": "./rcswx_wasm.d.ts", "import": "./rcswx_wasm.js"},
    "./browser": {"types": "./browser.d.ts"},
    "./rcswx_wasm_bg.wasm": "./rcswx_wasm_bg.wasm",
}
(package / "package.json").write_text(json.dumps(metadata, indent=2) + "\n")

graph = json.loads(
    subprocess.check_output(
        [
            "cargo",
            "metadata",
            "--locked",
            "--format-version",
            "1",
            "--filter-platform",
            "wasm32-unknown-unknown",
        ],
        cwd=root,
        text=True,
    )
)
packages = {item["id"]: item for item in graph["packages"]}
nodes = {item["id"]: item for item in graph["resolve"]["nodes"]}
pending = [item["id"] for item in graph["packages"] if item["name"] == "rcswx-wasm"]
seen = set()
while pending:
    identity = pending.pop()
    if identity in seen:
        continue
    seen.add(identity)
    pending.extend(
        dep["pkg"]
        for dep in nodes[identity]["deps"]
        if any(kind["kind"] != "dev" for kind in dep["dep_kinds"])
    )
notices = ["RCSWX WebAssembly dependency notices", (root / "LICENSE.einsearch").read_text()]
for identity in sorted(seen):
    dependency = packages[identity]
    if dependency["source"] is None:
        continue
    directory = Path(dependency["manifest_path"]).parent
    licenses = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.name.upper().startswith(("LICENSE", "COPYING", "NOTICE"))
    )
    if not licenses:
        raise RuntimeError(f"No vendored license text found for {dependency['name']}")
    notices.append(
        f"\n{'=' * 72}\n{dependency['name']} {dependency['version']} ({dependency['license']})"
    )
    notices.extend(f"\n{path.name}\n{path.read_text()}" for path in licenses)
(package / "THIRD_PARTY_NOTICES").write_text("\n".join(notices) + "\n")
public = root / "examples" / "web" / "public"
public.mkdir(exist_ok=True)
shutil.copyfile(package / "THIRD_PARTY_NOTICES", public / "THIRD_PARTY_NOTICES")
