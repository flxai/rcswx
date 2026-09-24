{
  description = "RCSWX CPU Python/Rust library and development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = {
    nixpkgs,
    rust-overlay,
    ...
  }: let
    system = "x86_64-linux";
    pkgs = import nixpkgs {
      inherit system;
      overlays = [(import rust-overlay)];
    };
    rustToolchain = pkgs.rust-bin.stable."1.85.0".default.override {
      targets = ["wasm32-unknown-unknown"];
    };
    packageSource = pkgs.lib.fileset.toSource {
      root = ./.;
      fileset = pkgs.lib.fileset.unions [
        ./Cargo.toml
        ./Cargo.lock
        ./crates
        ./pyproject.toml
        ./README.md
        ./LICENSE.einsearch
        ./LICENSE.rust-dependencies
        (pkgs.lib.fileset.fileFilter
          (file: file.hasExt "py" || file.hasExt "pyi" || file.name == "py.typed")
          ./python)
      ];
    };
    supportedPython = lib: pythonPackages:
      lib.versionAtLeast pythonPackages.python.pythonVersion "3.12"
      && lib.versionOlder pythonPackages.python.pythonVersion "3.15";
    mkRcswx = pythonPackages: propagatedBuildInputs:
      pythonPackages.buildPythonPackage {
        pname = "rcswx";
        version = "0.1.0";
        pyproject = true;
        src = packageSource;
        cargoDeps = pkgs.rustPlatform.importCargoLock {
          lockFile = "${packageSource}/Cargo.lock";
        };
        nativeBuildInputs = with pkgs.rustPlatform; [
          cargoSetupHook
          maturinBuildHook
        ];
        inherit propagatedBuildInputs;
        pythonImportsCheck = ["rcswx" "rcswx._core"];
      };
    variants = pythonPackages: let
      torchInputs = with pythonPackages; [torch psutil rich termcolor tqdm];
      referenceInputs = with pythonPackages; [numpy scipy];
    in rec {
      rcswx = mkRcswx pythonPackages [];
      rcswx-torch = mkRcswx pythonPackages torchInputs;
      rcswx-reference = mkRcswx pythonPackages referenceInputs;
      rcswx-full = mkRcswx pythonPackages (torchInputs ++ referenceInputs);
    };
  in {
    formatter.${system} = pkgs.alejandra;
    overlays.default = final: prev: {
      pythonPackagesExtensions =
        (prev.pythonPackagesExtensions or [])
        ++ [
          (pyFinal: pyPrev:
            final.lib.optionalAttrs (supportedPython final.lib pyPrev) (variants pyFinal))
        ];
    };
    packages.${system} = let
      built = variants pkgs.python314Packages;
    in {
      inherit (built) rcswx rcswx-torch rcswx-reference rcswx-full;
      default = built.rcswx;
    };
    devShells.${system}.default = pkgs.mkShell {
      packages = with pkgs; [
        rustToolchain
        python314
        uv
        pkg-config
        alejandra
        ruff
        wasm-bindgen-cli
        nodejs
      ];
      LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [pkgs.stdenv.cc.cc.lib pkgs.zlib];
      UV_PYTHON_DOWNLOADS = "never";
      UV_PYTHON = "${pkgs.python314}/bin/python3";
    };
  };
}
