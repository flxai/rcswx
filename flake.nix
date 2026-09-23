{
  description = "RCSWX CPU Python/Rust library and development environment";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = {nixpkgs, ...}: let
    system = "x86_64-linux";
    pkgs = nixpkgs.legacyPackages.${system};
    supportedPython = lib: pythonPackages:
      lib.versionAtLeast pythonPackages.python.pythonVersion "3.12"
      && lib.versionOlder pythonPackages.python.pythonVersion "3.15";
    mkRcswx = pkgs: pythonPackages:
      pythonPackages.buildPythonPackage {
        pname = "rcswx";
        version = "0.1.0";
        pyproject = true;
        src = ./.;
        cargoDeps = pkgs.rustPlatform.importCargoLock {
          lockFile = ./Cargo.lock;
        };
        nativeBuildInputs = with pkgs.rustPlatform; [
          cargoSetupHook
          maturinBuildHook
        ];
        propagatedBuildInputs = with pythonPackages; [
          numpy
          psutil
          rich
          scipy
          termcolor
          torch
          tqdm
        ];
        pythonRelaxDeps = ["rich"];
        pythonImportsCheck = ["rcswx"];
      };
  in {
    formatter.${system} = pkgs.alejandra;
    overlays.default = final: prev: {
      pythonPackagesExtensions =
        (prev.pythonPackagesExtensions or [])
        ++ [
          (pyFinal: pyPrev:
            final.lib.optionalAttrs (supportedPython final.lib pyPrev) {
              rcswx = mkRcswx final pyFinal;
            })
        ];
    };
    packages.${system} = rec {
      rcswx = mkRcswx pkgs pkgs.python314Packages;
      default = rcswx;
    };
    devShells.${system}.default = pkgs.mkShell {
      packages = with pkgs; [cargo rustc rustfmt clippy python314 uv pkg-config alejandra ruff];
      LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [pkgs.stdenv.cc.cc.lib pkgs.zlib];
      UV_PYTHON_DOWNLOADS = "never";
      UV_PYTHON = "${pkgs.python314}/bin/python3";
    };
  };
}
