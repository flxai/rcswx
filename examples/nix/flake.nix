{
  description = "A Python project using rcswx's PyTorch integration";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    rcswx.url = "github:flxai/rcswx";
    rcswx.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = {
    nixpkgs,
    rcswx,
    ...
  }: let
    system = "x86_64-linux";
    pkgs = import nixpkgs {
      inherit system;
      overlays = [rcswx.overlays.default];
    };
    python = pkgs.python314.withPackages (ps: [
      ps.rcswx-torch
      # Use ps.rcswx for portable trees only, ps.rcswx-reference for only
      # NumPy/SciPy reference selection, or ps.rcswx-full for both extras.
    ]);
  in {
    devShells.${system}.default = pkgs.mkShell {
      packages = [python];
    };
  };
}
