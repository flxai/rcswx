{
  description = "A Python project using rcswx";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    rcswx.url = "github:flxai/rcswx";
    rcswx.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = { nixpkgs, rcswx, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        overlays = [ rcswx.overlays.default ];
      };
      python = pkgs.python314.withPackages (ps: [
        ps.rcswx
        # Add other Python libraries here.
      ]);
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        packages = [ python ];
      };
    };
}
