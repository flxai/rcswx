{
  description = "RCSWX CPU Python/Rust development environment";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = {nixpkgs, ...}: let
    system = "x86_64-linux";
    pkgs = nixpkgs.legacyPackages.${system};
  in {
    formatter.${system} = pkgs.alejandra;
    devShells.${system}.default = pkgs.mkShell {
      packages = with pkgs; [cargo rustc rustfmt clippy python314 uv pkg-config alejandra ruff];
      LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [pkgs.stdenv.cc.cc.lib pkgs.zlib];
      UV_PYTHON_DOWNLOADS = "never";
      UV_PYTHON = "${pkgs.python314}/bin/python3";
    };
  };
}
