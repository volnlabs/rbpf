# AxiomOS managed-controller fixtures

The C sources, header, and build script are copied unchanged from AxiomOS
`feat/v0.5-runtime` at commit `9be704c8ef9078c84b98efc5482cfef0389f89f0`,
directory `examples/bpf/managed`. The checked-in `.bin` files contain their
actual little-endian BPF `.text`, compiled with Clang 22.1.8 using the supplied
`-target bpfel -mcpu=v3 -O2 -g0 -Wall -Werror` flags.

Regenerate outside the fixture directory and compare before updating:

```sh
tests/fixtures/axiomos/build.sh target/managed-bpf
for name in conservative_obstacle slow_approach clear_streak; do
    cmp "tests/fixtures/axiomos/$name.bin" "target/managed-bpf/$name.bin"
done
```

Ordinary Cargo tests use the checked-in bytecode and require neither Clang nor
an AxiomOS checkout. Instruction counts are 11, 13, and 32 respectively.

The native fixture harness models the R1 context wrapper, 56-byte payload,
8 KiB stack, instance-private map, and helpers 5/6/1009. It checks outcomes
against the C controllers' expected behavior, including reused code and helper
failure. It does not run AxiomOS's verifier, kernel memory mappings, scheduler,
or physical actuation path, and makes no hardware or timing qualification claim.
