// build.rs

fn main() {
    // Cargo sets this to the target triple, e.g. "riscv64gc-unknown-linux-gnu"
    let target = std::env::var("TARGET").unwrap();

    // Only compile our RVV C code when the target arch is RISC-V 64
    if target.starts_with("riscv64") {
        cc::Build::new()
            .file("src/vector_dot.c")
            .flag("-O3")
            .flag("-march=rv64gcv")
            .flag("-mabi=lp64d")
            .compile("vector_dot");

        // Tell the linker where to find the .a and what to link
        let out_dir = std::env::var("OUT_DIR").unwrap();
        println!("cargo:rustc-link-search=native={}", out_dir);
        println!("cargo:rustc-link-lib=static=vector_dot");
    }

    // If you're also building matmul8x8.c for RVV, you can do the same
    // under the same `if target.starts_with("riscv64")` guard.
}
