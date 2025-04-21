# rust-gemm
A GEMM like set of functions that use CPU vector instructions

To run the code:  
rustup override set nightly  
RUSTFLAGS="-C target-cpu=native" cargo run --release  
