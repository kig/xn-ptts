# Optimized Pocket-TTS WebSocket server — drop-in replacement for
# Dockerfile.xnptts.current (same bases, same CMD shape, same /models mount).
#
# Differences from the current image:
#   - builds from the local tree instead of `git clone` of master, so it
#     includes the q8_0-quantized KV cache commit (3.8x less KV memory);
#   - RAYON_NUM_THREADS=1 baked in: xn fans every generation over the rayon
#     pool; 1 stream x 16 threads measured 0.74x realtime vs 18.5x for 16
#     concurrent single-threaded streams (experiments/notebook.md). With this
#     env, each ws connection becomes its own single-threaded stream.
#   - deploy with --quant q8_0 (1.4-1.8x over f32) and --cpuset-cpus=0-15
#     (physical cores only): see docker-compose.yml.
FROM rust:1-slim AS builder
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake pkg-config libclang-dev protobuf-compiler ca-certificates git \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /build
COPY Cargo.toml Cargo.lock rustfmt.toml ./
COPY .cargo ./.cargo
COPY ptts ptts/
COPY ptts-pyo3 ptts-pyo3/
COPY ptts-wasm ptts-wasm/
COPY ptts-ws-server ptts-ws-server/
# xn pins target-cpu=native via .cargo/config.toml; keep for host build.
RUN cargo build --release -p ptts-ws-server

FROM debian:trixie-slim
# curl: the deploy entrypoint (scripts/fetch-tts-models.sh in the paperradio
# repo) downloads the gated model artifacts into /models before exec'ing the
# server; the slim base has no HTTP client at all.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*
COPY --from=builder /build/target/release/ptts-ws-server /usr/local/bin/ptts-ws-server
ENV RAYON_NUM_THREADS=1
EXPOSE 8000
CMD ["ptts-ws-server", "--addr", "0.0.0.0:8000"]
