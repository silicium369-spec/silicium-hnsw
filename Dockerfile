FROM ann-benchmarks

ARG GITHUB_TOKEN=""

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && \
    mkdir -p /usr/local/lib && \
    if [ -n "$GITHUB_TOKEN" ]; then \
        curl -fsSL -H "Authorization: token $GITHUB_TOKEN" \
            -o /usr/local/lib/libsilicium_ann_x86_64.so \
            "https://github.com/silicium369-spec/silicium-hnsw/releases/download/ann-v1.0.0/libsilicium_ann_x86_64.so"; \
    else \
        curl -fsSL -o /usr/local/lib/libsilicium_ann_x86_64.so \
            "https://github.com/silicium369-spec/silicium-hnsw/releases/download/ann-v1.0.0/libsilicium_ann_x86_64.so"; \
    fi && \
    echo "11bfdd30bfa78fbffc2b9d6ae88b3ea18310a274d684a167e410ca17b266ee58  /usr/local/lib/libsilicium_ann_x86_64.so" | sha256sum -c - && \
    ldconfig && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN python3 -c 'import ctypes; ctypes.CDLL("/usr/local/lib/libsilicium_ann_x86_64.so"); print("[silicium] Build environment verified")'
