# insight-vio C++17 + Eigen

Incremental port of the Python MSCKF stack. See [`docs/cpp_port.md`](../docs/cpp_port.md)
for the module map and remaining work.

## Quick build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

Requires CMake ≥ 3.16 and a C++17 compiler. Eigen and GoogleTest are downloaded on configure.
