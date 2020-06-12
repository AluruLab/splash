include directory contains the following subdirectories:
- algo:     reusable algorithms
- ds:       reusable data structures
- io:       file reader and writer for commonly used file types
- kernel:   base kernel (transformation) interfaces and common kernels.
- patterns: common data processing patterns, including sequential and parallel patterns.

dependencies:
    ds
        - io
        - kernel
            - algo
            - pattern