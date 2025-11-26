# Third-Party Dependencies

Place the upstream repositories that VGGT-Qwen3 relies on inside this directory.

Recommended layout:

```
third_party/
├── Qwen3/    # clone from https://github.com/QwenLM/Qwen3
└── vggt/     # clone from https://github.com/Sycamorers/vggt (or your fork)
```

The main project `.gitignore` keeps the contents of this folder out of version control so you can drop in private or proprietary repos locally. If you prefer Git submodules, remove the `third_party/*` ignore rule and add submodules manually.
