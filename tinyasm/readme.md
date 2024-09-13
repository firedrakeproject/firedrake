# TinyASM

A simple implementation of PETSc's ASM preconditioner that is focussed on the
case of small matrices. We avoid the overhead of KSP and PC objects for each
block and just use the dense inverse.

Originally hosted on [Florian Wechung's git repo](https://github.com/florianwechsung/TinyASM)
