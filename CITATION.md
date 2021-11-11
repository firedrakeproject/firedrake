# Citing Firedrake

If you publish results using Firedrake, we would be grateful if you would cite the relevant papers.

Please visit https://www.firedrakeproject.org/citing.html for the instructions.

The simplest way to determine the relevant papers is by asking Firedrake itself. You can ask that a list of citations relevant to your computation be printed when exiting by calling `Citations.print_at_exit` after importing Firedrake:
```python
from firedrake import *

Citations.print_at_exit()
```
Alternatively, you can pass a command-line option `-citations` to obtain the same result.

## Archiving your code with Zenodo

In order to make your simulation results traceable and reproducible, we can provide you with a citeable archive of the exact version of Firedrake and its key dependencies that you used in your simulations. For information on how to do this see https://firedrakeproject.org/zenodo.html.
