from __future__ import absolute_import, print_function, division


__all__ = ["pre_traverse_dags", "post_traverse_dags"]


def pre_traverse_dags(expr_dag):
    """
    """
    seen = set()
    container = [expr_dag]

    while container:
        tensor = container.pop()
        yield tensor

        for operand in tensor.operands:
            if operand not in seen:
                seen.add(operand)
                container.append(operand)


def post_traverse_dags(expr_dag):
    """
    """
    seen = set()
    container = []
    container.append((expr_dag, list(expr_dag.operands)))

    while container:
        tensor, operands = container[-1]
        for i, operand in enumerate(operands):
            if operand not in seen and operand is not None:
                container.append((operand, list(operand.operands)))
                operands[i] = None
                break
        else:
            seen.add(tensor)
            container.pop()
            yield tensor
