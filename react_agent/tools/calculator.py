"""Calculator tool with safe expression evaluation."""


def calculator(expression: str) -> str:
    """Safely evaluate a mathematical expression and return the result."""
    try:
        allowed_names = {"abs": abs, "round": round}
        code = compile(expression, "<string>", "eval")
        for name in code.co_names:
            if name not in allowed_names:
                raise NameError(f"Use of {name} not allowed")

        result = eval(code, {"__builtins__": {}}, allowed_names)
        return str(float(result))
    except Exception as e:
        return f"Error evaluating expression: {e}"
