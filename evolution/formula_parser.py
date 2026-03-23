"""
evolution/formula_parser.py
Parse human-readable GP formula strings into expression trees.

Supported grammar (same format as Node.to_string()):
    expr  :=  unary_op '(' expr ')'
           |  '(' expr binary_op expr ')'
           |  feature_name
           |  number
           |  'if(' expr '> 0,' expr ',' expr ')'   # ConditionalNode

Unary ops : neg abs sign sqrt square inv log sigmoid tanh rank zscore
Binary ops: add sub mul div max min avg
"""

from __future__ import annotations
from typing import Tuple, List

from evolution.nodes import (
    Node,
    FeatureNode,
    ConstantNode,
    BinaryOpNode,
    UnaryOpNode,
    ConditionalNode,
)

UNARY_OPS = frozenset(
    {"neg", "abs", "sign", "sqrt", "square", "inv", "log", "sigmoid", "tanh", "rank", "zscore"}
)
BINARY_OPS = frozenset({"add", "sub", "mul", "div", "max", "min", "avg"})

# ─────────────────────────────────────────────────────────────────────────────
# Tokeniser
# ─────────────────────────────────────────────────────────────────────────────

def _tokenize(formula: str) -> List:
    """Return list of tokens.  Each token is one of:
        '(' | ')'  | ',' | '>'
        ('NUM', float)
        ('WORD', str)
    """
    tokens: List = []
    i = 0
    n = len(formula)
    while i < n:
        c = formula[i]
        if c in " \t\n":
            i += 1
            continue
        if c in "(),>":
            tokens.append(c)
            i += 1
            continue
        # Negative number (only if previous token wasn't a closing paren or word)
        if (c == "-" and i + 1 < n and (formula[i + 1].isdigit() or formula[i + 1] == ".")
                and (not tokens or tokens[-1] == "(")):
            j = i + 1
            while j < n and (formula[j].isdigit() or formula[j] == "."):
                j += 1
            tokens.append(("NUM", float(formula[i:j])))
            i = j
            continue
        if c.isdigit() or c == ".":
            j = i + 1
            while j < n and (formula[j].isdigit() or formula[j] == "."):
                j += 1
            tokens.append(("NUM", float(formula[i:j])))
            i = j
            continue
        if c.isalpha() or c == "_":
            j = i + 1
            while j < n and (formula[j].isalnum() or formula[j] == "_"):
                j += 1
            tokens.append(("WORD", formula[i:j]))
            i = j
            continue
        # Skip unknown chars (e.g. stray whitespace artefacts)
        i += 1
    return tokens


# ─────────────────────────────────────────────────────────────────────────────
# Recursive-descent parser
# ─────────────────────────────────────────────────────────────────────────────

def _parse(tokens: List, pos: int) -> Tuple[Node, int]:
    """Return (node, new_pos).  Raises ValueError on parse errors."""
    if pos >= len(tokens):
        raise ValueError("Unexpected end of formula")

    tok = tokens[pos]

    # ── UnaryOpNode:  word '(' expr ')'  ─────────────────────────────────────
    if isinstance(tok, tuple) and tok[0] == "WORD" and tok[1] in UNARY_OPS:
        op = tok[1]
        pos += 1
        _expect(tokens, pos, "(", f"'(' after unary op '{op}'")
        pos += 1
        child, pos = _parse(tokens, pos)
        _expect(tokens, pos, ")", f"')' closing unary op '{op}'")
        pos += 1
        return UnaryOpNode(op, child), pos

    # ── BinaryOpNode function-call style:  word '(' expr ',' expr ')'  ──────
    # Allows writing avg(a, b) instead of (a avg b) — more natural for users.
    if isinstance(tok, tuple) and tok[0] == "WORD" and tok[1] in BINARY_OPS:
        op = tok[1]
        pos += 1
        _expect(tokens, pos, "(", f"'(' after binary op '{op}'")
        pos += 1
        left, pos = _parse(tokens, pos)
        _expect(tokens, pos, ",", f"',' separating args of '{op}'")
        pos += 1
        right, pos = _parse(tokens, pos)
        _expect(tokens, pos, ")", f"')' closing binary op '{op}'")
        pos += 1
        return BinaryOpNode(op, left, right), pos

    # ── ConditionalNode:  'if' '(' expr '>' '0' ',' expr ',' expr ')'  ──────
    if isinstance(tok, tuple) and tok[0] == "WORD" and tok[1] == "if":
        pos += 1
        _expect(tokens, pos, "(", "'(' after 'if'")
        pos += 1
        cond, pos = _parse(tokens, pos)
        # consume "> 0,"
        _expect(tokens, pos, ">", "'>' in conditional")
        pos += 1
        # next token should be 0 or it might be missing — consume optionally
        if pos < len(tokens) and isinstance(tokens[pos], tuple) and tokens[pos][0] == "NUM":
            pos += 1
        _expect(tokens, pos, ",", "',' after condition threshold")
        pos += 1
        true_branch, pos = _parse(tokens, pos)
        _expect(tokens, pos, ",", "',' between conditional branches")
        pos += 1
        false_branch, pos = _parse(tokens, pos)
        _expect(tokens, pos, ")", "')' closing conditional")
        pos += 1
        return ConditionalNode(cond, true_branch, false_branch), pos

    # ── BinaryOpNode:  '(' expr binary_op expr ')'  ──────────────────────────
    if tok == "(":
        pos += 1
        left, pos = _parse(tokens, pos)
        if pos >= len(tokens):
            raise ValueError("Unexpected end inside binary expression")
        op_tok = tokens[pos]
        if not (isinstance(op_tok, tuple) and op_tok[0] == "WORD" and op_tok[1] in BINARY_OPS):
            raise ValueError(
                f"Expected binary operator inside '(…)', got {op_tok!r} at token {pos}"
            )
        op = op_tok[1]
        pos += 1
        right, pos = _parse(tokens, pos)
        _expect(tokens, pos, ")", f"')' closing binary op '{op}'")
        pos += 1
        return BinaryOpNode(op, left, right), pos

    # ── ConstantNode  ─────────────────────────────────────────────────────────
    if isinstance(tok, tuple) and tok[0] == "NUM":
        return ConstantNode(tok[1]), pos + 1

    # ── FeatureNode  ──────────────────────────────────────────────────────────
    if isinstance(tok, tuple) and tok[0] == "WORD":
        return FeatureNode(tok[1]), pos + 1

    raise ValueError(f"Unexpected token {tok!r} at position {pos}")


def _expect(tokens: List, pos: int, expected: str, context: str = ""):
    if pos >= len(tokens) or tokens[pos] != expected:
        got = tokens[pos] if pos < len(tokens) else "EOF"
        raise ValueError(f"Expected {expected!r} ({context}), got {got!r} at token {pos}")


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def parse_formula(formula: str) -> Node:
    """Parse a formula string into an expression tree.

    Args:
        formula: Human-readable formula string produced by Node.to_string() or
                 hand-written in the same format.

    Returns:
        Root Node of the expression tree.

    Raises:
        ValueError: If the formula cannot be parsed.
    """
    formula = formula.strip()
    if not formula:
        raise ValueError("Empty formula")

    tokens = _tokenize(formula)
    if not tokens:
        raise ValueError("Formula tokenised to nothing")

    node, pos = _parse(tokens, 0)
    if pos < len(tokens):
        raise ValueError(
            f"Formula parsed successfully up to token {pos} but trailing tokens remain: "
            f"{tokens[pos:]!r}"
        )
    return node
