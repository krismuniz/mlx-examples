from typing import Callable, Optional, Union

END = {"@pop": {}}

TokenTrie = dict[Union[str, int], Optional["TokenTrie"]]
PrimitiveOrTokenTrie = Union[str, int, TokenTrie]

CallableClosing = lambda node: END
Closing = Union[CallableClosing, PrimitiveOrTokenTrie]


def Token(value: PrimitiveOrTokenTrie):
    return T(value) if isinstance(value, str) else value


def ClosingTrie(close: Closing, *args: list):
    return close(*args) if callable(close) else Token(close)


def T(*next: TokenTrie, close: Closing = lambda _: END) -> TokenTrie:
    """
    Produces a prefix tree or trie data structure.
    It is the most primitive data structure to represent a branch in a decision tree.
    It should have at least a head, which is a string key, and the next branches.
    Through the `close` keyword argument, it is possible to define behaviors when the branch ends.
    """
    key = next[0]

    if isinstance(key, int):
        return {key: ClosingTrie(close, next)}

    if not isinstance(key, str):
        return T("", *next, close=close)

    if len(next) == 1:
        return {key: ClosingTrie(close, next)}

    result = {key: value for dict in next[1:] for key, value in Token(dict).items()}

    if close is not None:
        result["@push"] = ClosingTrie(close, next)

    result = {key: result}

    return result


def compile(
    tree: TokenTrie,
    tokenize: Callable[[str], list[int]],
    end: list[TokenTrie] = [],
):
    """
    Tokenize the keys of a tree recursively.
    """
    new_tree = {}

    if "@push" in tree:
        if isinstance(tree["@push"], dict) and "@pop" not in tree["@push"]:
            end.append(tree["@push"])

    for key, value in tree.items():
        tokens = tokenize(key) if isinstance(key, str) else [key]
        current_dict = new_tree

        if key == "@push":
            continue

        if key == "@pop":
            if len(end) > 0:
                current_dict.update(compile(end[-1], tokenize, end[:-1]))
            continue

        for token in tokens:
            if token not in current_dict:
                current_dict[token] = {}

            current_dict = current_dict[token]

        current_dict.update(compile(value, tokenize, end))

    return new_tree
