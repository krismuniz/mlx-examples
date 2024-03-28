from .base import T, TokenTrie


def Statement(*next: TokenTrie):
    return T(*next, close=";")


def OneOf(*next: TokenTrie):
    return T(*next)


def MembersOf(key: str, *next: TokenTrie):
    return T(key, *next)


def Property(key: str, *next: TokenTrie):
    return T("." + key, *next)


def ComputedProperty(key: str, *next: TokenTrie):
    return T(
        "[" + key,
        *next,
        close="]",
    )


def CallOf(key: str, *next: TokenTrie):
    return T(key + "(", *next, close=")")


def StringLiteral(value: str):
    return T('"', value, close='"')
