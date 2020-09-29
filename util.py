from typing import Set, Tuple


def set2str(s: Set[int]) -> str:
    return "#".join([str(i) for i in sorted(list(s))])

def pair2str(p: Tuple[int, int]) -> str:
    return f"{p[0]}#{p[1]}"