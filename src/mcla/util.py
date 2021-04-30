from typing import Set, List


def jaccard(set1: Set[int], set2: Set[int]) -> float:
    union = set(list(set1) + list(set2))
    intersection = []
    for item in set1:
        if item in set2:
            intersection.append(item)
    return len(intersection) / len(union)


def jaccard_single(item: int, list: List[int]) -> float:
    occurrences = 0
    for item0 in list:
        if item0 == item:
            occurrences += 1
    return occurrences / len(list)
