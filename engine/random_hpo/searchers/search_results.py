from typing import Dict, List


class _SearchResults:
    def __init__(self) -> None:
        self.__results: Dict[str, List] = {}

    def add(self, name: str, value: any) -> None:
        if self.__results.get(name) is None:
            self.__results[name] = [value]
        else:
            self.__results[name].append(value)

    def get_results(self) -> Dict[str, any]:
        return self.__results
