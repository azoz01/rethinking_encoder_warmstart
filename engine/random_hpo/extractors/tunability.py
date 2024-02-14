import numpy as np

from typing import List, Dict, Callable


class TunabilityExtractor:
    def __init__(
        self,
        search_results: List[Dict[str, List]],
        lowest_best: bool,
        aggregation_func: Callable[..., float] = np.mean,
    ) -> None:
        self.__check_results(search_results)
        self.search_results = search_results
        self.lowest_best = lowest_best
        self.aggregation_func = aggregation_func

        self.default_idx = None
        self.default_hpo = None

    def __check_results(
        self,
        search_results: List[Dict[str, List]],
    ) -> None:
        first_seach_results = search_results[0]
        keys = list(first_seach_results.keys())
        lens = list(map(lambda x: len(first_seach_results[x]), keys))

        for key, len_ in zip(keys, lens):
            for search_result in search_results:
                assert len(search_result[key]) == len_

    def extract_default_hpo(self) -> Dict[str, any]:
        n_iter = len(self.search_results[0]["hpo"])
        n_datasets = len(self.search_results)
        hpos = self.search_results[0]["hpo"]

        scores_arr = np.array([])
        for row in self.search_results:
            scores_arr = np.concatenate([scores_arr, row["mean_score"]])

        scores_arr = scores_arr.reshape([n_datasets, n_iter])

        scores_aggregated = np.apply_along_axis(
            self.aggregation_func, arr=scores_arr, axis=0
        )

        if self.lowest_best:
            best_idx = np.argmin(scores_aggregated)
        else:
            best_idx = np.argmax(scores_aggregated)

        self.default_idx = best_idx
        self.default_hpo = hpos[best_idx]
        return hpos[best_idx]

    def extract_gains(self):
        if self.default_idx is None or self.default_hpo is None:
            raise ValueError(
                "Default hpo was not set yet. Please run extract_default_hpo() first."
            )

        default_scores = np.array(
            list(
                map(
                    lambda results: results["mean_score"][self.default_idx],
                    self.search_results,
                )
            )
        )
        if self.lowest_best:
            best_scores = np.array(
                list(
                    map(
                        lambda results: np.min(results["mean_score"]),
                        self.search_results,
                    )
                )
            )
            gains = default_scores - best_scores
        else:
            best_scores = np.array(
                list(
                    map(
                        lambda results: np.max(results["mean_score"]),
                        self.search_results,
                    )
                )
            )
            gains = best_scores - default_scores

        return gains
