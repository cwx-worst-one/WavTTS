from typing import Any, List


class TokenGeneratorObserver:
    def on_generate_finished(self, result: Any) -> None:
        pass

    def on_sample_finished(self, result: Any) -> None:
        pass

    def on_reset(self) -> None:
        pass


class ResultRecordObserver(TokenGeneratorObserver):
    """A simple example observer that record sample results into a list
    """
    def on_sample_finished(self, next_index: int) -> None:
        self.all_ids.append(next_index)
    
    def on_reset(self) -> None:
        self.all_ids: List[int] = []