import weakref
from typing import Any


# From: torch/utils/hooks.py
class RemovableHandle:
    """Returned by the register_*_hook methods: remove() unregisters the hook.

    Holds the hooks dict weakly, so a handle kept around does not keep a
    deleted module alive.
    """

    next_id: int = 0

    def __init__(self, hooks_dict: dict) -> None:
        self.hooks_dict_ref = weakref.ref(hooks_dict)
        self.id = RemovableHandle.next_id
        RemovableHandle.next_id += 1

    def remove(self) -> None:
        hooks_dict = self.hooks_dict_ref()
        if hooks_dict is not None and self.id in hooks_dict:
            del hooks_dict[self.id]

    def __enter__(self) -> "RemovableHandle":
        return self

    def __exit__(self, *args: Any) -> None:
        self.remove()
