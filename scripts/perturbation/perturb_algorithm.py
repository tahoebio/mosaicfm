# Copyright (C) Vevo Therapeutics 2024. All rights reserved.
from composer import Algorithm, Event, Logger, State


class SetFindUnusedParameters(Algorithm):
    def match(self, state: State, event: Event, logger: Logger) -> bool:
        # This algorithm can run at any event or you can specify a specific event.
        return event == Event.AFTER_LOAD

    def apply(self, state: State, event: Event, logger: Logger):
        # No specific changes needed in state, this is just to set find_unused_parameters
        pass

    @property
    def find_unused_parameters(self) -> bool:
        return True
