import time
import torch

class StepTimer:
    """
    A class to help measure time intervals for various named steps.
    If using GPU, uses torch.cuda.Event for more accurate timing.
    Accumulates stats in a dictionary so we can log them later.
    """

    def __init__(self, device='cpu'):
        self.device = device
        self.current_step_name = None
        
        # We'll keep start/end references
        # or a single pair of events for each measurement
        self.start_event = None
        self.end_event   = None

        self.start_time = 0  # CPU fallback

        # A dictionary to store cumulative times
        # e.g. { 'data': 0.0, 'forward': 0.0, 'backward': 0.0, 'model_layer1': 0.0, ... }
        self.stats = {}

        if 'cuda' in str(device):
            # We will create events on the fly in start()
            pass

    def start(self, step_name):
        """
        Begin timing for the named step
        """
        self.current_step_name = step_name
        if 'cuda' in str(self.device):
            # Create new events
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event   = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            self.start_event.record()
        else:
            self.start_time = time.time()

    def stop(self, num_events=1):
        """
        End timing for the current step, accumulate time, and increment event count.
        """
        if self.current_step_name is None:
            return  # No step was started

        elapsed = 0.0
        if 'cuda' in str(self.device):
            self.end_event.record()
            torch.cuda.synchronize()
            elapsed_ms = self.start_event.elapsed_time(self.end_event)
            elapsed = elapsed_ms / 1000.0  # convert ms to seconds
        else:
            elapsed = time.time() - self.start_time

        # accumulate into stats
        if self.current_step_name not in self.stats:
            self.stats[self.current_step_name] = 0.0
        self.stats[self.current_step_name] += elapsed

        # reset current step name
        self.current_step_name = None
        

    def reset_stats(self):
        """
        Clear the stats dictionary
        """
        self.stats.clear()
        self.total_events = 0

    def get_stats(self, reset=False):
        """
        Return a copy of the stats. If reset=True, also clears them.
        """
        out = dict(self.stats)
        if reset:
            self.reset_stats()
        return out