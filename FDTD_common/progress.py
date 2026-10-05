"""Terminal progress driven by nonblocking CUDA completion queries."""
from time import monotonic, sleep


class GpuProgress:
    """Record bounded completion markers without waiting in the launch loop.

    Events are allocated before stepping. Queries return immediately; no stream,
    event, or device synchronization API is used. finish() polls on the CPU so
    the synchronous run API can safely read results after the last kernel.
    """

    def __init__(self, cuda, steps, enabled=True, desc="FDTD simulation"):
        self.steps = int(steps)
        self.chunk = max(1, (self.steps + 127) // 128)
        self.simulator = bool(cuda.config.ENABLE_CUDASIM)
        self.bar = None
        self.recorded = []
        self.completed = 0
        self.last_poll = monotonic()
        # A final marker is needed even with display disabled, for safe readback.
        count = ((self.steps + self.chunk - 1) // self.chunk if enabled else 1)
        self.events = ([] if self.simulator else
                       [cuda.event(timing=False) for _ in range(count)])
        self.enabled = enabled
        if enabled:
            from tqdm import tqdm
            self.bar = tqdm(total=self.steps, desc=str(desc), unit="step",
                            mininterval=0.2, dynamic_ncols=True)

    def submitted(self, done):
        """Called after kernels for a step are queued; never waits for them."""
        if self.simulator:
            # Simulator launches are synchronous and its events lack query().
            if self.bar is not None:
                self.bar.update(done - self.completed)
            self.completed = done
            return
        if done == self.steps or (self.enabled and done % self.chunk == 0):
            event = self.events[len(self.recorded)]
            event.record()
            self.recorded.append((event, done))
        now = monotonic()
        if now - self.last_poll >= 0.2:
            self.poll()
            self.last_poll = now

    def poll(self):
        """Return immediately if the next completion marker is pending."""
        for event, done in self.recorded:
            if done <= self.completed:
                continue
            if not event.query():
                break
            if self.bar is not None:
                self.bar.update(done - self.completed)
            self.completed = done

    def finish(self):
        while self.completed < self.steps:
            self.poll()
            if self.completed < self.steps:
                sleep(0.01)

    def close(self):
        if self.bar is not None:
            self.bar.close()
