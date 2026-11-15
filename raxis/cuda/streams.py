"""CUDA streams and events, as torch.cuda.Stream and torch.cuda.Event.

A stream is a queue of GPU work: the kernels and copies of one stream run in
the order they were queued, and those of two streams may overlap. Every op
queues its work on the current stream of its device, which is the default
stream until a thread switches with ``with raxis.cuda.stream(s):`` or
``set_stream(s)``. An event marks a point in a stream: other streams (or the
CPU) can wait for it, and two events with timing measure the GPU time between
them.

Memory used across streams needs one more step. The caching allocator reuses
a freed block on the stream it was allocated on, so a tensor that another
stream still reads must be marked with ``tensor.record_stream(s)`` before it
is freed. The usual pattern, a copy on a side stream::

    copy_stream = raxis.cuda.Stream()
    with raxis.cuda.stream(copy_stream):
        x = batch.to("cuda", non_blocking=True)   # batch is pinned
    raxis.cuda.current_stream().wait_stream(copy_stream)
    x.record_stream(raxis.cuda.current_stream())
"""
from raxis import _C


__all__ = ["Stream", "Event", "StreamContext", "stream", "current_stream", "default_stream", "set_stream"]


def _device_index(device) -> int:
    from raxis.cuda import _device_index
    return _device_index(device)


class Stream:
    """A CUDA stream of the device's pool (torch.cuda.Stream).

    ``priority`` is 0 (normal) or negative (higher: its kernels are scheduled
    first when both streams have work ready). The pool has 32 streams per
    priority and device, handed out in turn, so two Stream objects can share
    one underlying stream, as in PyTorch.
    """

    def __init__(self, device=None, priority: int = 0, *, _handle: int | None = None):
        self.device_index = _device_index(device)
        if _handle is None:
            _handle = _C._cuda_stream_from_pool(self.device_index, priority)
        self.cuda_stream = _handle

    @property
    def device(self):
        return _C.device(f"cuda:{self.device_index}")

    def wait_event(self, event: "Event") -> None:
        """Work queued on this stream from now on waits until ``event`` completes (the CPU does not wait)."""
        event.wait(self)

    def wait_stream(self, stream: "Stream") -> None:
        """Work queued on this stream from now on waits for the work already queued on ``stream``."""
        self.wait_event(stream.record_event())

    def record_event(self, event: "Event | None" = None) -> "Event":
        """Records an event (a new one by default) at this point of the stream and returns it."""
        if event is None:
            event = Event()
        event.record(self)
        return event

    def query(self) -> bool:
        """Whether all the work queued on the stream has finished."""
        return _C._cuda_stream_query(self.device_index, self.cuda_stream)

    def synchronize(self) -> None:
        """Waits until all the work queued on the stream has finished."""
        _C._cuda_stream_synchronize(self.device_index, self.cuda_stream)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Stream):
            return NotImplemented
        return self.cuda_stream == other.cuda_stream and self.device_index == other.device_index

    def __hash__(self) -> int:
        return hash((self.cuda_stream, self.device_index))

    def __repr__(self) -> str:
        return f"<raxis.cuda.Stream device={self.device} cuda_stream={self.cuda_stream:#x}>"


class Event:
    """A CUDA event (torch.cuda.Event): a point in a stream to wait for or to time.

    The event is created on first record(), on the device of that stream.
    ``enable_timing`` makes elapsed_time() possible (events without it are
    cheaper); ``blocking`` makes synchronize() sleep instead of spin.
    """

    def __init__(self, enable_timing: bool = False, blocking: bool = False, interprocess: bool = False):
        if interprocess:
            raise NotImplementedError("raxis.cuda.Event does not support interprocess events")
        self.enable_timing = enable_timing
        self.blocking = blocking
        self.cuda_event = 0
        self.device_index = -1

    def _create(self, device_index: int) -> None:
        if self.cuda_event == 0:
            self.cuda_event = _C._cuda_event_create(device_index, self.enable_timing, self.blocking)
            self.device_index = device_index
        elif self.device_index != device_index:
            raise ValueError(
                f"Event was recorded on cuda:{self.device_index} and cannot be recorded on cuda:{device_index}")

    def record(self, stream: Stream | None = None) -> None:
        """Marks the current point of ``stream`` (the current stream by default)."""
        if stream is None:
            stream = current_stream()
        self._create(stream.device_index)
        _C._cuda_event_record(self.device_index, self.cuda_event, stream.cuda_stream)

    def wait(self, stream: Stream | None = None) -> None:
        """Makes the work queued on ``stream`` from now on wait for this event. A no-op before any record()."""
        if stream is None:
            stream = current_stream()
        if self.cuda_event != 0:
            _C._cuda_stream_wait_event(stream.device_index, stream.cuda_stream, self.cuda_event)

    def query(self) -> bool:
        """Whether the work before the recorded point has finished (True if never recorded)."""
        if self.cuda_event == 0:
            return True
        return _C._cuda_event_query(self.device_index, self.cuda_event)

    def synchronize(self) -> None:
        """Waits until the work before the recorded point has finished."""
        if self.cuda_event != 0:
            _C._cuda_event_synchronize(self.device_index, self.cuda_event)

    def elapsed_time(self, end_event: "Event") -> float:
        """Milliseconds of GPU time between this event and ``end_event``, both recorded and completed."""
        if not (self.enable_timing and end_event.enable_timing):
            raise ValueError("Both events must be created with enable_timing=True to measure the time between them")
        if self.cuda_event == 0 or end_event.cuda_event == 0:
            raise ValueError("Both events must be recorded before calculating elapsed time")
        return _C._cuda_event_elapsed_time(self.device_index, self.cuda_event, end_event.cuda_event)

    def __del__(self):
        if getattr(self, "cuda_event", 0) != 0 and _C is not None:
            _C._cuda_event_destroy(self.device_index, self.cuda_event)
            self.cuda_event = 0

    def __repr__(self) -> str:
        if self.cuda_event == 0:
            return "<raxis.cuda.Event uninitialized>"
        return f"<raxis.cuda.Event {self.cuda_event:#x}>"


def current_stream(device=None) -> Stream:
    """The stream this thread queues work on, on ``device`` (the current device by default)."""
    index = _device_index(device)
    return Stream(index, _handle=_C._cuda_current_stream(index))


def default_stream(device=None) -> Stream:
    index = _device_index(device)
    return Stream(index, _handle=_C._cuda_default_stream(index))


def set_stream(stream: Stream | None) -> None:
    """Makes ``stream`` the current stream of its device, and its device the current device (as torch)."""
    if stream is None:
        return
    if _C._cuda_current_device() != stream.device_index:
        _C._cuda_set_device(stream.device_index)
    _C._cuda_set_stream(stream.device_index, stream.cuda_stream)


class StreamContext:
    """Selects ``stream`` inside a with block and restores the previous stream after it (a no-op for None).

    Entering also makes the stream's device the current one, as set_stream;
    leaving restores the stream of both devices and the previous device.
    """

    def __init__(self, stream: Stream | None):
        self.stream = stream
        self.src_prev_stream = None
        self.dst_prev_stream = None

    def __enter__(self):
        if self.stream is None:
            return
        self.src_prev_stream = current_stream()
        if self.src_prev_stream.device_index != self.stream.device_index:
            self.dst_prev_stream = current_stream(self.stream.device_index)
        set_stream(self.stream)

    def __exit__(self, *exc):
        if self.stream is None:
            return
        if self.dst_prev_stream is not None:
            set_stream(self.dst_prev_stream)
        set_stream(self.src_prev_stream)


def stream(stream: Stream | None) -> StreamContext:
    """``with raxis.cuda.stream(s):`` queues the ops inside on ``s``."""
    return StreamContext(stream)


def _record_stream(self, stream: Stream) -> None:
    """Marks this CUDA tensor's memory as used by work queued on ``stream``.

    The caching allocator reuses a freed block only on the stream it was
    allocated on. When a tensor is also used on another stream, call this
    before freeing it: its block then waits until the work queued on
    ``stream`` so far has finished.
    """
    if not self.is_cuda:
        raise RuntimeError("record_stream expects a CUDA tensor")
    self._record_stream(stream.cuda_stream)


_C.Tensor.record_stream = _record_stream
