"""CUDA streams and events (raxis.cuda.Stream / Event, Tensor.record_stream),
compared with torch's, and the caching allocator's rules for blocks used on
several streams.

The ordering tests keep a stream busy with a spinning kernel (_sleep), so
that work on another stream that was not told to wait would overtake it:
the result tells whether the two streams were ordered. A test must not
assume that the host returns from every launch while the GPU spins: on
Windows (WDDM) a launch sometimes blocks until the busy stream is done, in
torch as in raxis. Overlap is therefore measured on the GPU, with timing
events."""
import gc

import numpy as np
import pytest
import raxis
import torch


pytestmark = pytest.mark.skipif(not raxis.cuda.is_available(), reason="no CUDA device")

LIBS = [raxis, torch]
SLEEP = 100_000_000   # GPU clock cycles, ~40 ms at 2.5 GHz
N = 1 << 20           # 4 MiB of float32: a block of its own in a 20 MiB segment


def address(t):
    return t.untyped_storage().data_ptr()


@pytest.fixture(autouse=True)
def idle_gpu():
    """Every test starts and ends with the GPU idle and the default stream current."""
    for lib in LIBS:
        lib.cuda.synchronize()
    yield
    for lib in LIBS:
        lib.cuda.set_stream(lib.cuda.default_stream())
        lib.cuda.synchronize()


class TestStreamApi:

    @pytest.mark.parametrize("lib", LIBS)
    def test_default_and_current(self, lib):
        default = lib.cuda.default_stream()
        assert lib.cuda.current_stream() == default
        s = lib.cuda.Stream()
        assert s != default
        assert s.device == lib.device("cuda:0")
        with lib.cuda.stream(s):
            assert lib.cuda.current_stream() == s
            with lib.cuda.stream(None):
                assert lib.cuda.current_stream() == s
        assert lib.cuda.current_stream() == default
        lib.cuda.set_stream(s)
        assert lib.cuda.current_stream() == s
        lib.cuda.set_stream(default)
        assert lib.cuda.current_stream() == default

    def test_hash_and_repr(self):
        s = raxis.cuda.Stream()
        assert raxis.cuda.Stream(_handle=s.cuda_stream) == s
        assert len({s, raxis.cuda.Stream(_handle=s.cuda_stream)}) == 1
        assert repr(s).startswith("<raxis.cuda.Stream device=cuda:0 cuda_stream=0x")

    @pytest.mark.parametrize("lib", LIBS)
    def test_pool_of_32(self, lib):
        # Streams come round robin from a pool of 32 per priority level
        handles = [lib.cuda.Stream().cuda_stream for _ in range(64)]
        assert len(set(handles)) == 32
        assert handles[:32] == handles[32:]
        high = {lib.cuda.Stream(priority=-1).cuda_stream for _ in range(32)}
        assert len(high) == 32 and not high & set(handles)

    @pytest.mark.parametrize("lib", LIBS)
    def test_query_and_synchronize(self, lib):
        s = lib.cuda.Stream()
        with lib.cuda.stream(s):
            lib.cuda._sleep(SLEEP)
        assert not s.query()
        # The default stream does not wait for s: it is a non-blocking stream
        assert lib.cuda.default_stream().query()
        s.synchronize()
        assert s.query()

    def test_without_cuda_tensor(self):
        with pytest.raises(RuntimeError, match="record_stream expects a CUDA tensor"):
            raxis.ones(3).record_stream(raxis.cuda.current_stream())


class TestOrdering:

    @pytest.mark.parametrize("lib", LIBS)
    def test_streams_overlap(self, lib):
        # Work on a side stream that does not wait runs while the default
        # stream spins: it ends long before the spin, on the GPU's clock
        x = lib.zeros(N, device="cuda")
        s = lib.cuda.Stream()
        start, spin_end, fill_end = (lib.cuda.Event(enable_timing=True) for _ in range(3))
        start.record()
        lib.cuda._sleep(SLEEP)
        spin_end.record()
        with lib.cuda.stream(s):
            x.fill_(1)
            fill_end.record()
        lib.cuda.synchronize()
        assert start.elapsed_time(fill_end) < start.elapsed_time(spin_end) / 4

    @pytest.mark.parametrize("lib", LIBS)
    def test_wait_stream(self, lib):
        x = lib.zeros(N, device="cuda")
        s = lib.cuda.Stream()
        s.wait_stream(lib.cuda.current_stream())
        with lib.cuda.stream(s):
            lib.cuda._sleep(SLEEP)
            x.add_(1)
        lib.cuda.current_stream().wait_stream(s)
        y = x * 2
        assert y.min().item() == 2

    @pytest.mark.parametrize("lib", LIBS)
    def test_wait_event(self, lib):
        x = lib.zeros(N, device="cuda")
        s = lib.cuda.Stream()
        with lib.cuda.stream(s):
            lib.cuda._sleep(SLEEP)
            x.add_(1)
            done = lib.cuda.Event()
            done.record()
            lib.cuda._sleep(SLEEP)
            x.add_(1)              # after the event: not waited for
        lib.cuda.current_stream().wait_event(done)
        y = x.clone()
        s.synchronize()
        assert y.min().item() == 1

    @pytest.mark.parametrize("lib", LIBS)
    def test_ops_on_a_side_stream(self, lib):
        # Elementwise, reductions, cuBLAS, sort and a host round trip, all
        # queued on s behind a spin: each must read its inputs after it
        rng = np.random.RandomState(0)
        a_np = rng.randn(256, 256).astype(np.float32)
        a = lib.from_numpy(a_np).cuda()
        s = lib.cuda.Stream()
        s.wait_stream(lib.cuda.current_stream())
        with lib.cuda.stream(s):
            lib.cuda._sleep(SLEEP)
            b = a.exp() + 1
            m = b @ b.t()
            total = m.sum(dim=1)
            srt = total.sort().values
            out = srt.cpu()        # waits for s, the current stream
        b_np = np.exp(a_np.astype(np.float64)) + 1
        np.testing.assert_allclose(out.numpy(), np.sort((b_np @ b_np.T).sum(axis=1)), rtol=1e-4)

    @pytest.mark.parametrize("lib", LIBS)
    def test_copy_on_a_side_stream(self, lib):
        # The usual pattern: the next batch is copied on a side stream while
        # the default stream computes; the result waits for the copy
        batch = lib.arange(N, dtype=lib.float32).pin_memory()
        copy_stream = lib.cuda.Stream()
        lib.cuda._sleep(SLEEP)     # the default stream is busy
        with lib.cuda.stream(copy_stream):
            x = batch.to("cuda", non_blocking=True)
        # The copy does not wait for the default stream's spin
        copy_stream.synchronize()
        assert not lib.cuda.default_stream().query()
        lib.cuda.current_stream().wait_stream(copy_stream)
        x.record_stream(lib.cuda.current_stream())
        assert (x * 2)[-1].item() == 2 * (N - 1)


class TestEvent:

    @pytest.mark.parametrize("lib", LIBS)
    def test_elapsed_time(self, lib):
        start = lib.cuda.Event(enable_timing=True)
        end = lib.cuda.Event(enable_timing=True)
        start.record()
        lib.cuda._sleep(SLEEP)
        end.record()
        assert not end.query()
        end.synchronize()
        assert end.query()
        assert 1 < start.elapsed_time(end) < 2000

    def test_not_recorded(self):
        e = raxis.cuda.Event()
        assert e.query()
        e.synchronize()
        e.wait()
        assert repr(e) == "<raxis.cuda.Event uninitialized>"

    def test_errors(self):
        a, b = raxis.cuda.Event(), raxis.cuda.Event()
        a.record()
        b.record()
        with pytest.raises(ValueError, match="enable_timing=True"):
            a.elapsed_time(b)
        t = raxis.cuda.Event(enable_timing=True)
        with pytest.raises(ValueError, match="must be recorded"):
            t.elapsed_time(raxis.cuda.Event(enable_timing=True))
        with pytest.raises(NotImplementedError):
            raxis.cuda.Event(interprocess=True)

    @pytest.mark.parametrize("lib", LIBS)
    def test_record_event(self, lib):
        s = lib.cuda.Stream()
        with lib.cuda.stream(s):
            lib.cuda._sleep(SLEEP)
        e = s.record_event()
        assert not e.query()
        e.synchronize()
        assert e.query() and s.query()


class TestCachingAllocator:
    """A freed block is reused only on its own stream, and record_stream
    holds it back until the other streams that used it are done."""

    @pytest.mark.parametrize("lib", LIBS)
    def test_reuse_on_its_own_stream(self, lib):
        s = lib.cuda.Stream()
        with lib.cuda.stream(s):
            x = lib.empty(N, device="cuda")
        a = address(x)
        del x
        # The default stream gets another block ...
        y = lib.empty(N, device="cuda")
        assert address(y) != a
        # ... and s gets its own back
        with lib.cuda.stream(s):
            z = lib.empty(N, device="cuda")
        assert address(z) == a

    @pytest.mark.parametrize("lib", LIBS)
    def test_record_stream_delays_reuse(self, lib):
        x = lib.ones(N, device="cuda")
        a = address(x)
        s = lib.cuda.Stream()
        s.wait_stream(lib.cuda.current_stream())
        with lib.cuda.stream(s):
            lib.cuda._sleep(SLEEP)
            y = x * 2
        x.record_stream(s)
        del x
        # s still reads the block: a new tensor gets another one
        z = lib.empty(N, device="cuda")
        assert address(z) != a
        s.synchronize()
        # Once s is done, the block is back in the pool
        w = lib.empty(N, device="cuda")
        assert address(w) == a
        assert y.min().item() == 2

    @pytest.mark.parametrize("lib", LIBS)
    def test_record_stream_protects_the_data(self, lib):
        x = lib.ones(N, device="cuda")
        s = lib.cuda.Stream()
        s.wait_stream(lib.cuda.current_stream())
        with lib.cuda.stream(s):
            lib.cuda._sleep(SLEEP)
            y = x + 0
        x.record_stream(s)
        del x
        z = lib.full((N,), 5.0, device="cuda")
        s.synchronize()
        assert y.max().item() == 1
        del z

    @pytest.mark.parametrize("lib", LIBS)
    def test_without_record_stream_the_block_is_overwritten(self, lib):
        # The race record_stream prevents: the freed block goes back to the
        # default stream's pool at once, and the fill runs before s reads it
        x = lib.ones(N, device="cuda")
        a = address(x)
        s = lib.cuda.Stream()
        s.wait_stream(lib.cuda.current_stream())
        with lib.cuda.stream(s):
            lib.cuda._sleep(SLEEP)
            y = x + 0
        del x
        z = lib.full((N,), 5.0, device="cuda")
        assert address(z) == a
        s.synchronize()
        assert y.max().item() == 5

    def test_split_blocks_wait_before_merging(self):
        # Two small tensors in one 2 MiB segment; the one used on s is freed
        # first and waits for its event: its neighbour must not merge it
        gc.collect()
        raxis.cuda.empty_cache()
        x = raxis.ones(1000, device="cuda")
        y = raxis.ones(1000, device="cuda")
        s = raxis.cuda.Stream()
        s.wait_stream(raxis.cuda.current_stream())
        with raxis.cuda.stream(s):
            raxis.cuda._sleep(SLEEP)
            r = x * 3
        x.record_stream(s)
        a = address(x)
        del x, y
        z = raxis.full((1000,), 7.0, device="cuda")
        assert address(z) != a
        s.synchronize()
        assert r.max().item() == 3
        del z, r
        raxis.cuda.empty_cache()

    def test_empty_cache_waits_for_pending_blocks(self):
        gc.collect()
        raxis.cuda.empty_cache()
        before = raxis.cuda.memory_reserved()
        x = raxis.ones(N, device="cuda")
        s = raxis.cuda.Stream()
        s.wait_stream(raxis.cuda.current_stream())
        with raxis.cuda.stream(s):
            raxis.cuda._sleep(SLEEP)
            y = x.sum()
        x.record_stream(s)
        del x
        raxis.cuda.empty_cache()
        # x's 20 MiB segment is released; y, a small block, keeps a 2 MiB one
        assert raxis.cuda.memory_reserved() - before <= 2 * 1024 * 1024
        assert y.item() == N

    def test_memory_stays_flat(self):
        # A copy stream feeding the default stream. While the host runs
        # ahead of the GPU, the blocks freed with record_stream wait for
        # their events, so the cache grows with the steps in flight (torch's
        # too); a round of steps with the GPU caught up after it must find
        # them all back in the pool, and none may stay counted as allocated
        gc.collect()
        batch = raxis.ones(N).pin_memory()
        copy_stream = raxis.cuda.Stream()
        w = raxis.ones(N, device="cuda")
        base = raxis.cuda.memory_allocated()
        reserved = []
        for _ in range(4):
            for _ in range(10):
                with raxis.cuda.stream(copy_stream):
                    x = batch.to("cuda", non_blocking=True)
                raxis.cuda.current_stream().wait_stream(copy_stream)
                x.record_stream(raxis.cuda.current_stream())
                w = w * 0.5 + x
                del x
            raxis.cuda.synchronize()
            reserved.append(raxis.cuda.memory_reserved())
        assert reserved[-1] == reserved[-2]
        assert raxis.cuda.memory_allocated() == base
        np.testing.assert_allclose(w[:3].cpu().numpy(), [2.0] * 3, rtol=1e-6)
