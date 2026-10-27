import threading
import unittest

from sensoranalyzer.buffer import RingBuffer
from sensoranalyzer.clock import ReplayClock


class RingBufferTest(unittest.TestCase):
    def test_overflow_drops_oldest(self):
        buf = RingBuffer(3)
        for i in range(5):
            buf.push(i)
        self.assertEqual(buf.dropped, 2)
        self.assertEqual([buf.pop(0), buf.pop(0), buf.pop(0)], [2, 3, 4])

    def test_close_drains_then_none(self):
        buf = RingBuffer(4)
        buf.push(1)
        buf.close()
        buf.push(2)  # после закрытия не принимается
        self.assertEqual(buf.pop(0), 1)
        self.assertIsNone(buf.pop(0))

    def test_threads(self):
        buf = RingBuffer(10_000)
        n = 5000

        def writer():
            for i in range(n):
                buf.push(i)
            buf.close()

        threading.Thread(target=writer).start()
        got = []
        while (item := buf.pop(1.0)) is not None:
            got.append(item)
        self.assertEqual(got, list(range(n)))


class BlockingPushTest(unittest.TestCase):
    def test_writer_waits_instead_of_dropping(self):
        buf = RingBuffer(8)
        n = 2000

        def writer():
            for i in range(n):
                buf.push(i, block=True)
            buf.close()

        threading.Thread(target=writer).start()
        got = []
        while (item := buf.pop(1.0)) is not None:
            got.append(item)
        self.assertEqual(got, list(range(n)))
        self.assertEqual(buf.dropped, 0)
        self.assertLessEqual(buf.max_fill, 8)


class FakeTime:
    def __init__(self):
        self.now = 100.0
        self.sleeps = []

    def sleep(self, d):
        self.sleeps.append(d)
        self.now += d


class ReplayClockTest(unittest.TestCase):
    def test_paces_by_sample_time(self):
        ft = FakeTime()
        clock = ReplayClock(2.0, now=lambda: ft.now, sleep=ft.sleep)
        clock.wait_until(10.0)  # опорная точка
        clock.wait_until(11.0)  # секунда записи при speed=2 - полсекунды
        self.assertAlmostEqual(ft.sleeps[-1], 0.5)

    def test_late_processing_does_not_wait(self):
        ft = FakeTime()
        clock = ReplayClock(1.0, now=lambda: ft.now, sleep=ft.sleep)
        clock.wait_until(0.0)
        ft.now += 5.0
        clock.wait_until(1.0)
        self.assertEqual(ft.sleeps, [])

    def test_speed_zero_never_waits(self):
        ft = FakeTime()
        clock = ReplayClock(0, now=lambda: ft.now, sleep=ft.sleep)
        for t in range(10):
            clock.wait_until(float(t))
        self.assertEqual(ft.sleeps, [])


if __name__ == "__main__":
    unittest.main()
