import unittest

from sensoranalyzer.protocol import (Config, FrameParser, Sample, StreamTime, Tick, TIME_MASK, crc16, decode,
                                     encode)

SAMPLE = Sample(7, 123456789, 1, (5, -249, 4286), (3, -12, -8))
TICK = Tick(3, 123460000, 5, (0.01, -0.02, 0.0, 0.9997), (6.5, -12.25, -6.0), 0.0095)
CONFIG = Config(1, 50_000_000, 475.0, 25.0, 100.0, 0.00390625, 50.0, 2.0, 2.25, (-4086, 4117, -4159, 4057, -4050, 4211))


class CrcTest(unittest.TestCase):
    def test_check_value(self):
        # Контрольное значение CRC-16/CCITT-FALSE: тот же алгоритм, что в прошивке
        self.assertEqual(crc16(b"123456789"), 0x29B1)


class EncodeDecodeTest(unittest.TestCase):
    def test_sizes_match_firmware(self):
        self.assertEqual(len(encode(SAMPLE)), 4 + 19 + 2)
        self.assertEqual(len(encode(TICK)), 4 + 39 + 2)
        self.assertEqual(len(encode(CONFIG)), 4 + 45 + 2)

    def test_roundtrip(self):
        for frame in (SAMPLE, CONFIG):
            raw = encode(frame)
            self.assertEqual(decode(raw[2], raw[4:-2]), frame)
        tick = decode(TYPE := encode(TICK)[2], encode(TICK)[4:-2])
        self.assertEqual(TYPE, 2)
        for a, b in zip(tick.quat + tick.gyro_shift + (tick.gain,), TICK.quat + TICK.gyro_shift + (TICK.gain,)):
            self.assertAlmostEqual(a, b, places=6)  # float32


class ConfigV2Test(unittest.TestCase):
    def test_roundtrip_and_size(self):
        from dataclasses import replace
        v2 = replace(CONFIG, version=2, acc_k=0.5, gyr_k=0.75, gyro_scale=16.5,
                     filter_state=tuple(float(i) for i in range(12)), gyro_shift=(1.0, -2.0, 3.5))
        raw = encode(v2)
        self.assertEqual(len(raw), 4 + 117 + 2)
        back = FrameParser().feed(raw)[0]
        self.assertEqual(back, v2)
        self.assertEqual(back.filter_state, v2.filter_state)
        self.assertEqual(back.gyro_shift, v2.gyro_shift)

    def test_state_not_part_of_settings(self):
        from dataclasses import replace
        a = replace(CONFIG, filter_state=(0.0,) * 12)
        b = replace(CONFIG, filter_state=(1.0,) * 12)
        self.assertEqual(a, b)


class ParserTest(unittest.TestCase):
    def test_byte_by_byte(self):
        stream = encode(SAMPLE) + encode(CONFIG) + encode(SAMPLE)
        parser = FrameParser()
        frames = [f for b in stream for f in parser.feed(bytes([b]))]
        self.assertEqual(frames, [SAMPLE, CONFIG, SAMPLE])
        self.assertEqual(parser.skipped_bytes, 0)

    def test_garbage_and_resync(self):
        good = encode(SAMPLE)
        broken = bytearray(encode(SAMPLE))
        broken[10] ^= 0xFF
        parser = FrameParser()
        frames = parser.feed(b"\x00\xA5\x13\xA5" + bytes(broken) + good + b"\xA5")
        self.assertEqual(frames, [SAMPLE])
        self.assertEqual(parser.crc_errors, 1)
        # Хвост 0xA5 - возможное начало следующего кадра, он не теряется
        self.assertEqual(parser.feed(good[1:]), [SAMPLE])

    def test_split_frame(self):
        raw = encode(CONFIG)
        parser = FrameParser()
        self.assertEqual(parser.feed(raw[:20]), [])
        self.assertEqual(parser.feed(raw[20:]), [CONFIG])


class StreamTimeTest(unittest.TestCase):
    def test_wraparound(self):
        clock = StreamTime(1000)
        self.assertEqual(clock.seconds(TIME_MASK - 499), 0.0)
        self.assertAlmostEqual(clock.seconds(TIME_MASK), 0.499)
        self.assertAlmostEqual(clock.seconds(500), 1.0)  # через переполнение


if __name__ == "__main__":
    unittest.main()
