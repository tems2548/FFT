"""Wire protocol tests: CRC-16/CCITT-FALSE and SerialReader._parse().

_parse() is tested directly against a bytearray buffer, bypassing
SerialReader.__init__ entirely (which would otherwise require a real
serial port) via SerialReader.__new__ -- a bare instance with just the
attributes _parse() actually touches.
"""
import queue
import struct

import pytest

import FFT


def make_reader():
    """A SerialReader instance with none of __init__'s serial.Serial()
    side effect -- just the state _parse() reads and writes."""
    reader = FFT.SerialReader.__new__(FFT.SerialReader)
    reader.sample_queue = queue.Queue()
    reader.sample_rate = None
    reader.temp_c = None
    reader.packets_ok = 0
    reader.packets_bad = 0
    return reader


def build_meta_packet(rate, temp_c):
    payload = struct.pack("<If", rate, temp_c)
    crc = FFT.crc16_ccitt(payload)
    return b"META" + payload + struct.pack("<H", crc)


def build_data_packet(samples_mv):
    payload = struct.pack("<H", len(samples_mv)) + struct.pack(f"<{len(samples_mv)}h", *samples_mv)
    crc = FFT.crc16_ccitt(payload)
    return b"DATA" + payload + struct.pack("<H", crc)


class TestCrc16:
    def test_standard_test_vector(self):
        # CRC-16/CCITT-FALSE's canonical check value, from the CRC RevEng
        # catalogue -- the one fixed point every independent implementation
        # (this one, and main.c's crc16_ccitt_update()) must agree on.
        assert FFT.crc16_ccitt(b"123456789") == 0x29B1

    def test_empty_input_returns_seed(self):
        assert FFT.crc16_ccitt(b"") == 0xFFFF

    def test_chained_crc_matches_single_call(self):
        # main.c computes the META and DATA CRCs in one call each, but
        # crc16_ccitt() accepting a running `crc` argument (for symmetry
        # with the C implementation's update-style API) needs the two
        # forms to actually agree.
        data = bytes(range(50))
        whole = FFT.crc16_ccitt(data)
        chained = FFT.crc16_ccitt(data[25:], FFT.crc16_ccitt(data[:25]))
        assert whole == chained

    def test_single_bit_corruption_changes_crc(self):
        data = b"hello world, this is a test payload"
        original = FFT.crc16_ccitt(data)
        corrupted = bytearray(data)
        corrupted[10] ^= 0x01
        assert FFT.crc16_ccitt(bytes(corrupted)) != original


class TestParseMeta:
    def test_valid_meta_updates_rate_and_temp(self):
        reader = make_reader()
        buf = bytearray(build_meta_packet(80000, 42.5))
        reader._parse(buf)
        assert reader.sample_rate == 80000
        assert reader.temp_c == pytest.approx(42.5, abs=1e-4)
        assert reader.packets_ok == 1
        assert reader.packets_bad == 0
        assert len(buf) == 0  # fully consumed

    def test_meta_with_nan_temp_survives_round_trip(self):
        # Firmware sends NaN when its die temperature sensor is
        # unavailable -- must not be treated as a malformed packet.
        reader = make_reader()
        buf = bytearray(build_meta_packet(80000, float("nan")))
        reader._parse(buf)
        assert reader.packets_ok == 1
        assert reader.sample_rate == 80000
        import math
        assert math.isnan(reader.temp_c)

    def test_bad_crc_is_rejected(self):
        reader = make_reader()
        packet = bytearray(build_meta_packet(80000, 42.5))
        packet[-1] ^= 0xFF  # corrupt the CRC's high byte
        reader._parse(packet)
        assert reader.sample_rate is None
        assert reader.packets_ok == 0
        assert reader.packets_bad == 1

    def test_partial_packet_is_left_for_next_read(self):
        reader = make_reader()
        full = build_meta_packet(80000, 42.5)
        buf = bytearray(full[:10])  # short by 4 bytes
        reader._parse(buf)
        assert reader.sample_rate is None
        assert reader.packets_ok == 0
        assert reader.packets_bad == 0
        assert bytes(buf) == full[:10]  # untouched, waiting for the rest


class TestParseData:
    def test_valid_data_converts_millivolts_to_volts(self):
        reader = make_reader()
        buf = bytearray(build_data_packet([1000, -500, 0, 3300]))
        reader._parse(buf)
        assert reader.packets_ok == 1
        drained = []
        while True:
            try:
                drained.append(reader.sample_queue.get_nowait())
            except queue.Empty:
                break
        assert drained == pytest.approx([1.0, -0.5, 0.0, 3.3])

    def test_bad_crc_data_packet_pushes_nothing(self):
        reader = make_reader()
        packet = bytearray(build_data_packet([1000, 2000]))
        packet[-1] ^= 0xFF
        reader._parse(packet)
        assert reader.packets_bad == 1
        assert reader.sample_queue.empty()

    def test_empty_data_packet(self):
        reader = make_reader()
        buf = bytearray(build_data_packet([]))
        reader._parse(buf)
        assert reader.packets_ok == 1
        assert reader.sample_queue.empty()


class TestResyncAndFraming:
    def test_resyncs_past_leading_garbage(self):
        # Simulates stray ESP_LOG text sharing the UART landing in front
        # of a valid packet -- the parser must find the magic word and
        # discard everything before it, not choke on the garbage.
        reader = make_reader()
        garbage = b"I (1234) wifi: some unrelated log line\r\n"
        buf = bytearray(garbage + build_meta_packet(80000, 25.0))
        reader._parse(buf)
        assert reader.packets_ok == 1
        assert reader.sample_rate == 80000

    def test_bad_packet_does_not_break_sync_on_next_valid_one(self):
        reader = make_reader()
        bad = bytearray(build_data_packet([111, 222]))
        bad[-1] ^= 0xFF
        buf = bytearray(bytes(bad) + build_data_packet([333, 444]))
        reader._parse(buf)
        assert reader.packets_bad == 1
        assert reader.packets_ok == 1
        values = []
        while True:
            try:
                values.append(reader.sample_queue.get_nowait())
            except queue.Empty:
                break
        assert values == pytest.approx([0.333, 0.444])

    def test_short_trailing_fragment_without_magic_word_is_kept_short(self):
        # Fewer than 4 bytes with no magic word: can't yet tell if a magic
        # word is split across reads, so only a 3-byte tail is kept
        # (matches _parse()'s own del buf[:max(0, len(buf)-3)] rule).
        reader = make_reader()
        buf = bytearray(b"xxxxxxMET")  # "MET" -- possible start of "META"
        reader._parse(buf)
        assert bytes(buf) == b"MET"


class TestParseGoertzelTargets:
    @pytest.mark.parametrize(
        "text,fs,expected",
        [
            ("50, 60, 120", 1000.0, [50.0, 60.0, 120.0]),
            ("120, 50, 60", 1000.0, [50.0, 60.0, 120.0]),  # sorted
            ("50,60,", 1000.0, [50.0, 60.0]),  # trailing comma
            ("", 1000.0, []),
            ("not-a-number, 50", 1000.0, [50.0]),  # bad token dropped
            ("-10, 0, 499, 500, 5000", 1000.0, [499.0]),  # out-of-range dropped (Nyquist=500, exclusive)
            ("  50.5  ,  60.25  ", 1000.0, [50.5, 60.25]),  # whitespace tolerated
        ],
    )
    def test_parses_and_filters(self, text, fs, expected):
        assert FFT.parse_goertzel_targets(text, fs) == pytest.approx(expected)
