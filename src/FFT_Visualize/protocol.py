"""Wire protocol for the ESP32 <-> FFT.py serial link: CRC-16/CCITT framing
and the background SerialReader thread that decodes it into voltage
samples. Mirrors main.c's packet format bit-for-bit (see crc16_ccitt's
docstring).
"""
import queue
import struct
import threading

try:
    import serial
except ImportError:
    serial = None


MAGIC_META = b"META"
MAGIC_DATA = b"DATA"


def crc16_ccitt(data, crc=0xFFFF):
    """CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF). Must match main.c's
    crc16_ccitt_update() bit-for-bit -- verified against the standard test
    vector (CRC of b"123456789" == 0x29B1) when this was written."""
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


class SerialReader:
    """Reads main.c's framed ADC stream off a serial port in a background
    thread and feeds decoded voltage samples into a queue.

    Packets are framed with an ASCII magic word (META/DATA) so the parser
    can resync past any ESP_LOG text that lands in the same UART stream.
    """

    def __init__(self, port, baud):
        if serial is None:
            raise RuntimeError("--serial requires pyserial: pip install pyserial")
        self.ser = serial.Serial(port, baud, timeout=0.5)
        self.sample_queue = queue.Queue()
        self.sample_rate = None
        self.temp_c = None  # ESP32-S3 die temperature; NaN if the board has no working sensor
        self.packets_ok = 0
        self.packets_bad = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=1)
        self.ser.close()

    def _run(self):
        buf = bytearray()
        while not self._stop.is_set():
            chunk = self.ser.read(4096)
            if chunk:
                buf.extend(chunk)
                self._parse(buf)

    def _parse(self, buf):
        """Consumes complete packets from buf in place, leaving any
        trailing partial packet for the next read."""
        while True:
            idx_meta = buf.find(MAGIC_META)
            idx_data = buf.find(MAGIC_DATA)
            candidates = [i for i in (idx_meta, idx_data) if i != -1]
            if not candidates:
                # Keep a short tail in case a magic word is split across reads.
                del buf[: max(0, len(buf) - 3)]
                return
            idx = min(candidates)
            if idx > 0:
                del buf[:idx]

            if buf.startswith(MAGIC_META):
                if len(buf) < 14:
                    return
                payload = bytes(buf[4:12])  # rate(4) + temp_c float32(4)
                (crc_received,) = struct.unpack_from("<H", buf, 12)
                if crc16_ccitt(payload) == crc_received:
                    self.sample_rate, self.temp_c = struct.unpack_from("<If", payload, 0)
                    self.packets_ok += 1
                else:
                    self.packets_bad += 1
                del buf[:14]
            else:  # MAGIC_DATA
                if len(buf) < 6:
                    return
                (count,) = struct.unpack_from("<H", buf, 4)
                needed = 8 + count * 2  # magic(4) + count(2) + samples + crc(2)
                if len(buf) < needed:
                    return
                payload = bytes(buf[4 : 6 + count * 2])  # count field + samples
                (crc_received,) = struct.unpack_from("<H", buf, 6 + count * 2)
                if crc16_ccitt(payload) == crc_received:
                    samples_mv = struct.unpack_from(f"<{count}h", payload, 2)
                    for mv in samples_mv:
                        self.sample_queue.put(mv / 1000.0)  # convert mV to V
                    self.packets_ok += 1
                else:
                    self.packets_bad += 1
                del buf[:needed]


CONNECT_TIMEOUT_S = 6.0  # ESP32 reboots on port-open + runs a 1s rate
                          # measurement before its first META packet.
