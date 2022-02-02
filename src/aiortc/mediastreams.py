import asyncio
import fractions
import time
import uuid
from abc import ABCMeta, abstractmethod
from typing import Tuple, List

from av import AudioFrame, VideoFrame
from av.frame import Frame
from pyee import AsyncIOEventEmitter

AUDIO_PTIME = 0.020  # 20ms audio packetization
VIDEO_CLOCK_RATE = 90000
VIDEO_PTIME = 1 / 30  # 30fps
VIDEO_TIME_BASE = fractions.Fraction(1, VIDEO_CLOCK_RATE)


def convert_timebase(
    pts: int, from_base: fractions.Fraction, to_base: fractions.Fraction
) -> int:
    if from_base != to_base:
        pts = int(pts * from_base / to_base)
    return pts


class MediaStreamError(Exception):
    pass


class MediaStreamTrack(AsyncIOEventEmitter, metaclass=ABCMeta):
    """
    A single media track within a stream.
    """

    kind = "unknown"
    
    def __init__(self) -> None:
        super().__init__()
        self.__ended = False
        self._id = str(uuid.uuid4())
        self._recv_encoded_mode = False

    @property
    def id(self) -> str:
        """
        An automatically generated globally unique ID.
        """
        return self._id

    @property
    def readyState(self) -> str:
        return "ended" if self.__ended else "live"

    @property
    def recv_encoded_mode(self) -> bool:
        """
        The recv mode for data to send across the RTP channel.
        When True, the RTCRtpSender will await the track 'recv_encoded' vice 'recv' for pre-encoded payloads & timestamp.
        """
        return self._recv_encoded_mode

    @recv_encoded_mode.setter
    def recv_encoded_mode(self, mode: bool):
        self._recv_encoded_mode = mode

    @abstractmethod
    async def recv(self) -> Frame:
        """
        Receive the next :class:`~av.audio.frame.AudioFrame` or :class:`~av.video.frame.VideoFrame`.
        """

    @abstractmethod
    async def recv_encoded(self) -> Tuple[List[bytes], int]:
        """
        Receive the next set of pre-encoded payloads & timestamp
        """

    def stop(self) -> None:
        if not self.__ended:
            self.__ended = True
            self.emit("ended")

            # no more events will be emitted, so remove all event listeners
            # to facilitate garbage collection.
            self.remove_all_listeners()


class AudioStreamTrack(MediaStreamTrack):
    """
    A dummy audio track which reads silence.
    """

    kind = "audio"

    _start: float
    _timestamp: int

    async def recv(self) -> Frame:
        """
        Receive the next :class:`~av.audio.frame.AudioFrame`.

        The base implementation just reads silence, subclass
        :class:`AudioStreamTrack` to provide a useful implementation.
        """
        if self.readyState != "live":
            raise MediaStreamError

        sample_rate = 8000
        samples = int(AUDIO_PTIME * sample_rate)

        if hasattr(self, "_timestamp"):
            self._timestamp += samples
            wait = self._start + (self._timestamp / sample_rate) - time.time()
            await asyncio.sleep(wait)
        else:
            self._start = time.time()
            self._timestamp = 0

        frame = AudioFrame(format="s16", layout="mono", samples=samples)
        for p in frame.planes:
            p.update(bytes(p.buffer_size))
        frame.pts = self._timestamp
        frame.sample_rate = sample_rate
        frame.time_base = fractions.Fraction(1, sample_rate)
        return frame

    async def recv_encoded(self) -> Tuple[List[bytes], int]:
        """
        Receive the next set of pre-encoded payloads & timestamp from a previous encoder.encode call.

        The base implementation is stubbed.
        subclass :class:`AudioStreamTrack` to provide a useful implementation.
        """
        return [], 0

class VideoStreamTrack(MediaStreamTrack):
    """
    A dummy video track which reads green frames.
    """

    kind = "video"

    _start: float
    _timestamp: int

    async def next_timestamp(self) -> Tuple[int, fractions.Fraction]:
        if self.readyState != "live":
            raise MediaStreamError

        if hasattr(self, "_timestamp"):
            self._timestamp += int(VIDEO_PTIME * VIDEO_CLOCK_RATE)
            wait = self._start + (self._timestamp / VIDEO_CLOCK_RATE) - time.time()
            await asyncio.sleep(wait)
        else:
            self._start = time.time()
            self._timestamp = 0
        return self._timestamp, VIDEO_TIME_BASE

    async def recv(self) -> Frame:
        """
        Receive the next :class:`~av.video.frame.VideoFrame`.

        The base implementation just reads a 640x480 green frame at 30fps,
        subclass :class:`VideoStreamTrack` to provide a useful implementation.
        """
        pts, time_base = await self.next_timestamp()

        frame = VideoFrame(width=640, height=480)
        for p in frame.planes:
            p.update(bytes(p.buffer_size))
        frame.pts = pts
        frame.time_base = time_base
        return frame

    async def recv_encoded(self) -> Tuple[List[bytes], int]:
        """
        Receive the next set of pre-encoded payloads & timestamp from a previous encoder.encode call.

        The base implementation is stubbed.
        subclass :class:`VideoStreamTrack` to provide a useful implementation.
        """
        return [], 0
