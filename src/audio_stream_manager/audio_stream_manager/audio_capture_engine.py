import threading
import time
from typing import Callable, Optional

import numpy as np

from audio_stream_manager.utils.audio_models import ActiveDevice
from audio_stream_manager.utils.audio_utils import compute_rms, resample_audio
from audio_stream_manager.utils.sound_device_manager import SoundDeviceManager

# ---------------------------------------------------------------------------
# DeviceWatchdog — owns disconnection + recovery threads
# ---------------------------------------------------------------------------


class DeviceWatchdog:
    """
    Manages device disconnection detection and primary-device recovery.

    Runs two background daemon threads and triggers callbacks into the engine
    so neither the engine nor the ROS2 node needs to manage threading for these concerns.
    """

    def __init__(
        self,
        *,
        disconnection_timeout: float,
        disconnection_check_interval: float,
        primary_device_check_interval: float,
        on_disconnected: Callable,
        on_check_recovery: Callable,
        get_time_since_callback: Callable[[], float],
        logger,
    ):
        self.disconnection_timeout = disconnection_timeout
        self._check_interval = disconnection_check_interval
        self._recovery_interval = primary_device_check_interval
        self._on_disconnected = on_disconnected
        self._on_check_recovery = on_check_recovery
        self._get_time_since_callback = get_time_since_callback
        self._logger = logger

        # Shared state — engine reads/writes these via watchdog attributes
        self.handling_disconnection: bool = False
        self.is_using_fallback: bool = False
        self.primary_device_name: Optional[str] = None

        self._disconnection_running = False
        self._primary_running = False
        self._disconnection_thread: Optional[threading.Thread] = None
        self._primary_thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._disconnection_running = True
        self._primary_running = True
        self._disconnection_thread = threading.Thread(target=self._disconnection_loop, daemon=True)
        self._primary_thread = threading.Thread(target=self._primary_loop, daemon=True)
        self._disconnection_thread.start()
        self._primary_thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._disconnection_running = False
        self._primary_running = False
        if self._disconnection_thread and self._disconnection_thread.is_alive():
            self._disconnection_thread.join(timeout=timeout)
        if self._primary_thread and self._primary_thread.is_alive():
            self._primary_thread.join(timeout=timeout)

    def _disconnection_loop(self) -> None:
        while self._disconnection_running:
            self._check_disconnection()
            time.sleep(self._check_interval)

    def _primary_loop(self) -> None:
        while self._primary_running:
            should_check = (
                self.is_using_fallback
                and self.primary_device_name is not None
                and self.primary_device_name != ""
            )
            if should_check:
                self._on_check_recovery()
            time.sleep(self._recovery_interval)

    def _check_disconnection(self) -> None:
        time_since = self._get_time_since_callback()
        self._logger.debug(f"Time since last callback: {time_since:.2f} seconds")
        if time_since > self.disconnection_timeout and not self.handling_disconnection:
            self.handling_disconnection = True
            self._logger.error(
                f"No callback for {self.disconnection_timeout:.2f} seconds. Device may be disconnected."
            )
            self._on_disconnected()


# ---------------------------------------------------------------------------
# AudioCaptureEngine — pure Python, zero ROS2 imports
# ---------------------------------------------------------------------------


class AudioCaptureEngine:
    """
    Handles device setup, streaming, buffering, and resampling.

    Communicates outward only via two callbacks:
      - on_chunk_ready(chunk, device_name, device_id, samplerate): called for each
        audio chunk ready to be published. The ROS2 node stamps and publishes it.
      - on_device_changed(device_name): called when the active device changes,
        so the node can update its parameters if needed.

    This class has zero ROS2 imports and can be tested standalone.
    """

    def __init__(
        self,
        *,
        dtype: str,
        channels: int,
        chunk: int,
        target_samplerate: int,
        disconnection_timeout: float,
        disconnection_check_interval: float,
        primary_device_check_interval: float,
        on_chunk_ready: Callable[[np.ndarray, str, int, float], None],
        on_device_changed: Callable[[str], None],
        logger,
    ):
        self.dtype = dtype
        self.channels = channels
        self.chunk = chunk
        self.target_samplerate = target_samplerate
        self._on_chunk_ready = on_chunk_ready
        self._on_device_changed = on_device_changed
        self._logger = logger

        self.sd_manager = SoundDeviceManager()

        self.active_device: Optional[ActiveDevice] = None
        self.device_muted: bool = False

        self.stream = None
        self.audio_buffer = np.array([], dtype=np.float32)
        self.target_chunk_size = self.chunk

        # Last-callback timestamp
        self._callback_lock = threading.Lock()
        self.last_callback_time = time.time()

        self.devices = None
        self.available_devices: list = []

        self.watchdog = DeviceWatchdog(
            disconnection_timeout=disconnection_timeout,
            disconnection_check_interval=disconnection_check_interval,
            primary_device_check_interval=primary_device_check_interval,
            on_disconnected=self._on_device_disconnected,
            on_check_recovery=self._check_primary_device_recovery,
            get_time_since_callback=self._time_since_last_callback,
            logger=self._logger,
        )

    # ------------------------------------------------------------------
    # Public lifecycle
    # ------------------------------------------------------------------

    def start(self, device_name_param: str) -> None:
        """Set up a working device and start the watchdog. Call once at startup."""
        self.setup_working_device(device_name_param)
        self.watchdog.start()

    def stop(self) -> None:
        """Stop the watchdog and close the audio stream."""
        self.watchdog.stop()
        self._stop_stream()

    # ------------------------------------------------------------------
    # Timing
    # ------------------------------------------------------------------

    def _time_since_last_callback(self) -> float:
        with self._callback_lock:
            return time.time() - self.last_callback_time

    # ------------------------------------------------------------------
    # Device state helpers
    # ------------------------------------------------------------------

    def _apply_device_info(self, active_device: ActiveDevice) -> None:
        self.active_device = active_device
        self._logger.debug(
            f"Device {active_device.index} ({active_device.name}) is receiving audio."
        )

    # ------------------------------------------------------------------
    # Stream management
    # ------------------------------------------------------------------

    def create_audio_stream(self) -> None:
        """Open and start a stream for `self.active_device`."""
        try:
            self.stream = self.sd_manager.open_stream(
                self.active_device, self.dtype, self.chunk, self.input_callback
            )
            self._logger.info(
                f"Successfully started audio stream with {self.active_device.name} "
                f"and samplerate: {self.active_device.samplerate}."
            )
        except Exception as e:
            self._logger.error(f"Failed to start stream: {e}")
            raise

    def _stop_stream(self) -> None:
        if self.stream is not None:
            try:
                self.sd_manager.stop_stream(self.stream)
                self._logger.debug("Audio stream stopped.")
            except Exception as e:
                self._logger.error(f"Error stopping stream: {e}")
            finally:
                self.stream = None
                self.audio_buffer = np.array([], dtype=np.float32)

    # ------------------------------------------------------------------
    # Device connection methods
    # ------------------------------------------------------------------

    def _connect_fallback_device(self) -> bool:
        """Try every available device until one receives audio."""
        for device_index in self.available_devices:
            device_name = self.devices[device_index]["name"]
            self._logger.debug(f"Testing device {device_index}: {device_name}")
            success, active_device = self.sd_manager.test_device(
                device_index, self.devices, self.channels, self.dtype, self.chunk
            )
            if success:
                self._apply_device_info(active_device)
                self.create_audio_stream()
                self.audio_buffer = np.array([], dtype=np.float32)
                self._on_device_changed(device_name)
                self._logger.debug(f"Successfully connected to fallback device: {device_name}.")
                return True
            self._logger.warn(
                f"Device {device_index} ({device_name}) is not receiving audio or failed to open."
            )
        self._logger.error(
            "No working audio devices found that are receiving input during disconnection recovery."
        )
        return False

    def _connect_primary_device(self, device_name_param: str) -> bool:
        """Try devices matching device_name_param. Normal startup path."""
        matching_devices = self.sd_manager.find_by_name(
            device_name_param, self.available_devices, self.devices, self._logger
        )
        if not matching_devices:
            self._logger.warn(
                f"No devices found containing '{device_name_param}'. Trying all available devices."
            )
            matching_devices = self.available_devices

        for device_index in matching_devices:
            self._logger.debug(
                f"Testing device {device_index}: {self.devices[device_index]['name']}"
            )
            success, active_device = self.sd_manager.test_device(
                device_index, self.devices, self.channels, self.dtype, self.chunk
            )
            if success:
                self._apply_device_info(active_device)
                self.create_audio_stream()
                self.audio_buffer = np.array([], dtype=np.float32)
                return True
            self._logger.warn(f"Device {device_index} is not receiving audio or failed to open.")
        self._logger.error(
            "No working audio devices found that are receiving input. Retrying in 2 seconds..."
        )
        return False

    def setup_working_device(self, device_name_param: str) -> None:
        """Dispatcher: delegates to primary or fallback connection based on device_name_param."""
        if self.watchdog.primary_device_name is None and device_name_param != "":
            self.watchdog.primary_device_name = device_name_param
            self._logger.info(f"Primary device name set to: {device_name_param}")

        self.devices, available_devices = self.sd_manager.query_input_devices()
        self._logger.info("-" * 70)
        self._logger.info("Available devices:")
        for i in available_devices:
            dev = self.devices[i]
            self._logger.info(f"{i}: {dev['name']} ({dev['max_input_channels']} channels)")
        self._logger.info("-" * 70)

        while True:
            try:
                self.devices, self.available_devices = self.sd_manager.query_input_devices()
                if device_name_param == "":
                    self.watchdog.is_using_fallback = True
                    if self._connect_fallback_device():
                        return
                else:
                    self.watchdog.is_using_fallback = False
                    if self._connect_primary_device(device_name_param):
                        return
                time.sleep(2)
            except KeyboardInterrupt:
                self._logger.debug("Operation cancelled by user.")
                raise
            except Exception as e:
                self._logger.error(f"Failed to setup device: {e}")
                self._logger.debug("Retrying in 2 seconds...")
                time.sleep(2)

    # ------------------------------------------------------------------
    # Audio callback
    # ------------------------------------------------------------------

    def input_callback(self, indata, frames, time_input, status) -> None:
        with self._callback_lock:
            self.last_callback_time = time.time()

        audio_data = indata[:, 0].astype(np.float32)

        if status:
            self._logger.warn(f"Stream status: {status}")

        rms = compute_rms(indata)
        self._logger.debug(f"Audio buffer noise level (RMS): {rms:.6f}")

        if rms < 0.001:
            audio_data = np.zeros_like(audio_data)
            self._logger.debug("Audio data muted due to very low noise level (RMS < 0.001).")

        if rms >= 0:
            device_sr = (
                self.active_device.samplerate if self.active_device else self.target_samplerate
            )
            if device_sr != self.target_samplerate:
                if (
                    not hasattr(self, "_last_resampled_index")
                    or self.active_device.index != self._last_resampled_index
                ):
                    self._logger.debug(
                        f"Resampling audio from {device_sr} to {self.target_samplerate}"
                    )
                    self._last_resampled_index = self.active_device.index
                audio_data = resample_audio(
                    audio_data, device_sr, self.target_samplerate, self._logger
                )

            if rms == 0 and not self.device_muted:
                self._logger.warn("RMS is zero, probably the device is muted")
                self.device_muted = True
            elif rms > 0 and self.device_muted:
                self._logger.info("RMS is non-zero again, device has been re-activated")
                self.device_muted = False

            self.audio_buffer = np.concatenate([self.audio_buffer, audio_data])

            while len(self.audio_buffer) >= self.target_chunk_size:
                chunk_data = self.audio_buffer[: self.target_chunk_size]
                self.audio_buffer = self.audio_buffer[self.target_chunk_size :]
                # Deliver to node for publishing — no ROS2 types here
                self._on_chunk_ready(
                    chunk_data.astype(np.float32),
                    self.active_device.name if self.active_device else "",
                    self.active_device.index if self.active_device else -1,
                    float(self.target_samplerate),
                )

    # ------------------------------------------------------------------
    # Disconnection and recovery — called by DeviceWatchdog via callbacks
    # ------------------------------------------------------------------

    def _on_device_disconnected(self) -> None:
        """Watchdog callback: triggered when no audio arrives within timeout."""
        self._on_device_changed("")  # notify node to clear its device_name param
        self._logger.error("Device disconnected. Stopping stream and searching for replacement.")
        self._stop_stream()
        self._handle_device_disconnection()

    def _handle_device_disconnection(self) -> None:
        try:
            self.setup_working_device("")
            with self._callback_lock:
                self.last_callback_time = time.time()
            self.watchdog.handling_disconnection = False
        except KeyboardInterrupt:
            self._logger.debug("Device selection cancelled by user.")
        except Exception as e:
            self._logger.error(f"Failed to setup new device: {e}")

    def _check_primary_device_recovery(self) -> None:
        """Watchdog callback: periodically checks if the primary device is back."""
        try:
            self._logger.debug(
                f"Checking if primary device '{self.watchdog.primary_device_name}' is available..."
            )
            current_devices, current_available = self.sd_manager.query_input_devices()
            matching_devices = self.sd_manager.find_by_name(
                self.watchdog.primary_device_name, current_available, current_devices
            )
            if not matching_devices:
                self._logger.debug(
                    f"Primary device '{self.watchdog.primary_device_name}' not found yet."
                )
                return

            for device_index in matching_devices:
                success, active_device = self.sd_manager.test_device(
                    device_index, current_devices, self.channels, self.dtype, self.chunk
                )
                if success:
                    self._logger.debug(
                        f"Primary device '{active_device.name}' is available again! Switching back..."
                    )
                    self._stop_stream()
                    self.devices = current_devices
                    self.available_devices = current_available
                    self._apply_device_info(active_device)
                    self._on_device_changed(self.watchdog.primary_device_name)
                    self.create_audio_stream()
                    self.audio_buffer = np.array([], dtype=np.float32)
                    self.watchdog.is_using_fallback = False
                    with self._callback_lock:
                        self.last_callback_time = time.time()
                    self._logger.debug(
                        f"Successfully switched back to primary device: {active_device.name}"
                    )
                    return
        except Exception as e:
            self._logger.error(f"Error during primary device recovery check: {e}")
