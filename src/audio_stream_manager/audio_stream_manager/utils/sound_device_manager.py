import sounddevice as sd

from .audio_models import ActiveDevice
from .audio_utils import SuppressStderr, compute_rms


class SoundDeviceManager:
    """Centralized manager for sounddevice interactions, including device discovery, testing, and stream lifecycle management."""

    # ------------------------------------------------------------------
    # Device discovery
    # ------------------------------------------------------------------

    def query_input_devices(self) -> tuple:
        """Query the system for all input-capable devices.

        Returns:
            Tuple of `(full_device_list, list_of_available_indices)`.
        """
        devices = sd.query_devices()
        available = [i for i, dev in enumerate(devices) if dev["max_input_channels"] > 0]
        return devices, available

    def find_by_name(self, name: str, available_devices: list, devices, logger=None) -> list:
        """Return device indices whose names contain *name* (case-insensitive).

        Args:
            name: Substring to search for. If empty, all *available_devices* are returned.
            available_devices: Indices to search within.
            devices: Full device list from :meth:`query_input_devices`.
            logger: Optional ROS2 logger for debug messages.

        Returns:
            List of matching device indices.
        """
        if not name:
            return available_devices

        name_upper = name.upper()
        matching = []
        for idx in available_devices:
            if name_upper in devices[idx]["name"].upper():
                matching.append(idx)
                if logger:
                    logger.debug(f"Found matching device: {devices[idx]['name']} (index {idx})")
        return matching

    # ------------------------------------------------------------------
    # Device testing
    # ------------------------------------------------------------------

    def _get_device_parameters(self, device_index: int, devices, channels: int) -> tuple:
        """Get device parameters for a given device index.

        Returns:
            Tuple of `(device_index, device_dict, samplerate, actual_channels)`.
        """
        device_index = int(device_index)
        device = devices[device_index]
        samplerate = int(device["default_samplerate"])
        actual_channels = min(device["max_input_channels"], channels)
        return device_index, device, samplerate, actual_channels

    def _test_device_stream(
        self, device_index: int, devices, channels: int, dtype: str, chunk: int
    ) -> tuple:
        """Open a short test stream and check whether the device is receiving audio.

        Returns:
            `(True, info_dict)` if RMS > 0, `(False, None)` otherwise.
        """
        try:
            device_index, device, samplerate, actual_channels = self._get_device_parameters(
                device_index, devices, channels
            )
            with SuppressStderr():
                test_stream = sd.InputStream(
                    device=device_index,
                    samplerate=samplerate,
                    channels=actual_channels,
                    dtype=dtype,
                    blocksize=chunk,
                    latency="low",
                )
                test_stream.start()
                audio_data, _ = test_stream.read(chunk)
                test_stream.stop()
                test_stream.close()

            if compute_rms(audio_data) > 0:
                return True, {
                    "device": device,
                    "device_index": device_index,
                    "device_samplerate": samplerate,
                    "device_channels": actual_channels,
                }
            return False, None
        except Exception:
            return False, None

    def test_device(
        self, device_index: int, devices, channels: int, dtype: str, chunk: int
    ) -> tuple:
        """Test a device and return an :class:`ActiveDevice` on success.

        Returns:
            `(True, ActiveDevice)` if receiving audio, `(False, None)` otherwise.
        """
        success, info = self._test_device_stream(device_index, devices, channels, dtype, chunk)
        if success:
            device = info["device"]
            return True, ActiveDevice(
                device=device,
                name=device["name"],
                index=info["device_index"],
                samplerate=info["device_samplerate"],
                channels=info["device_channels"],
            )
        return False, None

    # ------------------------------------------------------------------
    # Stream lifecycle
    # ------------------------------------------------------------------

    def open_stream(
        self, active_device: ActiveDevice, dtype: str, chunk: int, callback
    ) -> "sd.InputStream":
        """Open and start an InputStream for *active_device*.

        Args:
            active_device: The device to open.
            dtype: Audio data type string (e.g. ``"float32"``).
            chunk: Block size in samples.
            callback: Audio callback passed to ``sd.InputStream``.

        Returns:
            A started ``sd.InputStream``.
        """
        with SuppressStderr():
            stream = sd.InputStream(
                device=active_device.index,
                samplerate=active_device.samplerate,
                channels=active_device.channels,
                dtype=dtype,
                blocksize=chunk,
                callback=callback,
                latency="low",
            )
            stream.start()
        return stream

    def stop_stream(self, stream: "sd.InputStream") -> None:
        """Stop and close *stream* safely, suppressing ALSA/PortAudio noise."""
        with SuppressStderr():
            stream.stop()
            stream.close()
