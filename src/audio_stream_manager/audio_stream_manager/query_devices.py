import sounddevice as sd


def list_audio_devices():
    """List all available audio devices with their IDs and info"""
    print("=== AUDIO DEVICES AVAILABLE ===\n")

    # Get all devices
    devices = sd.query_devices()

    print("\nINPUT DEVICES ONLY:")
    print("-" * 30)
    input_devices = [
        i for i, dev in enumerate(devices) if dev["max_input_channels"] > 0
    ]
    for device_id in input_devices:
        device = devices[device_id]
        print(
            f"ID: {device_id} - {device['name']} ({device['max_input_channels']} channels)"
        )


if __name__ == "__main__":
    try:
        list_audio_devices()
    except Exception as e:
        print(f"Error listing devices: {e}")
