"""Physical PCA9685 neck backend for BUDDY."""

from __future__ import annotations

from buddy_core.neck.base import BuddyNeckError, NeckBackend


class PCA9685NeckBackend(NeckBackend):
    """Physical neck backend using Adafruit ServoKit."""

    def __init__(
        self,
        *,
        i2c_address: int,
        frequency_hz: float,
        pan_channel: int,
        tilt_channel: int,
    ) -> None:
        try:
            from adafruit_servokit import ServoKit
        except (ImportError, RuntimeError) as exc:
            raise BuddyNeckError(
                "Physical neck mode was requested, but "
                "adafruit_servokit is unavailable."
            ) from exc

        try:
            self._kit = ServoKit(
                channels=16,
                address=int(i2c_address),
                frequency=float(frequency_hz),
            )
        except Exception as exc:
            raise BuddyNeckError(
                "Failed to initialize PCA9685 neck controller."
            ) from exc

        self._pan_channel = int(pan_channel)
        self._tilt_channel = int(tilt_channel)
        self._cleaned_up = False

    def set_pan(self, angle_deg: float) -> None:
        try:
            self._kit.servo[self._pan_channel].angle = float(angle_deg)
        except Exception as exc:
            raise BuddyNeckError(
                "Failed to set PAN servo angle."
            ) from exc

    def set_tilt(self, angle_deg: float) -> None:
        try:
            self._kit.servo[self._tilt_channel].angle = float(angle_deg)
        except Exception as exc:
            raise BuddyNeckError(
                "Failed to set TILT servo angle."
            ) from exc

    def release(self) -> None:
        for channel in (
            self._pan_channel,
            self._tilt_channel,
        ):
            try:
                self._kit.servo[channel].angle = None
            except Exception:
                pass

    def cleanup(self) -> None:
        if self._cleaned_up:
            return

        self.release()

        try:
            pca = getattr(self._kit, "_pca", None)

            if pca is not None:
                pca.deinit()
        except Exception:
            pass
        finally:
            self._cleaned_up = True
