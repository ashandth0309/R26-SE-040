"""One small wrapper so every model runs through onnxruntime when it is
installed and through OpenCV's DNN module otherwise (CPU only)."""

from __future__ import annotations

import numpy as np

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover
    ort = None

import cv2


class OnnxSession:
    def __init__(self, path: str, threads: int = 2,
                 prefer_onnxruntime: bool = True) -> None:
        self.path = str(path)
        self.backend = "opencv_dnn"
        self._net = None
        self._sess = None

        if prefer_onnxruntime and ort is not None:
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = int(threads)
            opts.inter_op_num_threads = 1
            opts.log_severity_level = 3
            self._sess = ort.InferenceSession(
                self.path, opts, providers=["CPUExecutionProvider"])
            self.backend = "onnxruntime"
            self.input_names = [i.name for i in self._sess.get_inputs()]
            self.input_shapes = [i.shape for i in self._sess.get_inputs()]
            self.output_names = [o.name for o in self._sess.get_outputs()]
        else:
            self._net = cv2.dnn.readNetFromONNX(self.path)
            self.input_names = []
            self.input_shapes = []
            self.output_names = list(self._net.getUnconnectedOutLayersNames())

    def run(self, *inputs: np.ndarray) -> list[np.ndarray]:
        if self._sess is not None:
            feed = {n: np.ascontiguousarray(a)
                    for n, a in zip(self.input_names, inputs, strict=True)}

            return self._sess.run(self.output_names, feed)

        self._net.setInput(np.ascontiguousarray(inputs[0]))

        return [np.asarray(o) for o in self._net.forward(self.output_names)]
