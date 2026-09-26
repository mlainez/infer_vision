defmodule InferVision do
  @moduledoc """
  Nx-tensor vision wrappers with a pluggable backend.

  * `InferVision.YOLO` — object detection (v5 / v8 output layouts)
  * `InferVision.Onnx` — generic ONNX inference (float inputs)
  * `InferVision.Preprocess` — JPEG/PNG decode + classifier preprocess
  * `InferVision.Image` — raw RGB buffer helpers (requires `arm_ai` + `nx_arm`)
  * `InferVision.Detection` — NMS, IoU, box conversion
  """
end
