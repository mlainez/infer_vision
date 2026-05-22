defmodule InferVision do
  @moduledoc """
  Nx-tensor vision wrappers on ARM CPUs.

  * `InferVision.YOLO` — object detection (v5 / v8 layouts) via tract-onnx
  * `InferVision.OCR` — text recognition
  * `InferVision.Face` — detection + recognition
  * `InferVision.Onnx` — generic ONNX inference
  * `InferVision.StableDiffusion` — demo-only on phone-class ARM
  * `InferVision.Preprocess` — JPEG/PNG decode + classifier preprocess
  * `InferVision.Image` — Nx-tensor image manipulation helpers
  * `InferVision.Detection` — NMS, IoU, anchor decoding
  """
end
