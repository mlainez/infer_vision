defmodule ArmVision do
  @moduledoc """
  Nx-tensor vision wrappers on ARM CPUs.

  * `ArmVision.YOLO` — object detection (v5 / v8 layouts) via tract-onnx
  * `ArmVision.OCR` — text recognition
  * `ArmVision.Face` — detection + recognition
  * `ArmVision.Onnx` — generic ONNX inference
  * `ArmVision.StableDiffusion` — demo-only on phone-class ARM
  * `ArmVision.Preprocess` — JPEG/PNG decode + classifier preprocess
  * `ArmVision.Image` — Nx-tensor image manipulation helpers
  * `ArmVision.Detection` — NMS, IoU, anchor decoding
  """
end
