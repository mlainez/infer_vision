# infer_vision

> ### ⚠️ Very early work — built for a workshop, not for production
>
> This package was written for the **Goatmire Elixir workshop** on running
> Nerves on Fairphone 3 hardware. It exists for tinkering and teaching.
>
> There are no stability guarantees and APIs will change without notice.
>
> See [`nerves_ai`](https://github.com/mlainez/nerves_ai) for the full
> stack and the workshop context.

Nx-tensor vision wrappers for Nerves devices on ARM CPUs.

Part of the [`nerves_ai`](https://github.com/mlainez/nerves_ai) edge-AI
stack. This package is the generic API; ONNX execution and image
decoding sit behind the `InferVision.Backend` behaviour.

## What's here

| Module | What it does |
|---|---|
| `InferVision.YOLO` | Object detection, v5 and v8 output layouts |
| `InferVision.Onnx` | Generic ONNX inference for models with float inputs |
| `InferVision.Preprocess` | JPEG/PNG decode + classifier preprocessing |
| `InferVision.Image` | Resize and normalise raw RGB buffers (needs `arm_ai` + `nx_arm`) |
| `InferVision.Detection` | NMS, IoU, box-format conversion |

## Install

```elixir
defp deps do
  [
    {:infer_vision, github: "mlainez/infer_vision"},
    # plus a backend — on ARM:
    {:arm_ai, github: "mlainez/arm_ai"}
  ]
end
```

```elixir
config :infer_vision, backend: ArmAI.VisionBackend
```

If you depend on `nerves_ai`, this wiring happens for you at boot.

## Usage

### Object detection

```elixir
{:ok, yolo} = InferVision.YOLO.load("/data/models/yolov5n.onnx",
                input_shape: {640, 640})

input = InferVision.Preprocess.load_for_classifier("/data/sample.jpg",
          size: {640, 640},
          layout: :nchw,
          # YOLO wants 0..1 RGB, not ImageNet mean/std
          mean: {0.0, 0.0, 0.0},
          std:  {1.0, 1.0, 1.0})

detections = InferVision.YOLO.detect(yolo, input,
               iou_threshold: 0.45,
               score_threshold: 0.25,
               max_output: 50)
```

Pick the output decoder with `decoder: InferVision.YOLO.Decoders.V5`
(the default, for YOLOv5/v7) or `.V8` (YOLOv8/v11). The legacy
`layout: :v5 | :v8` shortcut still works. The V8 decode step does more
tensor work; set `Nx.global_default_backend(NxArm.Backend)` to keep it
fast.

Both decoders are tested against real models: Ultralytics' YOLOv5n
export (half precision, run in f32) and a YOLOv8n export.

### Generic ONNX

```elixir
{:ok, model} = InferVision.Onnx.load("/data/models/whatever.onnx")
outputs = InferVision.Onnx.run(model, %{"input" => tensor})
```

With `ArmAI.VisionBackend` (tract-onnx), inputs must be float tensors.
Models that take integer inputs, such as token ids for text encoders,
return `{:error, {:unsupported_input_type, name, type}}`. Outputs are
always f32.

## Choosing a YOLO model

YOLOv5n is the recommended starting point: the official Ultralytics
release loads into tract with no conversion step, is about 3× smaller
than v8n (3.9 MB), and detects in about 0.4 s per 640×640 image on a
desktop x86 CPU. It hasn't been timed on the phone with the current code.
Older exports that use the deprecated `Upsample` op don't load; re-export
them at opset 11 or newer.

## Example

`examples/yolo_detection/run.exs` runs YOLOv5n on an image and prints
the detections with COCO labels.

## Toolchain

Built and tested with Erlang/OTP 29.1.1 and Elixir 1.20.4, matching the
official Nerves systems (see `.tool-versions`).

## License

Apache-2.0
