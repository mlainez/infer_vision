# infer_vision

> ### ⚠️ Very early work — built for a workshop, not for production
>
> This package was written for the **Goatmire Elixir workshop** on running
> Nerves on Fairphone 3 hardware. It exists for tinkering and teaching.
>
> It is **not an actively maintained project** (yet). There are no
> stability guarantees, APIs will change without notice, and parts of it
> are wired-but-unproven. Treat it as a starting point to hack on, not as
> a dependency to build a product on.
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
| `InferVision.Onnx` | Generic ONNX inference — load a model, run tensors through it |
| `InferVision.Preprocess` | JPEG/PNG decode + classifier preprocessing |
| `InferVision.Image` | Resize, normalise, tensor image helpers |
| `InferVision.Detection` | NMS, IoU, box decoding |
| `InferVision.OCR` | Two-stage OCR pipeline glue |
| `InferVision.Face` | Face detection + recognition pipeline glue |
| `InferVision.StableDiffusion` | Diffusion wiring — demo-only on phone-class ARM |

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
(the default) or `.V8`. The legacy `layout: :v5 | :v8` shortcut still
works.

### Generic ONNX

```elixir
{:ok, model} = InferVision.Onnx.load("/data/models/whatever.onnx")
output = InferVision.Onnx.run(model, %{"input" => tensor})
```

## Choosing a YOLO model

YOLOv5n is the recommended starting point: the official Ultralytics
release ships at **opset 17** with `Resize`, so it loads into
`tract-onnx` with no conversion step. It's also ~3× smaller than v8
(3.8 MB) and a bit faster on CPU. Expect roughly **1 s per 640×640 image**
on a Cortex-A73.

## Caveats

**`InferVision.Detection.decode_yolov5/3` is an unimplemented stub** —
it raises. Use `InferVision.YOLO.Decoders.V5` via `YOLO.detect/3`
instead; `Detection`'s `nms/3`, `iou/2` and `decode_xywh_to_xyxy/1` are
real and used by that path.

**`OCR` and `Face` are pipeline glue, not models.** They give you the
load/detect/recognize plumbing around an ONNX pair; you supply the
models and their pre/post-processing. Neither has been validated
end-to-end on device.

**`StableDiffusion` is not interactive.** Diffusion on a Cortex-A73 takes
*minutes* per image even at 256×256. It's wired for the "generate while
the user is away" case. For interactive latency, run it on a host.

## License

Apache-2.0
