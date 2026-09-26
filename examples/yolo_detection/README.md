# Still-image object detection (YOLOv5n)

Runs the official Ultralytics YOLOv5n ONNX export on one image and
prints the detections with their COCO labels.

The model ships at opset 17, so tract-onnx loads it without conversion.
Boxes are in 640×640 input coordinates. Scale by `width / 640` and
`height / 640` to map them back onto the source image.

## Run it

1. Merge `config.exs` into your firmware config. On first boot the model
   hub downloads the 3.9 MB model to `/data/models/yolov5n.onnx`.
2. Put a JPEG or PNG at `/data/sample.jpg`.
3. From iex on the device: `Code.eval_file("/path/to/run.exs")`.

On a host, pass the paths explicitly:

```sh
mix run examples/yolo_detection/run.exs yolov5n.onnx bus.jpg
```

## Expected output

For Ultralytics' `bus.jpg` sample, on an x86_64 desktop:

```
Detected 5 objects in 424 ms
  person 0.85 [41, 237, 175, 537]
  person 0.8 [543, 227, 640, 520]
  person 0.76 [178, 243, 271, 509]
  person 0.48 [0, 328, 56, 524]
  bus 0.37 [36, 107, 652, 450]
```

This timing is from an x86_64 desktop; it hasn't been measured on the
phone with the current code.

For YOLOv8/v11 exports pass `decoder: InferVision.YOLO.Decoders.V8` to
`InferVision.YOLO.load/2`. Their decode step does more tensor work, so set
`Nx.global_default_backend(NxArm.Backend)` to keep it fast.
