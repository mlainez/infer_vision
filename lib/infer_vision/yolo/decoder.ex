defmodule InferVision.YOLO.Decoder do
  @moduledoc """
  Behaviour for YOLO output-tensor decoders.

  Different YOLO variants (v5, v8, v11, YOLOX, custom-trained) emit
  output tensors with different shapes and channel orderings. The
  inference step is the same; the *decoding* differs.

  A `Decoder` knows how to:

    * inspect the raw model output (a map keyed by output name)
    * return `{boxes_xyxy, scores, classes}` tensors that
      `InferVision.Detection.nms/3` can consume

  ## Built-in decoders

    * `InferVision.YOLO.Decoders.V5` — Ultralytics YOLOv5/v7 layout
      `(1, n_boxes, 5 + n_classes)` with `[cx, cy, w, h, obj, cls...]`
    * `InferVision.YOLO.Decoders.V8` — Ultralytics YOLOv8/v9/v11
      layout `(1, 4 + n_classes, n_boxes)` (channels-first)

  ## Adding a custom decoder

      defmodule MyDecoder do
        @behaviour InferVision.YOLO.Decoder

        @impl true
        def decode(outputs, _opts) do
          raw = outputs["output0"]
          # ...your decoding logic...
          {boxes_xyxy, scores, classes}
        end
      end

      {:ok, model} = InferVision.YOLO.load("custom.onnx", decoder: MyDecoder)
  """

  @doc """
  Take the raw map of model outputs (keyed by ONNX output name)
  and the decoder's keyword options. Return `{boxes_xyxy, scores,
  classes}` Nx tensors ready for NMS.

    * `boxes_xyxy` — `{n, 4}` f32, `[x1, y1, x2, y2]`
    * `scores` — `{n}` f32
    * `classes` — `{n}` integer
  """
  @callback decode(outputs :: %{String.t() => Nx.Tensor.t()}, opts :: keyword()) ::
              {Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t()}
end
