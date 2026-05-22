defmodule InferVision.YOLO.Decoders.V5 do
  @moduledoc """
  Decoder for Ultralytics YOLOv5 / YOLOv7 ONNX exports.

  Expected output: `(1, n_boxes, 5 + n_classes)` with each row
  `[cx, cy, w, h, obj, class_0, ..., class_K]`.
  """

  @behaviour InferVision.YOLO.Decoder

  @impl true
  def decode(outputs, _opts) do
    raw =
      outputs
      |> Map.values()
      |> List.first()

    {1, n, channels} = Nx.shape(raw)
    flat = Nx.reshape(raw, {n, channels})

    boxes_xywh = Nx.slice(flat, [0, 0], [n, 4])
    obj = Nx.slice(flat, [0, 4], [n, 1]) |> Nx.reshape({n})
    class_probs = Nx.slice(flat, [0, 5], [n, channels - 5])

    classes = Nx.argmax(class_probs, axis: -1)
    max_class = Nx.reduce_max(class_probs, axes: [-1])

    scores = Nx.multiply(obj, max_class)
    boxes_xyxy = InferVision.YOLO.Boxes.xywh_to_xyxy(boxes_xywh)
    {boxes_xyxy, scores, classes}
  end
end
