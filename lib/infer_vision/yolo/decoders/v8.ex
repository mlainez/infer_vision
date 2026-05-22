defmodule InferVision.YOLO.Decoders.V8 do
  @moduledoc """
  Decoder for Ultralytics YOLOv8 / YOLOv9 / YOLOv11 ONNX exports.

  Expected output: `(1, 4 + n_classes, n_boxes)` — channels-first
  (the transpose of v5).
  """

  @behaviour InferVision.YOLO.Decoder

  @impl true
  def decode(outputs, _opts) do
    raw =
      outputs
      |> Map.values()
      |> List.first()

    {1, channels, n} = Nx.shape(raw)
    flat = Nx.reshape(raw, {channels, n}) |> Nx.transpose(axes: [1, 0])

    boxes_xywh = Nx.slice(flat, [0, 0], [n, 4])
    class_probs = Nx.slice(flat, [0, 4], [n, channels - 4])

    classes = Nx.argmax(class_probs, axis: -1)
    scores = Nx.reduce_max(class_probs, axes: [-1])

    boxes_xyxy = InferVision.YOLO.Boxes.xywh_to_xyxy(boxes_xywh)
    {boxes_xyxy, scores, classes}
  end
end
