defmodule InferVision.YOLO.Decoders.V5 do
  @moduledoc """
  Decoder for Ultralytics YOLOv5 / YOLOv7 ONNX exports.

  Expected output: `(1, n_boxes, 5 + n_classes)` with each row
  `[cx, cy, w, h, obj, class_0, ..., class_K]`.
  """

  @behaviour InferVision.YOLO.Decoder

  @impl true
  def decode(outputs, opts) do
    raw =
      outputs
      |> Map.values()
      |> List.first()

    {1, n, channels} = Nx.shape(raw)
    flat = Nx.reshape(raw, {n, channels})

    # score = obj * max_class <= obj, so rows whose objectness is below
    # the score threshold can never be kept. Dropping them first turns a
    # 25200-row argmax into a few dozen rows.
    threshold = Keyword.get(opts, :score_threshold, 0.0)
    obj = flat |> Nx.slice([0, 4], [n, 1]) |> Nx.reshape({n})

    obj_list = Nx.to_flat_list(obj)

    keep =
      obj_list
      |> Enum.with_index()
      |> Enum.flat_map(fn {o, i} -> if o >= threshold, do: [i], else: [] end)

    # Nx has no empty tensors: keep the best row, which NMS's score
    # threshold then drops.
    keep = if keep == [], do: [Nx.to_number(Nx.argmax(obj))], else: keep

    rows = Nx.take(flat, Nx.tensor(keep), axis: 0)
    m = length(keep)

    boxes_xywh = Nx.slice(rows, [0, 0], [m, 4])
    obj = rows |> Nx.slice([0, 4], [m, 1]) |> Nx.reshape({m})
    class_probs = Nx.slice(rows, [0, 5], [m, channels - 5])

    classes = Nx.argmax(class_probs, axis: -1)
    max_class = Nx.reduce_max(class_probs, axes: [-1])

    {InferVision.YOLO.Boxes.xywh_to_xyxy(boxes_xywh), Nx.multiply(obj, max_class), classes}
  end
end
