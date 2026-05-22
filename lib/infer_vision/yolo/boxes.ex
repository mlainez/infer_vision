defmodule InferVision.YOLO.Boxes do
  @moduledoc false
  # Shared helpers across decoders.

  @doc "Convert `{n, 4}` xywh (cx, cy, w, h) → xyxy (x1, y1, x2, y2)."
  @spec xywh_to_xyxy(Nx.Tensor.t()) :: Nx.Tensor.t()
  def xywh_to_xyxy(boxes_xywh) do
    n = Nx.axis_size(boxes_xywh, 0)
    cx = Nx.slice(boxes_xywh, [0, 0], [n, 1])
    cy = Nx.slice(boxes_xywh, [0, 1], [n, 1])
    w = Nx.slice(boxes_xywh, [0, 2], [n, 1])
    h = Nx.slice(boxes_xywh, [0, 3], [n, 1])

    half_w = Nx.divide(w, 2)
    half_h = Nx.divide(h, 2)

    x1 = Nx.subtract(cx, half_w)
    y1 = Nx.subtract(cy, half_h)
    x2 = Nx.add(cx, half_w)
    y2 = Nx.add(cy, half_h)

    Nx.concatenate([x1, y1, x2, y2], axis: 1)
  end
end
