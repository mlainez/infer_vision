defmodule InferVision.YOLO do
  @moduledoc """
  YOLO object detection — generic over ONNX backend AND decoder.

  Two axes of extensibility:

    * **Backend** — set `:backend` to choose the ONNX runtime
      (`ArmAI.VisionBackend` for tract-onnx). Configured via
      `InferVision.Backend`.
    * **Decoder** — set `:decoder` to handle different YOLO output
      layouts. Built-in: `InferVision.YOLO.Decoders.V5` (v5/v7)
      and `InferVision.YOLO.Decoders.V8` (v8/v9/v11/YOLOX-style
      channels-first). Custom-trained models can implement the
      `InferVision.YOLO.Decoder` behaviour and slot in here.

  ## Example

      {:ok, model} = InferVision.YOLO.load("/data/yolov5n.onnx")  # default decoder: V5
      image = InferVision.Preprocess.load_for_classifier("/data/cat.jpg",
                size: {640, 640},
                mean: {0.0, 0.0, 0.0},
                std:  {1.0, 1.0, 1.0},
                layout: :nchw)
      detections = InferVision.YOLO.detect(model, image,
                     iou_threshold: 0.45,
                     score_threshold: 0.25)
      # detections = [%{box: {x1, y1, x2, y2}, class: 16, score: 0.91}, ...]

  For YOLOv8 / v11:

      {:ok, model} = InferVision.YOLO.load("/data/yolov8n.onnx",
                       decoder: InferVision.YOLO.Decoders.V8)
  """

  defstruct [:onnx, :input_name, :input_shape, :decoder]

  @doc """
  Load a YOLO ONNX file.

  ## Options

    * `:decoder` — module implementing `InferVision.YOLO.Decoder`.
      Default `InferVision.YOLO.Decoders.V5`. Accepts the legacy
      `layout: :v5 | :v8` shortcut for back-compat.
    * `:input_shape` — `{height, width}` of the model's input.
      Default `{640, 640}`.
    * `:backend` — override the `InferVision.Backend` for this load.
  """
  @spec load(Path.t(), keyword()) :: {:ok, %__MODULE__{}} | {:error, term()}
  def load(path, opts \\ []) do
    decoder = resolve_decoder(opts)
    input_shape = Keyword.get(opts, :input_shape, {640, 640})

    case InferVision.Onnx.load(path, opts) do
      {:ok, model} ->
        input_name = List.first(model.input_names) || "images"

        {:ok,
         %__MODULE__{
           onnx: model,
           input_name: input_name,
           input_shape: input_shape,
           decoder: decoder
         }}

      err ->
        err
    end
  end

  @doc """
  Run detection. `image` is a preprocessed `{1, 3, H, W}` (or `{3, H, W}`)
  tensor — use `InferVision.Preprocess.load_for_classifier/2` to build it.

  Returns a list of
  `%{box: {x1, y1, x2, y2}, class: integer, score: float}`.
  """
  @spec detect(%__MODULE__{}, Nx.Tensor.t(), keyword()) ::
          [%{box: {float(), float(), float(), float()}, class: non_neg_integer(), score: float()}]
          | {:error, term()}
  def detect(%__MODULE__{onnx: onnx, input_name: in_name, decoder: decoder}, image, opts \\ []) do
    iou_threshold = Keyword.get(opts, :iou_threshold, 0.45)
    score_threshold = Keyword.get(opts, :score_threshold, 0.25)
    max_output = Keyword.get(opts, :max_output, 100)

    image =
      case Nx.shape(image) do
        {_n, _c, _h, _w} -> image
        {c, h, w} -> Nx.reshape(image, {1, c, h, w})
      end

    case InferVision.Onnx.run(onnx, %{in_name => image}) do
      {:error, _} = err -> err
      outputs -> decode_and_filter(outputs, decoder, opts, iou_threshold, score_threshold, max_output)
    end
  end

  defp decode_and_filter(outputs, decoder, opts, iou_threshold, score_threshold, max_output) do
    {boxes, scores, classes} = decoder.decode(outputs, Keyword.put(opts, :score_threshold, score_threshold))

    kept =
      InferVision.Detection.nms(boxes, scores,
        iou_threshold: iou_threshold,
        score_threshold: score_threshold,
        max_output: max_output
      )

    boxes_list = Nx.to_list(boxes)
    scores_list = Nx.to_flat_list(scores)
    classes_list = Nx.to_flat_list(classes)

    for idx <- kept do
      [x1, y1, x2, y2] = Enum.at(boxes_list, idx)

      %{
        box: {x1 * 1.0, y1 * 1.0, x2 * 1.0, y2 * 1.0},
        class: Enum.at(classes_list, idx),
        score: Enum.at(scores_list, idx)
      }
    end
  end

  # Back-compat: accept the old `layout: :v5 | :v8` shortcut, but
  # `decoder:` is the new canonical option.
  defp resolve_decoder(opts) do
    case Keyword.get(opts, :decoder) do
      nil ->
        case Keyword.get(opts, :layout, :v5) do
          :v5 -> InferVision.YOLO.Decoders.V5
          :v8 -> InferVision.YOLO.Decoders.V8
          other -> raise ArgumentError, "unknown YOLO layout: #{inspect(other)}. Use `decoder: SomeModule` for custom layouts."
        end

      mod when is_atom(mod) ->
        mod
    end
  end
end
