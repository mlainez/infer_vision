defmodule InferVision.YOLOModelTest do
  # Needs, in $NERVES_AI_MODELS:
  #   yolov5n.onnx  github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.onnx
  #   yolov8n.onnx  huggingface.co/Kalray/yolov8 (yolov8n.onnx)
  #   bus.jpg       ultralytics.com/images/bus.jpg
  use ExUnit.Case, async: false

  @moduletag :models
  @moduletag timeout: 120_000

  for {file, decoder} <- [
        {"yolov5n.onnx", InferVision.YOLO.Decoders.V5},
        {"yolov8n.onnx", InferVision.YOLO.Decoders.V8}
      ] do
    test "#{file} detects the bus and the people in bus.jpg" do
      detect(unquote(file), unquote(decoder))
    end
  end

  defp detect(file, decoder) do
    dir = System.fetch_env!("NERVES_AI_MODELS")
    {:ok, yolo} = InferVision.YOLO.load(Path.join(dir, file), decoder: decoder)

    image =
      InferVision.Preprocess.load_for_classifier(Path.join(dir, "bus.jpg"),
        size: {640, 640},
        mean: {0.0, 0.0, 0.0},
        std: {1.0, 1.0, 1.0}
      )

    detections = InferVision.YOLO.detect(yolo, image)
    classes = Enum.map(detections, & &1.class)

    # COCO: 0 = person, 5 = bus
    assert 5 in classes
    assert Enum.count(classes, &(&1 == 0)) >= 3
  end
end
