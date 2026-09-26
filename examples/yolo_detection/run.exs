# Example — still-image object detection with YOLOv5n.
#
# Usage (from an iex session on the device, or `mix run` on a host):
#
#     Code.eval_file("run.exs")                       # default paths
#     System.argv(["/data/models/yolov5n.onnx", "/data/sample.jpg"])
#
# Needs `config :infer_vision, backend: ArmAI.VisionBackend`
# (`nerves_ai` sets this at boot).

[model_path, image_path] =
  case System.argv() do
    [m, i] -> [m, i]
    _ -> ["/data/models/yolov5n.onnx", "/data/sample.jpg"]
  end

for path <- [model_path, image_path], not File.exists?(path) do
  raise "missing #{path} — see config.exs for the model download"
end

coco = ~w(person bicycle car motorcycle airplane bus train truck boat traffic_light
  fire_hydrant stop_sign parking_meter bench bird cat dog horse sheep cow elephant bear
  zebra giraffe backpack umbrella handbag tie suitcase frisbee skis snowboard
  sports_ball kite baseball_bat baseball_glove skateboard surfboard tennis_racket bottle
  wine_glass cup fork knife spoon bowl banana apple sandwich orange broccoli carrot
  hot_dog pizza donut cake chair couch potted_plant bed dining_table toilet tv laptop
  mouse remote keyboard cell_phone microwave oven toaster sink refrigerator book clock
  vase scissors teddy_bear hair_drier toothbrush)

{:ok, yolo} = InferVision.YOLO.load(model_path, input_shape: {640, 640})

# YOLO expects 0–1 RGB, not ImageNet mean/std.
input =
  InferVision.Preprocess.load_for_classifier(image_path,
    size: {640, 640},
    mean: {0.0, 0.0, 0.0},
    std: {1.0, 1.0, 1.0}
  )

{us, detections} = :timer.tc(fn -> InferVision.YOLO.detect(yolo, input, score_threshold: 0.25) end)

IO.puts("Detected #{length(detections)} objects in #{div(us, 1000)} ms")

for %{class: c, score: s, box: {x1, y1, x2, y2}} <- detections do
  box = Enum.map([x1, y1, x2, y2], &round/1)
  IO.puts("  #{Enum.at(coco, c)} #{Float.round(s, 2)} #{inspect(box)}")
end
