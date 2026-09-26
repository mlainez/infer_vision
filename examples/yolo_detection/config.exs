import Config

# Fetch the model on first boot with nerves_ai's model hub.
config :nerves_ai,
  models: [
    yolov5n: [
      source: {:url, "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.onnx"},
      path: "/data/models/yolov5n.onnx"
    ]
  ]

# Optional: build only the Cargo features this example needs.
config :arm_ai, features: ["yolo"]
