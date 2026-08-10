defmodule InferVision.MixProject do
  use Mix.Project

  @version "0.1.0"

  def project do
    [
      app: :infer_vision,
      version: @version,
      elixir: "~> 1.15",
      elixirc_paths: elixirc_paths(Mix.env()),
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      name: "Vision",
      description:
        "Generic Nx-tensor vision model wrappers (YOLO / OCR / Face / generic ONNX) with a pluggable native backend (see `InferVision.Backend`).",
      package: package(),
      docs: [main: "readme", extras: ["README.md"]]
    ]
  end

  def application, do: [extra_applications: [:logger]]

  defp elixirc_paths(:test), do: ["lib", "test/support"]
  defp elixirc_paths(_), do: ["lib"]

  defp deps do
    [
      {:nx, "~> 0.9"},
      {:nx_primitives, path: "../nx_primitives"},
      {:arm_ai, path: "../arm_ai", only: [:dev, :test]},
      {:nx_arm, path: "../nx_arm", only: [:dev, :test]},
      {:rustler, "~> 0.36", optional: true},
      {:rustler_precompiled, "~> 0.8"}
    ]
  end

  defp package do
    [
      name: :infer_vision,
      licenses: ["Apache-2.0"],
      files: ~w(lib mix.exs README.md LICENSE),
      links: %{"GitHub" => "https://github.com/mlainez/infer_vision"}
    ]
  end
end
