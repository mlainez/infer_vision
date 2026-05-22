defmodule ArmVision.MixProject do
  use Mix.Project

  @version "0.1.0"

  def project do
    [
      app: :arm_vision,
      version: @version,
      elixir: "~> 1.15",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      name: "ArmVision",
      description:
        "Nx-tensor vision model wrappers + preprocessing on ARM CPUs (YOLO / OCR / Face / generic ONNX / Stable Diffusion via tract-onnx)",
      package: package(),
      docs: [main: "readme", extras: ["README.md"]]
    ]
  end

  def application, do: [extra_applications: [:logger]]

  defp deps do
    [
      {:rustler, "~> 0.36", optional: true},
      {:rustler_precompiled, "~> 0.8"},
      {:nx, "~> 0.9"},
      {:arm_ai, path: "../arm_ai"},
      {:nx_arm, path: "../nx_arm"},
      {:arm_nx_primitives, path: "../arm_nx_primitives"}
    ]
  end

  defp package do
    [
      name: :arm_vision,
      licenses: ["Apache-2.0"],
      files: ~w(lib mix.exs README.md),
      links: %{"GitHub" => "https://github.com/marclainez/arm_vision"}
    ]
  end
end
