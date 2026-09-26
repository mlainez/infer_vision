defmodule InferVision.MixProject do
  use Mix.Project

  @version "0.1.0"

  def project do
    [
      app: :infer_vision,
      version: @version,
      elixir: "~> 1.17",
      elixirc_paths: elixirc_paths(Mix.env()),
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      name: "InferVision",
      description:
        "Generic Nx-tensor vision wrappers (YOLO detection, generic ONNX, image preprocessing) with a pluggable native backend (see `InferVision.Backend`).",
      package: package(),
      docs: [main: "readme", extras: ["README.md"]]
    ]
  end

  def application, do: [extra_applications: [:logger]]

  defp elixirc_paths(:test), do: ["lib", "test/support"]
  defp elixirc_paths(_), do: ["lib"]

  defp deps do
    [
      {:nx, "~> 0.12.0"},
      # Optional: InferVision.Image calls the arm_ai NIF and NxArm.Backend
      # directly, and ArmAI.VisionBackend is the only backend today.
      {:arm_ai, github: "mlainez/arm_ai", optional: true},
      {:nx_arm, github: "mlainez/nx_arm", optional: true}
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
