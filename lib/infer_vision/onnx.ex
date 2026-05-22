defmodule InferVision.Onnx do
  @moduledoc """
  Generic ONNX inference — delegates to a configured
  `InferVision.Backend`.

  Today the canonical backend is `ArmAI.VisionBackend` (tract-onnx
  via the `arm_ai` NIF). Other backends can be plugged in by
  implementing the `InferVision.Backend` callbacks.

      {:ok, model} = InferVision.Onnx.load("/root/all-MiniLM-L6-v2.onnx")
      outputs = InferVision.Onnx.run(model, %{
        "input_ids"      => Nx.tensor([[101, 2129, 2024, 2017, 102]], type: :s64),
        "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1]], type: :s64)
      })

  All outputs come back as f32 Nx tensors.
  """

  defstruct [:handle, :backend, :input_names, :output_names]

  @type t :: %__MODULE__{
          handle: reference(),
          backend: module(),
          input_names: [String.t()],
          output_names: [String.t()]
        }

  @doc """
  Load an ONNX file. Returns the model + its declared input/output names.
  """
  @spec load(Path.t(), keyword()) :: {:ok, t()} | {:error, term()}
  def load(path, opts \\ []) do
    backend = InferVision.Backend.resolve(opts)

    case backend.onnx_load(path, opts) do
      {:ok, handle} ->
        in_specs = backend.onnx_input_specs(handle)
        out_specs = backend.onnx_output_specs(handle)

        {:ok,
         %__MODULE__{
           handle: handle,
           backend: backend,
           input_names: Enum.map(in_specs, &elem(&1, 0)),
           output_names: Enum.map(out_specs, &elem(&1, 0))
         }}

      {:error, _} = err ->
        err
    end
  end

  @doc """
  Run inference. Returns a map `%{name => Nx.Tensor}`.
  """
  @spec run(t(), %{String.t() => Nx.Tensor.t()}, keyword()) :: %{String.t() => Nx.Tensor.t()}
  def run(%__MODULE__{handle: handle, backend: backend}, inputs, opts \\ []) do
    backend.onnx_run(handle, inputs, opts)
  end

  @doc "Input specs of a loaded model: `[{name, dtype, shape}, ...]`."
  def input_specs(%__MODULE__{handle: handle, backend: backend}),
    do: backend.onnx_input_specs(handle)

  @doc "Output specs of a loaded model."
  def output_specs(%__MODULE__{handle: handle, backend: backend}),
    do: backend.onnx_output_specs(handle)
end
