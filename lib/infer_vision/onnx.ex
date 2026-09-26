defmodule InferVision.Onnx do
  @moduledoc """
  Generic ONNX inference — delegates to a configured
  `InferVision.Backend`.

  Today the canonical backend is `ArmAI.VisionBackend` (tract-onnx
  via the `arm_ai` NIF). Other backends can be plugged in by
  implementing the `InferVision.Backend` callbacks.

      {:ok, model} = InferVision.Onnx.load("/root/models/yolov5n.onnx")
      outputs = InferVision.Onnx.run(model, %{"images" => Nx.broadcast(0.0, {1, 3, 640, 640})})

  With `ArmAI.VisionBackend`, models must take float inputs (any
  float tensor is sent as f32). Integer inputs such as token ids
  return `{:error, {:unsupported_input_type, name, type}}`. All
  outputs come back as f32 Nx tensors.
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
        {:ok,
         %__MODULE__{
           handle: handle,
           backend: backend,
           input_names: backend.onnx_input_names(handle),
           output_names: backend.onnx_output_names(handle)
         }}

      {:error, _} = err ->
        err
    end
  end

  @doc """
  Run inference. Returns a map `%{name => Nx.Tensor}`, or
  `{:error, reason}`.
  """
  @spec run(t(), %{String.t() => Nx.Tensor.t()}, keyword()) ::
          %{String.t() => Nx.Tensor.t()} | {:error, term()}
  def run(%__MODULE__{handle: handle, backend: backend}, inputs, opts \\ []) do
    backend.onnx_run(handle, inputs, opts)
  end
end
