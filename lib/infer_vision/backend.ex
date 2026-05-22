defmodule InferVision.Backend do
  @moduledoc """
  Behaviour for vision inference backends.

  An implementation provides:

  * tract-onnx-style ONNX runtime (load + run)
  * YOLO load + detect
  * OCR / Face / StableDiffusion (specialised model wrappers)
  * image decode + classifier preprocess (JPEG/PNG → tensor)

  ## Configuring the active backend

      config :vision, backend: ArmAI.VisionBackend

  Override per-call with `backend:` on any `Vision.*` function.
  """

  @doc "Load an ONNX model. Returns an opaque backend handle."
  @callback onnx_load(path :: String.t(), opts :: keyword()) ::
              {:ok, term()} | {:error, term()}

  @doc "Run an ONNX model. Inputs and outputs use Nx tensors keyed by string names."
  @callback onnx_run(handle :: term(), inputs :: %{String.t() => Nx.Tensor.t()}, opts :: keyword()) ::
              %{String.t() => Nx.Tensor.t()} | {:error, term()}

  @doc "Input specs of a loaded ONNX model: `[{name, type, shape}, ...]`."
  @callback onnx_input_specs(handle :: term()) :: [{String.t(), atom(), tuple()}]

  @doc "Output specs of a loaded ONNX model: `[{name, type, shape}, ...]`."
  @callback onnx_output_specs(handle :: term()) :: [{String.t(), atom(), tuple()}]

  @doc "Decode an image file (JPEG/PNG) → `{rgb_bytes, width, height}`."
  @callback decode_to_rgb8(path :: String.t()) :: {:ok, {binary(), pos_integer(), pos_integer()}} | {:error, term()}

  @doc """
  Load + preprocess an image for a classifier: decode, resize,
  optionally CHW-transpose, and ImageNet-mean/std normalise.
  Returns an Nx tensor.
  """
  @callback load_for_classifier(path :: String.t(), opts :: keyword()) :: Nx.Tensor.t()

  @doc """
  Return the configured backend module. Reads the `:backend`
  option, falling back to `Application.get_env(:infer_vision, :backend)`.
  Raises if neither is set.
  """
  @spec resolve(keyword()) :: module()
  def resolve(opts) do
    case Keyword.get(opts, :backend) || Application.get_env(:infer_vision, :backend) do
      nil ->
        raise """
        No Vision backend configured. Add one to your config:

            config :vision, backend: ArmAI.VisionBackend

        Or pass `backend:` explicitly to the Vision.* call.
        """

      backend when is_atom(backend) ->
        backend
    end
  end
end
