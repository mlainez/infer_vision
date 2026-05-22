defmodule InferVision.Preprocess do
  @moduledoc """
  Image preprocessing for vision models — generic API delegating
  to a configured `InferVision.Backend`.

      # ImageNet-normalised 224×224 RGB tensor for ViT / EfficientNet:
      x = InferVision.Preprocess.load_for_classifier("/data/cat.jpg",
            size: {224, 224},
            mean: {0.485, 0.456, 0.406},
            std:  {0.229, 0.224, 0.225},
            layout: :nchw
          )
      # x is shape {3, 224, 224} as an Nx tensor.

  Or the step-by-step path when you need raw pixels:

      {:ok, {bin, w, h}} = InferVision.Preprocess.decode_to_rgb8("/data/cat.jpg")
  """

  @doc """
  Decode any container the backend understands (JPEG/PNG/WebP, depending
  on the impl) into raw RGB8 bytes plus dimensions.
  Returns `{:ok, {rgb_binary, width, height}}` or `{:error, reason}`.
  """
  @spec decode_to_rgb8(Path.t(), keyword()) ::
          {:ok, {binary(), pos_integer(), pos_integer()}} | {:error, term()}
  def decode_to_rgb8(path, opts \\ []) do
    InferVision.Backend.resolve(opts).decode_to_rgb8(path)
  end

  @doc """
  Decode + resize + normalise an image into a model-ready f32 tensor.

  ## Options

    * `:size` — `{height, width}` (required). The model's input shape.
    * `:mean` — `{r, g, b}` per-channel mean in [0, 1] (default ImageNet).
    * `:std` — `{r, g, b}` per-channel stddev in [0, 1] (default ImageNet).
    * `:layout` — `:nchw` (PyTorch / Bumblebee default) or `:nhwc`
      (TensorFlow). Default `:nchw`.
    * `:backend` — override the configured `InferVision.Backend`.

  Returns an Nx tensor:
    * Shape `{3, h, w}` when `:layout == :nchw`
    * Shape `{h, w, 3}` when `:layout == :nhwc`
  """
  @spec load_for_classifier(Path.t(), keyword()) :: Nx.Tensor.t()
  def load_for_classifier(path, opts) do
    InferVision.Backend.resolve(opts).load_for_classifier(path, opts)
  end
end
