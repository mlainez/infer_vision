defmodule InferVision.YOLOTest do
  use ExUnit.Case, async: true

  describe "load/2" do
    test "returns a struct error tuple for a missing file" do
      assert {:error, _} = InferVision.YOLO.load("/tmp/__nx_arm_no_such_yolo.onnx")
    end

    test "default decoder is V5 (legacy `layout: :v5` shortcut works)" do
      result = InferVision.YOLO.load("/tmp/__missing.onnx")
      assert match?({:error, _}, result)
    end

    test "passing legacy `layout: :v8` shortcut is accepted" do
      assert {:error, _} = InferVision.YOLO.load("/tmp/__missing.onnx", layout: :v8)
    end

    test "passing `decoder:` module is accepted" do
      assert {:error, _} =
               InferVision.YOLO.load("/tmp/__missing.onnx",
                 decoder: InferVision.YOLO.Decoders.V8
               )
    end

    test "unknown legacy layout raises ArgumentError" do
      assert_raise ArgumentError, ~r/unknown YOLO layout/, fn ->
        InferVision.YOLO.load("/tmp/__missing.onnx", layout: :v99)
      end
    end
  end

  describe "struct shape" do
    test "wraps :onnx, :input_name, :input_shape, :decoder" do
      yolo = %InferVision.YOLO{
        onnx: nil,
        input_name: "images",
        input_shape: {640, 640},
        decoder: InferVision.YOLO.Decoders.V5
      }

      assert yolo.decoder == InferVision.YOLO.Decoders.V5
      assert yolo.input_shape == {640, 640}
      assert yolo.input_name == "images"
    end
  end

  describe "Decoder behaviour" do
    test "V5 decoder is defined" do
      assert Code.ensure_loaded?(InferVision.YOLO.Decoders.V5)
      assert function_exported?(InferVision.YOLO.Decoders.V5, :decode, 2)
    end

    test "V8 decoder is defined" do
      assert Code.ensure_loaded?(InferVision.YOLO.Decoders.V8)
      assert function_exported?(InferVision.YOLO.Decoders.V8, :decode, 2)
    end
  end
end
