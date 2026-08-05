// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Text;

namespace GraphReader.Integration.Tests.IntegrationSmoke;

/// <summary>
/// Generates small executable ONNX graphs for production-composition contract tests.
/// These deterministic graphs are test infrastructure, not trained weights.
/// </summary>
internal static class GeneratedOcrContractOnnx
{
    internal static byte[] BuildDetectionModel()
    {
        byte[] input = ValueInfo("image", 1, 3, "H", "W");
        byte[] output = ValueInfo("text_probability", 1, 1, "H", "W");
        byte[] reduce = Node(
            ["image"],
            ["text_logits"],
            "detection_channel_mean",
            "ReduceMean",
            [AttributeInts("axes", 1), AttributeInt("keepdims", 1)]);
        byte[] sigmoid = Node(
            ["text_logits"],
            ["text_probability"],
            "detection_probability",
            "Sigmoid",
            []);
        return Model("generated_ocr_detection", [reduce, sigmoid], [], [input], [output]);
    }

    internal static byte[] BuildRecognitionModel(int timeSteps, int classCount)
    {
        byte[] input = ValueInfo("image", "N", 3, 48, 320);
        byte[] output = ValueInfo("ctc_logits", "N", timeSteps, classCount);
        byte[] indexZero = TensorInt64("index_zero", [], [0]);
        byte[] axesZero = TensorInt64("axes_zero", [1], [0]);
        byte[] tail = TensorInt64("tail", [2], [timeSteps, classCount]);
        byte[] shape = Node(["image"], ["input_shape"], "recognition_input_shape", "Shape", []);
        byte[] gather = Node(
            ["input_shape", "index_zero"],
            ["batch_scalar"],
            "recognition_batch_dimension",
            "Gather",
            [AttributeInt("axis", 0)]);
        byte[] unsqueeze = Node(
            ["batch_scalar", "axes_zero"],
            ["batch_vector"],
            "recognition_batch_vector",
            "Unsqueeze",
            []);
        byte[] concat = Node(
            ["batch_vector", "tail"],
            ["output_shape"],
            "recognition_output_shape",
            "Concat",
            [AttributeInt("axis", 0)]);
        byte[] constant = Node(
            ["output_shape"],
            ["ctc_logits"],
            "recognition_zero_logits",
            "ConstantOfShape",
            [AttributeTensor("value", TensorFloat(string.Empty, [1], [0f]))]);
        return Model(
            "generated_ocr_recognition",
            [shape, gather, unsqueeze, concat, constant],
            [indexZero, axesZero, tail],
            [input],
            [output]);
    }

    private static byte[] Model(
        string graphName,
        IReadOnlyList<byte[]> nodes,
        IReadOnlyList<byte[]> initializers,
        IReadOnlyList<byte[]> inputs,
        IReadOnlyList<byte[]> outputs)
    {
        byte[] graph = Message(writer =>
        {
            foreach (byte[] node in nodes)
            {
                writer.Bytes(1, node);
            }

            writer.String(2, graphName);
            foreach (byte[] initializer in initializers)
            {
                writer.Bytes(5, initializer);
            }

            foreach (byte[] input in inputs)
            {
                writer.Bytes(11, input);
            }

            foreach (byte[] output in outputs)
            {
                writer.Bytes(12, output);
            }
        });
        byte[] opset = Message(writer => writer.Varint(2, 13));
        return Message(writer =>
        {
            writer.Varint(1, 8);
            writer.String(2, "GraphReader.Integration.Tests");
            writer.String(3, "1");
            writer.Bytes(7, graph);
            writer.Bytes(8, opset);
        });
    }

    private static byte[] ValueInfo(string name, params object[] dimensions)
    {
        byte[] shape = Message(writer =>
        {
            foreach (object dimension in dimensions)
            {
                writer.Message(1, nested =>
                {
                    if (dimension is string symbol)
                    {
                        nested.String(2, symbol);
                    }
                    else
                    {
                        nested.Varint(1, checked((ulong)Convert.ToInt64(
                            dimension,
                            System.Globalization.CultureInfo.InvariantCulture)));
                    }
                });
            }
        });
        byte[] type = Message(writer => writer.Message(1, tensor =>
        {
            tensor.Varint(1, 1);
            tensor.Bytes(2, shape);
        }));
        return Message(writer =>
        {
            writer.String(1, name);
            writer.Bytes(2, type);
        });
    }

    private static byte[] Node(
        IReadOnlyList<string> inputs,
        IReadOnlyList<string> outputs,
        string name,
        string operation,
        IReadOnlyList<byte[]> attributes) =>
        Message(writer =>
        {
            foreach (string input in inputs)
            {
                writer.String(1, input);
            }

            foreach (string output in outputs)
            {
                writer.String(2, output);
            }

            writer.String(3, name);
            writer.String(4, operation);
            foreach (byte[] attribute in attributes)
            {
                writer.Bytes(5, attribute);
            }
        });

    private static byte[] AttributeInt(string name, long value) => Message(writer =>
    {
        writer.String(1, name);
        writer.Varint(3, checked((ulong)value));
        writer.Varint(20, 2);
    });

    private static byte[] AttributeInts(string name, params long[] values) => Message(writer =>
    {
        writer.String(1, name);
        foreach (long value in values)
        {
            writer.Varint(8, checked((ulong)value));
        }

        writer.Varint(20, 7);
    });

    private static byte[] AttributeTensor(string name, byte[] tensor) => Message(writer =>
    {
        writer.String(1, name);
        writer.Bytes(5, tensor);
        writer.Varint(20, 4);
    });

    private static byte[] TensorInt64(
        string name,
        IReadOnlyList<long> dimensions,
        IReadOnlyList<long> values) =>
        Message(writer =>
        {
            foreach (long dimension in dimensions)
            {
                writer.Varint(1, checked((ulong)dimension));
            }

            writer.Varint(2, 7);
            foreach (long value in values)
            {
                writer.Varint(7, checked((ulong)value));
            }

            writer.String(8, name);
        });

    private static byte[] TensorFloat(
        string name,
        IReadOnlyList<long> dimensions,
        IReadOnlyList<float> values) =>
        Message(writer =>
        {
            foreach (long dimension in dimensions)
            {
                writer.Varint(1, checked((ulong)dimension));
            }

            writer.Varint(2, 1);
            byte[] raw = new byte[checked(values.Count * sizeof(float))];
            for (int index = 0; index < values.Count; index++)
            {
                BitConverter.TryWriteBytes(raw.AsSpan(index * sizeof(float)), values[index]);
            }

            writer.String(8, name);
            writer.Bytes(9, raw);
        });

    private static byte[] Message(Action<ProtoWriter> write)
    {
        using var stream = new MemoryStream();
        write(new ProtoWriter(stream));
        return stream.ToArray();
    }

    private sealed class ProtoWriter(Stream stream)
    {
        internal void Varint(int fieldNumber, ulong value)
        {
            WriteVarint(checked((ulong)(fieldNumber << 3)));
            WriteVarint(value);
        }

        internal void Bytes(int fieldNumber, ReadOnlySpan<byte> value)
        {
            WriteVarint(checked((ulong)((fieldNumber << 3) | 2)));
            WriteVarint(checked((ulong)value.Length));
            stream.Write(value);
        }

        internal void String(int fieldNumber, string value) =>
            Bytes(fieldNumber, Encoding.UTF8.GetBytes(value));

        internal void Message(int fieldNumber, Action<ProtoWriter> write) =>
            Bytes(fieldNumber, GeneratedOcrContractOnnx.Message(write));

        private void WriteVarint(ulong value)
        {
            while (value >= 0x80)
            {
                stream.WriteByte((byte)(value | 0x80));
                value >>= 7;
            }

            stream.WriteByte((byte)value);
        }
    }
}
