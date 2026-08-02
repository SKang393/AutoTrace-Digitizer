// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Security.Cryptography;
using System.Text;

namespace GraphReader.Inference.Tests;

internal sealed class TestOnnxModel : IDisposable
{
    private TestOnnxModel(string directory, string path, string sha256)
    {
        Directory = directory;
        Path = path;
        Sha256 = sha256;
    }

    public string Directory { get; }

    public string Path { get; }

    public string Sha256 { get; }

    public static TestOnnxModel CreateIdentity()
    {
        var directory = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "GraphReaderInferenceTests", Guid.NewGuid().ToString("N"));
        System.IO.Directory.CreateDirectory(directory);
        var path = System.IO.Path.Combine(directory, "identity.onnx");
        var bytes = BuildIdentityModel();
        File.WriteAllBytes(path, bytes);
        return new TestOnnxModel(directory, path, Convert.ToHexString(SHA256.HashData(bytes)));
    }

    public void Dispose()
    {
        try
        {
            System.IO.Directory.Delete(Directory, recursive: true);
        }
        catch (IOException)
        {
        }
    }

    private static byte[] BuildIdentityModel()
    {
        var shape = Message(writer =>
        {
            writer.Message(1, dimension => dimension.Varint(1, 1));
            writer.Message(1, dimension => dimension.Varint(1, 3));
        });
        var type = Message(writer => writer.Message(1, tensor =>
        {
            tensor.Varint(1, 1); // TensorProto.FLOAT
            tensor.Bytes(2, shape);
        }));
        var input = Message(writer =>
        {
            writer.String(1, "x");
            writer.Bytes(2, type);
        });
        var output = Message(writer =>
        {
            writer.String(1, "y");
            writer.Bytes(2, type);
        });
        var node = Message(writer =>
        {
            writer.String(1, "x");
            writer.String(2, "y");
            writer.String(3, "identity_node");
            writer.String(4, "Identity");
        });
        var graph = Message(writer =>
        {
            writer.Bytes(1, node);
            writer.String(2, "deterministic_identity_graph");
            writer.Bytes(11, input);
            writer.Bytes(12, output);
        });
        var opset = Message(writer => writer.Varint(2, 13));
        return Message(writer =>
        {
            writer.Varint(1, 8); // ONNX IR version 8
            writer.String(2, "GraphReader.Tests");
            writer.String(3, "1");
            writer.Bytes(7, graph);
            writer.Bytes(8, opset);
        });
    }

    private static byte[] Message(Action<ProtoWriter> write)
    {
        using var stream = new MemoryStream();
        write(new ProtoWriter(stream));
        return stream.ToArray();
    }

    private sealed class ProtoWriter
    {
        private readonly Stream _stream;

        public ProtoWriter(Stream stream) => _stream = stream;

        public void Varint(int fieldNumber, ulong value)
        {
            WriteVarint(checked((ulong)(fieldNumber << 3)));
            WriteVarint(value);
        }

        public void Bytes(int fieldNumber, ReadOnlySpan<byte> value)
        {
            WriteVarint(checked((ulong)((fieldNumber << 3) | 2)));
            WriteVarint(checked((ulong)value.Length));
            _stream.Write(value);
        }

        public void String(int fieldNumber, string value) => Bytes(fieldNumber, Encoding.UTF8.GetBytes(value));

        public void Message(int fieldNumber, Action<ProtoWriter> write) => Bytes(fieldNumber, TestOnnxModel.Message(write));

        private void WriteVarint(ulong value)
        {
            while (value >= 0x80)
            {
                _stream.WriteByte((byte)(value | 0x80));
                value >>= 7;
            }

            _stream.WriteByte((byte)value);
        }
    }
}
