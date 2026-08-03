// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Security.Cryptography;
using System.Text;

namespace GraphReader.Markers.Tests.Detection;

/// <summary>
/// Generates an identity ONNX graph for exercising the deployed marker tensor contract.
/// It is test infrastructure, not a trained marker detector or release artifact.
/// </summary>
internal sealed class GeneratedMarkerContractOnnx : IDisposable
{
    private GeneratedMarkerContractOnnx(string directory, string path, string sha256)
    {
        Directory = directory;
        Path = path;
        Sha256 = sha256;
    }

    internal string Directory { get; }

    internal string Path { get; }

    internal string Sha256 { get; }

    internal static GeneratedMarkerContractOnnx CreateIdentity(int width, int height)
    {
        string directory = System.IO.Path.Combine(
            System.IO.Path.GetTempPath(),
            "GraphReaderMarkerTests",
            Guid.NewGuid().ToString("N"));
        System.IO.Directory.CreateDirectory(directory);
        string path = System.IO.Path.Combine(directory, "marker-contract-identity.onnx");
        byte[] bytes = BuildIdentityModel(width, height);
        File.WriteAllBytes(path, bytes);
        return new GeneratedMarkerContractOnnx(
            directory,
            path,
            Convert.ToHexString(SHA256.HashData(bytes)));
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

    private static byte[] BuildIdentityModel(int width, int height)
    {
        byte[] shape = Message(writer =>
        {
            writer.Message(1, dimension => dimension.Varint(1, 1));
            writer.Message(1, dimension => dimension.Varint(1, 3));
            writer.Message(1, dimension => dimension.Varint(1, checked((ulong)height)));
            writer.Message(1, dimension => dimension.Varint(1, checked((ulong)width)));
        });
        byte[] type = Message(writer => writer.Message(1, tensor =>
        {
            tensor.Varint(1, 1);
            tensor.Bytes(2, shape);
        }));
        byte[] input = Message(writer =>
        {
            writer.String(1, "image");
            writer.Bytes(2, type);
        });
        byte[] output = Message(writer =>
        {
            writer.String(1, "heads");
            writer.Bytes(2, type);
        });
        byte[] node = Message(writer =>
        {
            writer.String(1, "image");
            writer.String(2, "heads");
            writer.String(3, "marker_contract_identity");
            writer.String(4, "Identity");
        });
        byte[] graph = Message(writer =>
        {
            writer.Bytes(1, node);
            writer.String(2, "generated_marker_contract_graph");
            writer.Bytes(11, input);
            writer.Bytes(12, output);
        });
        byte[] opset = Message(writer => writer.Varint(2, 13));
        return Message(writer =>
        {
            writer.Varint(1, 8);
            writer.String(2, "GraphReader.Markers.Tests");
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

        internal ProtoWriter(Stream stream) => _stream = stream;

        internal void Varint(int fieldNumber, ulong value)
        {
            WriteVarint(checked((ulong)(fieldNumber << 3)));
            WriteVarint(value);
        }

        internal void Bytes(int fieldNumber, ReadOnlySpan<byte> value)
        {
            WriteVarint(checked((ulong)((fieldNumber << 3) | 2)));
            WriteVarint(checked((ulong)value.Length));
            _stream.Write(value);
        }

        internal void String(int fieldNumber, string value) =>
            Bytes(fieldNumber, Encoding.UTF8.GetBytes(value));

        internal void Message(int fieldNumber, Action<ProtoWriter> write) =>
            Bytes(fieldNumber, GeneratedMarkerContractOnnx.Message(write));

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
