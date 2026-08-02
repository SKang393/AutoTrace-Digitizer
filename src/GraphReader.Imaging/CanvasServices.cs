// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

namespace GraphReader.Imaging;

public readonly record struct CanvasViewState
{
    public CanvasViewState(double zoom, double offsetXDip, double offsetYDip, double dpiScaleX, double dpiScaleY)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(zoom);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(dpiScaleX);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(dpiScaleY);

        Zoom = zoom;
        OffsetXDip = offsetXDip;
        OffsetYDip = offsetYDip;
        DpiScaleX = dpiScaleX;
        DpiScaleY = dpiScaleY;
    }

    public double Zoom { get; }

    public double OffsetXDip { get; }

    public double OffsetYDip { get; }

    public double DpiScaleX { get; }

    public double DpiScaleY { get; }
}

public static class CanvasCoordinateMapper
{
    public static PixelPoint ImageToDip(PixelPoint imagePoint, CanvasViewState view) =>
        new(
            (imagePoint.X * view.Zoom) + view.OffsetXDip,
            (imagePoint.Y * view.Zoom) + view.OffsetYDip);

    public static PixelPoint DipToImage(PixelPoint dipPoint, CanvasViewState view) =>
        new(
            (dipPoint.X - view.OffsetXDip) / view.Zoom,
            (dipPoint.Y - view.OffsetYDip) / view.Zoom);

    public static PixelPoint ImageToDevicePixels(PixelPoint imagePoint, CanvasViewState view)
    {
        PixelPoint dip = ImageToDip(imagePoint, view);
        return new PixelPoint(dip.X * view.DpiScaleX, dip.Y * view.DpiScaleY);
    }

    public static PixelPoint DevicePixelsToImage(PixelPoint devicePoint, CanvasViewState view) =>
        DipToImage(
            new PixelPoint(devicePoint.X / view.DpiScaleX, devicePoint.Y / view.DpiScaleY),
            view);
}

public readonly record struct CanvasTile(int Column, int Row, int X, int Y, int Width, int Height);

public interface ICanvasTileProvider
{
    ValueTask<ReadOnlyMemory<byte>> GetTileAsync(CanvasTile tile, CancellationToken cancellationToken);
}

public sealed class TiledCanvasImageSource
{
    private readonly ICanvasTileProvider tileProvider;

    public TiledCanvasImageSource(int imageWidth, int imageHeight, int tileSize, ICanvasTileProvider tileProvider)
    {
        ArgumentNullException.ThrowIfNull(tileProvider);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(imageWidth);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(imageHeight);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(tileSize);

        ImageWidth = imageWidth;
        ImageHeight = imageHeight;
        TileSize = tileSize;
        this.tileProvider = tileProvider;
    }

    public int ImageWidth { get; }

    public int ImageHeight { get; }

    public int TileSize { get; }

    public IReadOnlyList<CanvasTile> GetVisibleTiles(PixelRect visibleImageBounds)
    {
        double left = Math.Max(0, visibleImageBounds.X);
        double top = Math.Max(0, visibleImageBounds.Y);
        double right = Math.Min(ImageWidth, visibleImageBounds.X + visibleImageBounds.Width);
        double bottom = Math.Min(ImageHeight, visibleImageBounds.Y + visibleImageBounds.Height);
        if (right <= left || bottom <= top)
        {
            return Array.Empty<CanvasTile>();
        }

        int firstColumn = (int)Math.Floor(left / TileSize);
        int firstRow = (int)Math.Floor(top / TileSize);
        int lastColumn = Math.Min((ImageWidth - 1) / TileSize, (int)Math.Floor(Math.BitDecrement(right) / TileSize));
        int lastRow = Math.Min((ImageHeight - 1) / TileSize, (int)Math.Floor(Math.BitDecrement(bottom) / TileSize));

        var tiles = new List<CanvasTile>((lastColumn - firstColumn + 1) * (lastRow - firstRow + 1));
        for (int row = firstRow; row <= lastRow; row++)
        {
            for (int column = firstColumn; column <= lastColumn; column++)
            {
                int x = column * TileSize;
                int y = row * TileSize;
                tiles.Add(new CanvasTile(
                    column,
                    row,
                    x,
                    y,
                    Math.Min(TileSize, ImageWidth - x),
                    Math.Min(TileSize, ImageHeight - y)));
            }
        }

        return tiles.AsReadOnly();
    }

    public ValueTask<ReadOnlyMemory<byte>> GetTileAsync(CanvasTile tile, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        ValidateTile(tile);
        return tileProvider.GetTileAsync(tile, cancellationToken);
    }

    private void ValidateTile(CanvasTile tile)
    {
        int columnCount = ((ImageWidth - 1) / TileSize) + 1;
        int rowCount = ((ImageHeight - 1) / TileSize) + 1;
        if (tile.Column < 0 || tile.Column >= columnCount || tile.Row < 0 || tile.Row >= rowCount)
        {
            throw new ArgumentOutOfRangeException(nameof(tile), "Tile grid coordinates are outside the image.");
        }

        int expectedX = tile.Column * TileSize;
        int expectedY = tile.Row * TileSize;
        int expectedWidth = Math.Min(TileSize, ImageWidth - expectedX);
        int expectedHeight = Math.Min(TileSize, ImageHeight - expectedY);
        if (tile.X != expectedX || tile.Y != expectedY || tile.Width != expectedWidth || tile.Height != expectedHeight)
        {
            throw new ArgumentException("Tile bounds do not match its grid coordinates and source dimensions.", nameof(tile));
        }
    }
}

public sealed class FakeCanvasTileProvider : ICanvasTileProvider
{
    public ValueTask<ReadOnlyMemory<byte>> GetTileAsync(CanvasTile tile, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        byte[] marker = [checked((byte)tile.Column), checked((byte)tile.Row)];
        return ValueTask.FromResult<ReadOnlyMemory<byte>>(marker);
    }
}
