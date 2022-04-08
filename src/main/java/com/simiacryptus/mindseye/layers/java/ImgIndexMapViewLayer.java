package com.simiacryptus.mindseye.layers.java;

import com.simiacryptus.math.Point;
import com.simiacryptus.math.Raster;

import javax.annotation.Nonnull;
import java.util.Arrays;

/**
 * This class represents a layer of an image index map view.
 *
 * @param pixelMap An array of integers representing the pixels in the layer.
 * @param sizeX    The width of the layer.
 * @param sizeY    The height of the layer.
 * @docgenVersion 9
 */
public class ImgIndexMapViewLayer extends ImgViewLayerBase {
  private final int[] pixelMap;
  private final int sizeX;
  private final int sizeY;

  public ImgIndexMapViewLayer(Raster raster, int[] pixelMap) {
    this.sizeX = raster.sizeX;
    this.sizeY = raster.sizeY;
    this.pixelMap = pixelMap;
    assert Arrays.stream(this.pixelMap).allMatch(x -> x >= -1);
    assert Arrays.stream(this.pixelMap).allMatch(x -> x < sizeX * sizeY);
  }

  @Override
  protected Point coordinateMapping(@Nonnull Point xy) {
    final Raster raster = new Raster(sizeX, sizeY);
    int i = raster.toIndex((int) xy.x, (int) xy.y);
    int j = pixelMap[i];
    if (j < 0) return new Point(0.0, 0.0);
    int[] dest = raster.fromIndex(j);
    return new Point(dest[0], dest[1]);
  }

  @Override
  public void _free() {
    super._free();
  }
}
