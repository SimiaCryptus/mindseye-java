package com.simiacryptus.math;

import com.simiacryptus.mindseye.layers.java.ImgIndexMapViewLayer;
import com.simiacryptus.mindseye.util.ImageUtil;
import org.jetbrains.annotations.NotNull;

import javax.annotation.Nonnull;
import java.awt.image.BufferedImage;
import java.awt.image.WritableRaster;
import java.util.Arrays;
import java.util.List;
import java.util.function.UnaryOperator;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * This class represents a raster.
 *
 * @param sizeX The size in the x direction.
 * @param sizeY The size in the y direction.
 * @docgenVersion 9
 */
public class Raster {
  public final int sizeX;
  public final int sizeY;
  private boolean filterCircle = true;

  public Raster(int sizeX, int sizeY) {
    this.sizeX = sizeX;
    this.sizeY = sizeY;
  }

  /**
   * Resizes an image.
   *
   * @param image the image to resize
   * @return the resized image
   * @docgenVersion 9
   */
  public @Nonnull BufferedImage resize(BufferedImage image) {
    return ImageUtil.resize(image, this.sizeX, this.sizeY);
  }

  /**
   * Returns a new ImgIndexMapViewLayer with the given transformation applied.
   *
   * @param transform the transformation to apply
   * @return a new ImgIndexMapViewLayer
   * @docgenVersion 9
   */
  @NotNull
  public ImgIndexMapViewLayer toLayer(UnaryOperator<Point> transform) {
    return new ImgIndexMapViewLayer(this, buildPixelMap(transform));
  }

  /**
   * Builds a pixel map.
   *
   * @param transform the transform to apply
   * @return the pixel map
   * @throws NullPointerException if transform is null
   * @docgenVersion 9
   */
  @NotNull
  public int[] buildPixelMap(UnaryOperator<Point> transform) {
    int[] pixelMap = new int[this.sizeX * this.sizeY];
    Arrays.fill(pixelMap, -2);
    allPixels().parallel().forEach(sourcePoint -> {
      int sourceIndex = toIndex(convertCoords(sourcePoint));
      pixelMap[sourceIndex] = -1;
    });
    assert Arrays.stream(pixelMap).allMatch(x -> x != -2);
    for (int pixel = 0; pixel < pixelMap.length; pixel++) pixelMap[pixel] = pixel;
    allPixels().parallel().forEach(sourcePoint -> {
      fillPixel(pixelMap, sourcePoint, transform);
    });
    return pixelMap;
  }

  private void fillPixel(int[] pixelMap, Point sourcePoint, UnaryOperator<Point> transform) {
    int sourceIndex = toIndex(convertCoords(sourcePoint));
    if (sourcePoint.rms() < 1 || !isFilterCircle()) {
      Point destPt = transform.apply(sourcePoint);
      if (destPt != null) {
        int targetIndex = toIndex(convertCoords(destPt));
        pixelMap[sourceIndex] = targetIndex;
      } else {
        pixelMap[sourceIndex] = -1;
      }
    } else {
      pixelMap[sourceIndex] = -1;
    }
  }

  /**
   * @param pixelMap the array of pixels to view
   * @param image    the image to view the array of pixels in
   * @return the image with the array of pixels viewed
   * @docgenVersion 9
   */
  @NotNull
  public BufferedImage view(int[] pixelMap, BufferedImage image) {
    BufferedImage imgview = getImage();
    IntStream.range(0, pixelMap.length).forEach(i -> {
      int j = pixelMap[i];
      if (j >= 0) {
        int[] xy1 = fromIndex(i);
        int[] xy2 = fromIndex(j);
        int[] buffer = new int[3];
        image.getRaster().getPixel(xy2[0], xy2[1], buffer);
        imgview.getRaster().setPixel(xy1[0], xy1[1], buffer);
      }
    });
    return imgview;
  }

  /**
   * @NotNull public BufferedImage paint(int[] pixels, int[] color) {
   * BufferedImage image = getImage();
   * paint(pixels, color, image.getRaster());
   * return image;
   * }
   * @docgenVersion 9
   */
  @NotNull
  public BufferedImage paint(int[] pixels, int[] color) {
    BufferedImage image = getImage();
    paint(pixels, color, image.getRaster());
    return image;
  }

  /**
   * Returns a new BufferedImage with dimensions sizeX and sizeY.
   *
   * @return a new BufferedImage
   * @docgenVersion 9
   */
  @NotNull
  public BufferedImage getImage() {
    return new BufferedImage(this.sizeX, this.sizeY, BufferedImage.TYPE_INT_RGB);
  }

  /**
   * @param pixels the array of pixels to be filled in
   * @param color  the color to use
   * @param raster the raster to use
   * @docgenVersion 9
   */
  void paint(int[] pixels, int[] color, WritableRaster raster) {
    int[] buffer = new int[3];
    List<int[]> pixelsToWrite = Arrays.stream(pixels).mapToObj(this::fromIndex).filter(xy -> {
      raster.getPixel(xy[0], xy[1], buffer);
      return Arrays.stream(buffer).sum() == 0;
    }).collect(Collectors.toList());
    pixelsToWrite.forEach(xy -> {
      raster.setPixel(xy[0], xy[1], color);
    });
  }

  /**
   * Converts the given xy coordinates into a Point.
   *
   * @param xy the xy coordinates to convert
   * @return the Point corresponding to the given xy coordinates
   * @docgenVersion 9
   */
  @NotNull
  public Point convertCoords(int[] xy) {
    Point point = new Point(
        (2.0 * xy[0] / sizeX) - 1,
        (2.0 * xy[1] / sizeY) - 1
    );
    assert Arrays.equals(xy, convertCoords(point));
    return point;
  }

  /**
   * Converts the given Point's x and y coordinates into an int array.
   *
   * @param xy the Point to convert
   * @return an array containing the Point's x and y coordinates
   * @throws NullPointerException if xy is null
   * @docgenVersion 9
   */
  @NotNull
  public int[] convertCoords(Point xy) {
//        assert xy.x >= -1;
//        assert xy.x <= 1;
//        assert xy.y >= -1;
//        assert xy.y <= 1;
    int[] ints = {
        Math.max(0, Math.min(sizeX - 1, (int) Math.round((sizeX / 2.0) * (xy.x + 1)))),
        Math.max(0, Math.min(sizeY - 1, (int) Math.round((sizeY / 2.0) * (xy.y + 1))))
    };
    return ints;
  }

  /**
   * Returns a stream of all the pixels in the image.
   *
   * @return a stream of all the pixels in the image
   * @docgenVersion 9
   */
  @NotNull
  public Stream<Point> allPixels() {
    return IntStream.range(0, sizeX * sizeY).mapToObj(this::fromIndex).map(this::convertCoords);
  }

  /**
   * Returns the index of the specified coordinates.
   *
   * @param xy the coordinates
   * @return the index of the specified coordinates
   * @docgenVersion 9
   */
  public int toIndex(int... xy) {
    if (xy[0] < 0) {
      throw new AssertionError();
    }
    assert xy[1] >= 0;
    assert xy[0] < sizeX;
    assert xy[1] < sizeY;
    int index = xy[0] * sizeY + xy[1];
    assert index >= 0;
    assert index < (sizeX * sizeY);
    return index;
  }

  /**
   * Returns an array of ints starting from the given index.
   *
   * @param i the starting index
   * @return an array of ints
   * @docgenVersion 9
   */
  public int[] fromIndex(int i) {
    if (i < 0) throw new IllegalArgumentException();
    if (i >= sizeX * sizeY) throw new IllegalArgumentException();
    int[] ints = {i / sizeY, i % sizeY};
    assert toIndex(ints) == i : String.format("%d -> %s -> %d", i, Arrays.toString(ints), toIndex(ints));
    return ints;
  }

  /**
   * Returns an array of all the pixels in a given hyperbolic polygon.
   *
   * @param polygon the hyperbolic polygon to find the pixels of
   * @return an array of all the pixels in the given hyperbolic polygon
   * @docgenVersion 9
   */
  public int[] pixels(HyperbolicPolygon polygon) {
    Point[] allpixels = allPixels().toArray(Point[]::new);
    Point[] filtered = polygon.filter(allpixels);
    return Arrays.stream(filtered)
        .map(this::convertCoords)
        .mapToInt(this::toIndex)
        .distinct().sorted().toArray();
  }

  /**
   * Returns true if the filter is a circle, false otherwise.
   *
   * @docgenVersion 9
   */
  public boolean isFilterCircle() {
    return filterCircle;
  }

  /**
   * Sets the filterCircle boolean to the specified value.
   *
   * @param filterCircle the value to set the boolean to
   * @return the Raster object, for chaining purposes
   * @docgenVersion 9
   */
  public Raster setFilterCircle(boolean filterCircle) {
    this.filterCircle = filterCircle;
    return this;
  }
}
