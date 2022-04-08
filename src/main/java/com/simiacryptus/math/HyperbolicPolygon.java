package com.simiacryptus.math;

import com.simiacryptus.mindseye.lang.Result;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.TensorList;
import com.simiacryptus.mindseye.layers.java.ImgIndexMapViewLayer;
import com.simiacryptus.mindseye.util.ImageUtil;
import org.apache.commons.math3.util.FastMath;
import org.jetbrains.annotations.NotNull;

import java.awt.image.BufferedImage;
import java.util.*;
import java.util.function.DoubleUnaryOperator;
import java.util.function.UnaryOperator;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * This class represents a hyperbolic polygon.
 *
 * @param vertices  an array of points that represent the vertices of the polygon
 * @param edges     an array of PoincareCircles that represent the edges of the polygon
 * @param edgeStats a map of PoincareCircles to DoubleSummaryStatistics that represents statistics about the edges of the polygon
 * @param xBounds   a DoubleSummaryStatistics object that represents the x-coordinate bounds of the polygon
 * @param yBounds   a DoubleSummaryStatistics object that represents the y-coordinate bounds of the polygon
 * @docgenVersion 9
 */
public class HyperbolicPolygon {
  public final Point[] vertices;
  public final PoincareCircle[] edges;
  public final Map<PoincareCircle, DoubleSummaryStatistics> edgeStats = new HashMap<>();
  private final DoubleSummaryStatistics xBounds;
  private final DoubleSummaryStatistics yBounds;

  public HyperbolicPolygon(Point[] vertices, PoincareCircle[] edges) {
    this.xBounds = Arrays.stream(vertices).mapToDouble(x -> x.x).summaryStatistics();
    this.yBounds = Arrays.stream(vertices).mapToDouble(x -> x.y).summaryStatistics();
    this.vertices = Arrays.stream(vertices).toArray(Point[]::new);//.collect(Collectors.toSet());
    this.edges = Arrays.stream(edges).toArray(PoincareCircle[]::new);//.collect(Collectors.toSet());
    for (PoincareCircle edge : edges) {
      edgeStats.put(edge, Arrays.stream(this.vertices)
          .mapToDouble(x -> edge.euclideanDistFromCenter(x))
          .summaryStatistics());
    }
    assert vertices.length == edges.length;
    //DoubleSummaryStatistics angleStats = IntStream.range(0, edges.length).mapToDouble(i -> edges[i].angle(edges[(i + 1) % edges.length])).summaryStatistics();
    //assert (angleStats.getMax() - angleStats.getMin()) < Geometry.SPACIAL_CHECKS;
    for (int i = 0; i < edges.length; i++) {
      int j = (i + 1) % edges.length;
      //assert !(edges[i].euclideanDistFromCircle(vertices[j]) > Geometry.SPACIAL_CHECKS);
      //assert !(edges[i].euclideanDistFromCircle(vertices[i]) > Geometry.SPACIAL_CHECKS);
    }
  }

  public HyperbolicPolygon(Point[] vertices) {
    this(vertices, getEdges(vertices));
  }

  /**
   * Returns an array of PoincareCircles that represent the edges of the given
   * set of points.
   *
   * @param vertices the set of points
   * @return an array of PoincareCircles that represent the edges of the given
   * set of points
   * @docgenVersion 9
   */
  @NotNull
  public static PoincareCircle[] getEdges(Point[] vertices) {
    Circle frame = new Circle(new Point(
        Arrays.stream(vertices).mapToDouble(x -> x.x).average().getAsDouble(),
        Arrays.stream(vertices).mapToDouble(x -> x.y).average().getAsDouble()
    ), 1);
    Point[] sorted = Arrays.stream(vertices)
        //.sorted(Comparator.comparing(x -> frame.theta(x)))
        .toArray(Point[]::new);
    return IntStream.range(0, sorted.length)
        .mapToObj(i -> PoincareCircle.intersecting(sorted[i], sorted[(i + 1) % sorted.length]).asPoincareCircle())
        .toArray(PoincareCircle[]::new);
  }

  /**
   * Returns a regular hyperbolic polygon with p sides and q angles.
   *
   * @docgenVersion 9
   */
  public static HyperbolicPolygon regularPolygon(int p, int q) {
    double sweep_angle = 2 * FastMath.PI / p;
    double interior_angle = 2 * FastMath.PI / q;
    double projected_radius = FastMath.pow((1 - FastMath.pow(FastMath.sin(sweep_angle / 2) / FastMath.sin(((FastMath.PI / 2) + (interior_angle / 2))), 2)), -0.5);
    double poly_radius = projected_radius * FastMath.sin((FastMath.PI - (sweep_angle + interior_angle)) / 2) / FastMath.sin((FastMath.PI + interior_angle) / 2);
    return new HyperbolicPolygon(
        IntStream.range(0, p).mapToObj(n -> new Point(
            poly_radius * FastMath.cos(2 * FastMath.PI * n / p),
            poly_radius * FastMath.sin(2 * FastMath.PI * n / p)
        )).toArray(Point[]::new),
        IntStream.range(0, p).mapToObj(n -> new PoincareCircle(new Point(
            projected_radius * FastMath.cos(2 * FastMath.PI * (n + 0.5) / p),
            projected_radius * FastMath.sin(2 * FastMath.PI * (n + 0.5) / p)
        ))).toArray(PoincareCircle[]::new)
    );
  }

  /**
   * @param pixelCoords an array of points representing pixel coordinates
   * @return an array of points filtered by some criteria
   * @docgenVersion 9
   */
  Point[] filter(Point[] pixelCoords) {
    for (PoincareCircle edge : this.edges) {
      DoubleSummaryStatistics distBounds = this.edgeStats.get(edge);
      pixelCoords = Arrays.stream(pixelCoords).filter(point -> {
        double distFromCenter = edge.euclideanDistFromCenter(point);
        if (distBounds.getMin() > distFromCenter) return false;
        if (distBounds.getMax() < distFromCenter) return false;
        return true;
      }).toArray(Point[]::new);
    }
    return pixelCoords;
  }

  /**
   * Returns true if the given point is contained within this shape.
   *
   * @param point the point to check
   * @return true if the point is contained, false otherwise
   * @docgenVersion 9
   */
  public boolean contains(Point point) {
    for (PoincareCircle edge : this.edges) {
      DoubleSummaryStatistics distBounds = this.edgeStats.get(edge);
      double distFromCenter = edge.euclideanDistFromCenter(point);
      if (distBounds.getMin() > distFromCenter) return false;
      if (distBounds.getMax() < distFromCenter) return false;
    }
    return true;
  }

  /**
   * Reflects this hyperbolic polygon across the given Poincare circle.
   *
   * @param reflectionSurface the Poincare circle to reflect across
   * @return the reflected hyperbolic polygon
   * @docgenVersion 9
   */
  public HyperbolicPolygon reflect(PoincareCircle reflectionSurface) {
    return new HyperbolicPolygon(Arrays.stream(this.vertices).map(reflectionSurface::reflect).toArray(Point[]::new));
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;
    HyperbolicPolygon that = (HyperbolicPolygon) o;
    return Arrays.stream(vertices).collect(Collectors.toSet()).equals(Arrays.stream(that.vertices).collect(Collectors.toSet()));
  }

  @Override
  public int hashCode() {
    return Arrays.hashCode(vertices);
  }

  /**
   * @param image       the image to process
   * @param supersample the amount to supersample by
   * @return the processed image
   * @docgenVersion 9
   */
  @NotNull
  public BufferedImage process_poincare(BufferedImage image, int supersample) {
    int width = image.getWidth();
    Raster raster = new Raster(width * supersample, width * supersample);
    HyperbolicTiling tiling = new HyperbolicTiling(this).expand(3);
    BufferedImage paint = raster.getImage();
    int[] pixelMap = tiling.buildPixelMap(paint, raster);
    ImgIndexMapViewLayer layer = new ImgIndexMapViewLayer(raster, pixelMap);
    Result eval = layer.eval(Tensor.fromRGB(raster.resize(image)));
    layer.freeRef();
    TensorList tensorList = eval.getData();
    eval.freeRef();
    Tensor tensor = tensorList.get(0);
    tensorList.freeRef();
    return ImageUtil.resize(Tensor.toRgbImage(tensor), width, width);
  }

  /**
   * @param image       the image to process
   * @param supersample the amount to supersample the image
   * @param zoom        the zoom level
   * @return the processed image
   * @docgenVersion 9
   */
  @NotNull
  public BufferedImage process_poincare_zoom(BufferedImage image, int supersample, DoubleUnaryOperator zoom) {
    int width = image.getWidth();
    Raster raster = new Raster(width * supersample, width * supersample);
    HyperbolicTiling tiling = new HyperbolicTiling(this).expand(3);
    int[] pixelMap = raster.buildPixelMap(point -> {
      double r = point.rms();
      return tiling.transform(r == 0 ? point : point.scale(zoom.applyAsDouble(r) / r));
    });
    ImgIndexMapViewLayer layer = new ImgIndexMapViewLayer(raster, pixelMap);
    Result eval = layer.eval(Tensor.fromRGB(raster.resize(image)));
    layer.freeRef();
    TensorList tensorList = eval.getData();
    eval.freeRef();
    Tensor tensor = tensorList.get(0);
    tensorList.freeRef();
    return ImageUtil.resize(Tensor.toRgbImage(tensor), width, width);
  }

  /**
   * @param image         the image to process
   * @param supersample   the supersample rate
   * @param magnification the magnification
   * @return the processed image
   * @docgenVersion 9
   */
  @NotNull
  public BufferedImage process_klien(BufferedImage image, int supersample, double magnification) {
    int width = (int) (image.getWidth() * magnification);
    image = ImageUtil.resize(image, width, width);
    Raster raster = new Raster(width * supersample, width * supersample);
    UnaryOperator<Point> transform = new HyperbolicTiling(this).expand(3).klien();
    ImgIndexMapViewLayer layer = raster.toLayer(transform);
    Result eval = layer.eval(Tensor.fromRGB(raster.resize(image)));
    layer.freeRef();
    TensorList tensorList = eval.getData();
    eval.freeRef();
    Tensor tensor = tensorList.get(0);
    tensorList.freeRef();
    return ImageUtil.resize(Tensor.toRgbImage(tensor), width, width);
  }
}
