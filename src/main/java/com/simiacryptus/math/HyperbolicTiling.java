package com.simiacryptus.math;

import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import javax.annotation.Nonnull;
import java.awt.image.BufferedImage;
import java.awt.image.WritableRaster;
import java.util.*;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.UnaryOperator;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * This class represents a hyperbolic tiling.
 *
 * @docgenVersion 9
 */
public class HyperbolicTiling {
  public static final double INDEX_TOL = 1e-8;

  public final Set<Tile> tiles = new HashSet<>();
  public final Set<HyperbolicPolygon> polys = new HashSet<>();
  public final List<PoincareCircle> edges = new ArrayList<>();
  public final List<Point> points = new ArrayList<>();
  private final Tile center;

  public HyperbolicTiling(HyperbolicPolygon center) {
    polys.add(center);
    this.center = new Tile(center, null, null);
  }

  /**
   * Returns a unary operator that transforms a point according to the Klien formula.
   *
   * @return a unary operator that transforms a point according to the Klien formula
   * @docgenVersion 9
   */
  public UnaryOperator<Point> klien() {
    return pt -> {
      if (pt != null) pt = pt.scale(1 / (1 + Math.sqrt(1 - Math.pow(pt.rms(), 2))));
      if (pt != null) pt = transform(pt);
      return pt;
    };
  }

  /**
   * Returns a unary operator that squares a point and then transforms it.
   *
   * @return a unary operator that squares a point and then transforms it
   * @docgenVersion 9
   */
  public UnaryOperator<Point> square() {
    return pt -> {
      pt = square(pt);
      if (pt == null) return null;
      pt = transform(pt);
      return pt;
    };
  }

  /**
   * @Nullable public Point square(Point pt);
   * @docgenVersion 9
   */
  @Nullable
  public Point square(Point pt) {
    if (pt == null) return null;
    double max = Math.max(Math.abs(pt.x), Math.abs(pt.y));
    double min = Math.min(Math.abs(pt.x), Math.abs(pt.y));
    if (max > 0) pt = pt.scale(0.999 / Math.sqrt(1.0 + Math.pow((min / max), 2)));
    return pt;
  }

  /**
   * @Nullable public Point invsquare(Point pt);
   * @docgenVersion 9
   */
  @Nullable
  public Point invsquare(Point pt) {
    if (pt == null) return null;
    double max = Math.max(Math.abs(pt.x), Math.abs(pt.y));
    double min = Math.min(Math.abs(pt.x), Math.abs(pt.y));
    if (max > 0) pt = pt.scale(Math.sqrt(1.0 + Math.pow((min / max), 2)));
    return pt;
  }

  /**
   * Returns a unary operator that transforms a point by scaling it by 2/(1+pt.sumSq()),
   * then transforming it.
   *
   * @return the unary operator
   * @docgenVersion 9
   */
  public UnaryOperator<Point> bubble() {
    return pt -> {
      if (pt != null) pt = pt.scale(2 / (1 + pt.sumSq()));
      if (pt != null) pt = transform(pt);
      return pt;
    };
  }

  /**
   * Returns a unary operator that squares a Point's x and y values.
   *
   * @return a unary operator that squares a Point's x and y values
   * @docgenVersion 9
   */
  public UnaryOperator<Point> square2() {
    return pt -> {
      if (pt == null) return null;
      double max = Math.max(Math.abs(pt.x), Math.abs(pt.y));
      double min = Math.min(Math.abs(pt.x), Math.abs(pt.y));
      if (max > 0) pt = pt.scale(0.999 / Math.sqrt(1.0 + Math.pow((min / max), 2)));
      if (pt != null) pt = pt.scale(2 / (1 + pt.sumSq()));
      pt = transform(pt);
      return pt;
    };
  }

  /**
   * Expands the hyperbolic tiling by the specified number of iterations.
   *
   * @param iterations the number of iterations to expand by
   * @return the expanded hyperbolic tiling
   * @docgenVersion 9
   */
  public HyperbolicTiling expand(int iterations) {
    tesselate(center, iterations);
    return this;
  }


  /**
   * This class represents a tile.
   *
   * @param poly      The hyperbolic polygon for this tile.
   * @param transform The spacial transform for this tile.
   * @param parent    The parent tile for this tile.
   * @param color     The color for this tile.
   * @docgenVersion 9
   */
  public class Tile {
    public final @Nonnull HyperbolicPolygon poly;
    public final SpacialTransform transform;
    public final Tile parent;
    public final int[] color = IntStream.generate(() -> Geometry.random.nextInt(200) + 50).limit(3).toArray();

    public Tile(@Nonnull HyperbolicPolygon poly, SpacialTransform transform, Tile parent) {
      this.poly = poly;
      this.transform = transform;
      this.parent = parent;
    }
  }

  /**
   * Registers a PoincareCircle.
   *
   * @param circle the PoincareCircle to register
   * @return the registered PoincareCircle
   * @docgenVersion 9
   */
  public PoincareCircle register(PoincareCircle circle) {
    synchronized (edges) {
      Optional<PoincareCircle> closest = edges.stream().filter(x -> Geometry.dist(x.center(), circle.center()) < INDEX_TOL).findFirst();
      if (closest.isPresent()) {
        return closest.get();
      } else {
        edges.add(circle);
        return circle;
      }
    }
  }


  /**
   * Registers a point.
   *
   * @param pt the point to register
   * @return the registered point
   * @docgenVersion 9
   */
  public Point register(Point pt) {
    synchronized (points) {
      Optional<Point> closest = points.stream().filter(x -> Geometry.dist(x, pt) < INDEX_TOL).findFirst();
      if (closest.isPresent()) {
        return closest.get();
      } else {
        points.add(pt);
        return pt;
      }
    }
  }


  /**
   * Tesselates a tile.
   *
   * @param parent the parent tile
   * @param n      the number of children
   * @docgenVersion 9
   */
  void tesselate(Tile parent, int n) {
    Arrays.stream(parent.poly.edges).parallel().forEach(edge -> {
      if (edge.radius > 0) {
        HyperbolicPolygon reflected = parent.poly.reflect(edge);
        reflected = new HyperbolicPolygon(
            Arrays.stream(reflected.vertices).map(this::register).toArray(Point[]::new),
            Arrays.stream(reflected.edges).map(this::register).toArray(PoincareCircle[]::new)
        );
        Tile child = new Tile(reflected, edge.reflection, parent);
        tiles.add(child);
        if (polys.add(reflected) && n > 0) {
          tesselate(child, n - 1);
        }
      }
    });
  }

  /**
   * Builds a pixel map from a given image and raster.
   *
   * @param paint  the image to build the map from
   * @param raster the raster to use
   * @return the pixel map
   * @throws NullPointerException if either the image or raster is null
   * @docgenVersion 9
   */
  @NotNull
  public int[] buildPixelMap(BufferedImage paint, Raster raster) {
    int[] pixelMap = new int[raster.sizeX * raster.sizeY];
    Arrays.fill(pixelMap, -2);
    raster.allPixels().parallel().forEach(sourcePoint -> {
      int sourceIndex = raster.toIndex(raster.convertCoords(sourcePoint));
      pixelMap[sourceIndex] = -1;
    });
    assert Arrays.stream(pixelMap).allMatch(x -> x != -2);
    int[] pixels = raster.pixels(this.center.poly);
    for (int pixel : pixels) pixelMap[pixel] = pixel;
    WritableRaster writableRaster = (null != paint) ? paint.getRaster() : null;
    if (null != writableRaster) {
      raster.paint(pixels, new int[]{255, 255, 255}, writableRaster);
    }
    raster.allPixels().parallel().forEach(sourcePoint -> {
      fillPoint(raster, pixelMap, sourcePoint, writableRaster);
    });
    assert Arrays.stream(pixelMap).allMatch(x -> x != -2);
    return pixelMap;
  }

  /**
   * Fills a point in a raster with a pixel map.
   *
   * @param raster         the raster to fill
   * @param pixelMap       the pixel map to use
   * @param sourcePoint    the point to fill
   * @param writableRaster the writable raster to use
   * @docgenVersion 9
   */
  void fillPoint(Raster raster, int[] pixelMap, Point sourcePoint, WritableRaster writableRaster) {
    int sourceIndex = raster.toIndex(raster.convertCoords(sourcePoint));
    if (sourcePoint.rms() < 1) {
      Tile tile = getTile(sourcePoint);
      Point destPt = transform(tile, sourcePoint);
      if (destPt != null) {
        int targetIndex = raster.toIndex(raster.convertCoords(destPt));
        int[] color = tile == null ? new int[]{126, 126, 126} : tile.color;
        if (null != writableRaster) raster.paint(new int[]{sourceIndex}, color, writableRaster);
        pixelMap[sourceIndex] = targetIndex;
      } else {
        pixelMap[sourceIndex] = -1;
      }
    } else {
      pixelMap[sourceIndex] = -1;
    }
  }

  @Nullable
  private Tile getTile(Point sourcePoint) {
    return tiles.stream().filter(x -> x.poly.contains(sourcePoint)).findAny().orElse(null);
  }

  /**
   * Transforms the given point according to this {@link Tile}.
   *
   * @param point the point to transform
   * @return the transformed point, or {@code null} if the transformation resulted in an invalid point
   * @docgenVersion 9
   */
  @Nullable
  public Point transform(Point point) {
    return transform(getTile(point), point);
  }

  /**
   * @return the Point transformed or null if the transformation fails
   * @Nullable
   * @docgenVersion 9
   */
  @Nullable
  protected Point transform(Tile tile, Point point) {
    if (tile != null) {
      Point destPt = null;
      if (tile != center) {
        destPt = point;
        if (null != destPt) {
          assert destPt.x >= -1;
          assert destPt.x <= 1;
          assert destPt.y >= -1;
          assert destPt.y <= 1;
        }
        for (Tile currentTile = tile; currentTile != center; ) {
          Point prior = destPt;
          destPt = currentTile.transform.apply(prior);
          if (null != destPt) {
            assert destPt.x >= -1;
            assert destPt.x <= 1;
            assert destPt.y >= -1;
            assert destPt.y <= 1;
          }
          currentTile = currentTile.parent;
        }
      }
      return destPt;
    } else {
//            if(1==1) return null;
      try {
        int recursion = transformRecursion.get().incrementAndGet();
        if (recursion <= 10) {
          List<PoincareCircle> circleList = edges.stream().parallel().sorted(Comparator.comparing(
              (PoincareCircle x) -> x.euclideanDistFromCircle(point)
          )).limit(10).collect(Collectors.toList()).stream().parallel().sorted(Comparator.comparing(
              (PoincareCircle x) -> x.reflect(point).rms()
          )).limit(recursion <= 1 ? 1 : 3).collect(Collectors.toList());
          for (PoincareCircle circle : circleList.stream().limit(3).collect(Collectors.toList())) {
            Point reflect = transform(circle.reflect(point));
            if (null != reflect) return reflect;
          }
        }
        return null;
      } finally {
        transformRecursion.get().decrementAndGet();
      }
    }
  }

  private final ThreadLocal<AtomicInteger> transformRecursion = ThreadLocal.withInitial(() -> new AtomicInteger());

}
