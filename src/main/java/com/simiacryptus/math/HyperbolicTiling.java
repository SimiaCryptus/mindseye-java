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

    public UnaryOperator<Point> klien() {
        return pt -> {
            if(pt != null) pt = pt.scale(1/(1+Math.sqrt(1-Math.pow(pt.rms(), 2))));
            //if (pt != null) pt = pt.scale(2 / (1 + pt.sumSq()));
            if (pt != null) pt = transform(pt);
            return pt;
        };
    }

    public HyperbolicTiling expand(int iterations) {
        tesselate(center, iterations);
        return this;
    }


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
        if(null != writableRaster) {
            raster.paint(pixels, new int[]{255, 255, 255}, writableRaster);
        }
        raster.allPixels().parallel().forEach(sourcePoint -> {
            fillPoint(raster, pixelMap, sourcePoint, writableRaster);
        });
        assert Arrays.stream(pixelMap).allMatch(x -> x != -2);
        return pixelMap;
    }

    void fillPoint(Raster raster, int[] pixelMap, Point sourcePoint, WritableRaster writableRaster) {
        int sourceIndex = raster.toIndex(raster.convertCoords(sourcePoint));
        if (sourcePoint.rms() < 1) {
            Tile tile = getTile(sourcePoint);
            Point destPt = transform(tile, sourcePoint);
            if (destPt != null) {
                int targetIndex = raster.toIndex(raster.convertCoords(destPt));
                int[] color = tile == null ? new int[]{126, 126, 126} : tile.color;
                if(null != writableRaster) raster.paint(new int[]{sourceIndex}, color, writableRaster);
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

    @Nullable
    public Point transform(Point point) {
        return transform(getTile(point), point);
    }

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
