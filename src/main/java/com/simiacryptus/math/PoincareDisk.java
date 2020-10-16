package com.simiacryptus.math;

import org.apache.commons.math3.analysis.UnivariateFunction;
import org.apache.commons.math3.exception.TooManyEvaluationsException;
import org.apache.commons.math3.optim.MaxEval;
import org.apache.commons.math3.optim.nonlinear.scalar.GoalType;
import org.apache.commons.math3.optim.univariate.BrentOptimizer;
import org.apache.commons.math3.optim.univariate.SearchInterval;
import org.apache.commons.math3.optim.univariate.UnivariateObjectiveFunction;
import org.apache.commons.math3.optim.univariate.UnivariatePointValuePair;
import org.apache.commons.math3.util.FastMath;
import org.jetbrains.annotations.NotNull;

import java.awt.image.BufferedImage;
import java.awt.image.WritableRaster;
import java.util.*;
import java.util.stream.*;

public class PoincareDisk {

    public static final Circle UNIT_CIRCLE = new Circle(new double[]{0, 0}, 1);

    public static final double TOL = 1e-12;
    public static final double SPACIAL_CHECKS = 1e-4;
    private static final Random random = new Random();
    public static final double SPACIAL_INF = 1e8;

    public static class Raster {
        public final int sizeX;
        public final int sizeY;

        public Raster(int sizeX, int sizeY) {
            this.sizeX = sizeX;
            this.sizeY = sizeY;
        }

        private void paintall(Polygon polygon, BufferedImage paint, int n, int[] pixelMap) {
            Arrays.stream(polygon.edges).parallel().forEach(edge -> {
                Polygon reflected = polygon.reflect(edge);
                int[] color = IntStream.generate(() -> random.nextInt(200) + 50).limit(3).toArray();
                int[] pixels = pixels(reflected);
                if (paint(pixels, paint, color, 0.1) && n > 0) {
                    double[] reflect = new double[]{Double.NaN, Double.NaN};
                    for (int pixel : pixels) {
                        if (pixelMap[pixel] != -1) continue;
                        double[] coords = convertCoords(fromIndex(pixel));
                        edge.reflect(coords, reflect);
                        int toIndex = toIndex(convertCoords(reflect));
                        if (pixelMap[toIndex] != -1) {
                            pixelMap[pixel] = toIndex;
                            if (pixelMap[toIndex] != toIndex) {
                                toIndex = pixelMap[toIndex];
                            }
                        }
                        pixelMap[pixel] = toIndex;
                    }
                    this.paintall(reflected, paint, n - 1, pixelMap);
                }
            });
        }

        public TilingResult pixelMap(Polygon polygon, int iterations) {
            int[] pixelMap = new int[this.sizeX * this.sizeY];
            Arrays.fill(pixelMap, -1);
            int[] pixels = pixels(polygon);
            for (int pixel : pixels) pixelMap[pixel] = pixel;
            BufferedImage paint = paint(pixels);
            this.paintall(polygon, paint, iterations, pixelMap);
            TilingResult tilingResult = new TilingResult(pixelMap, paint);
            return tilingResult;
        }

        @NotNull
        public BufferedImage view(int[] pixelMap, BufferedImage image) {
            BufferedImage imgview = new BufferedImage(this.sizeX, this.sizeY, BufferedImage.TYPE_INT_RGB);
            IntStream.range(0, pixelMap.length).forEach(i -> {
                int j = pixelMap[i];
                if (j >= 0) {
                    @NotNull int[] xy1 = fromIndex(i);
                    @NotNull int[] xy2 = fromIndex(j);
                    int[] buffer = new int[3];
                    image.getRaster().getPixel(xy2[0], xy2[1], buffer);
                    imgview.getRaster().setPixel(xy1[0], xy1[1], buffer);
                }
            });
            return imgview;
        }

        @NotNull
        public BufferedImage paint(int[] pixels) {
            BufferedImage image = new BufferedImage(this.sizeX, this.sizeY, BufferedImage.TYPE_INT_RGB);
            paint(pixels, image, new int[]{255, 255, 255}, 0.5);
            return image;
        }

        public boolean paint(int[] pixels, BufferedImage image, int[] color, double minFill) {
            WritableRaster raster = image.getRaster();
            int[] buffer = new int[3];
            List<@NotNull int[]> pixelsToWrite = Arrays.stream(pixels).mapToObj(this::fromIndex).filter(xy -> {
                raster.getPixel(xy[0], xy[1], buffer);
                return Arrays.stream(buffer).sum() == 0;
//                return true;
            }).collect(Collectors.toList());
            if (pixelsToWrite.size() > (pixels.length * minFill)) {
                pixelsToWrite.forEach(xy -> {
                    raster.setPixel(xy[0], xy[1], color);
                });
                return true;
            } else {
                return false;
            }
        }

        @NotNull
        public Stream<int[]> toPixels(Stream<double[]> stream) {
            return stream
                    .map(this::convertCoords)
                    .mapToInt(this::toIndex)
                    .distinct().sorted()
                    .mapToObj(this::fromIndex);
        }

        @NotNull
        public double[] convertCoords(int[] xy) {
            return new double[]{
                    (2.0 * xy[0] / sizeX) - 1,
                    (2.0 * xy[1] / sizeY) - 1
            };
        }

        @NotNull
        public int[] convertCoords(double[] xy) {
            return new int[]{
                    (int) ((sizeX / 2) * (xy[0] + 1)),
                    (int) ((sizeY / 2) * (xy[1] + 1))
            };
        }

        @NotNull
        public int[] fromIndex(int i) {
            int[] xy = new int[2];
            fromIndex(i, xy);
            return xy;
        }

        @NotNull
        public Stream<double[]> allPixels() {
            return IntStream.range(0, sizeX * 2).mapToObj(x -> x)
                    .flatMap(x -> IntStream.range(0, sizeY * 2)
                            .mapToObj(y -> {
                                int[] xy = new int[]{x, y};
                                return new double[]{
                                        (1.0 * xy[0] / sizeX) - 1,
                                        (1.0 * xy[1] / sizeY) - 1
                                };
                            })
                            .filter(v -> rms(v) <= 1));
        }

        public int toIndex(int[] xy) {
            return xy[0] + sizeX * xy[1];
        }

        public void fromIndex(int i, int[] xy) {
            xy[0] = FastMath.max(0, FastMath.min(sizeX - 1, i % sizeX));
            xy[1] = FastMath.max(0, FastMath.min(sizeY - 1, i / sizeX));
        }

        public int[] pixels(Polygon polygon) {
            double[][] allpixels = allPixels().toArray(double[][]::new);
            double[][] filtered = polygon.filter(allpixels);
            return toPixels(Arrays.stream(filtered))
                    .mapToInt(this::toIndex)
                    .distinct().sorted().toArray();
        }
    }

    public static class Circle {
        public final double centerX;
        public final double centerY;
        public final double radius;

        public Circle(double[] centerXY) {
            this(centerXY, FastMath.sqrt((centerXY[0] * centerXY[0] + centerXY[1] * centerXY[1]) - 1));
            //assert isPerpendicular(UNIT_CIRCLE);
        }

        public Circle(double[] centerXY, double radius) {
            this.centerX = centerXY[0];
            this.centerY = centerXY[1];
            this.radius = radius;
        }

        private boolean intersects(double[] xy) {
            return euclideanDistFromCircle(xy) < SPACIAL_CHECKS;
        }

        public PerpendicularResult perpendicular(double[] xy2) {
            if (euclideanDistFromCircle(xy2) < SPACIAL_CHECKS) throw new RuntimeException();
            double[] interfacePoint = theta(perpendicular(xy2, theta(xy2)));
            return finalPerpendicularResult(xy2, interfacePoint);
        }

        public PerpendicularResult perpendicular(double[] point, Circle guess) {
            if (euclideanDistFromCircle(point) < SPACIAL_CHECKS) throw new RuntimeException();
            List<double[]> intersections = intersect(guess);
            double[] interfacePoint = intersections.stream().sorted(Comparator.comparingDouble(pt->rms(new double[] {pt[0] - point[0], pt[1] - point[1]}))).findFirst().get();
            interfacePoint = theta(perpendicular(point, theta(interfacePoint)));
            return finalPerpendicularResult(point, interfacePoint);
        }

        private PoincareDisk.PerpendicularResult finalPerpendicularResult(double[] point, double[] interfacePoint) {
            Circle circle = intersecting(point, interfacePoint);
            int attempts = 0;
            while (!isPerpendicular(circle)) {
                if (attempts++ > 100) {
                    throw new RuntimeException();
                }
                interfacePoint = theta(perpendicular(point, (FastMath.random() - 0.5) * 2 * FastMath.PI));
                circle = intersecting(point, interfacePoint);
            }
            return new PerpendicularResult(circle, interfacePoint);
        }

        private List<double[]> intersect(Circle right) {
            double r = rms(new double[]{centerX - right.centerX, centerY - right.centerY});
            double a = (radius * radius - right.radius * right.radius) / (2 * r * r);
            double b = FastMath.sqrt(2 * (radius * radius + right.radius * right.radius) / (r * r) - FastMath.pow((radius * radius - right.radius * right.radius), 2) / FastMath.pow(r,4) - 1);
            List<double[]> collect = Stream.of(new double[]{
                    0.5 * (centerX + right.centerX) + a * (right.centerX - centerX) + 0.5 * b * (right.centerY - centerY),
                    0.5 * (centerY + right.centerY) + a * (right.centerY - centerY) + 0.5 * b * (right.centerX - centerX)
            }, new double[]{
                    0.5 * (centerX + right.centerX) + a * (right.centerX - centerX) - 0.5 * b * (right.centerY - centerY),
                    0.5 * (centerY + right.centerY) + a * (right.centerY - centerY) - 0.5 * b * (right.centerX - centerX)
            }).collect(Collectors.toList());
            for (double[] pt : collect) { // Correct numerical errors
                boolean repeat = false;
                do {
                    repeat = false;
                    if (euclideanDistFromCircle(pt) >= SPACIAL_CHECKS) {
                        project(pt);
                        if (euclideanDistFromCircle(pt) >= SPACIAL_CHECKS) {
                            throw new IllegalStateException();
                        }
                        //repeat = true;
                    }
                    if (right.euclideanDistFromCircle(pt) >= SPACIAL_CHECKS) {
                        right.project(pt);
                        if (right.euclideanDistFromCircle(pt) >= SPACIAL_CHECKS) {
                            throw new IllegalStateException();
                        }
                        //repeat = true;
                    }
                } while (repeat);
            }
            return collect;
        }

        private void project(double[] pt) {
            double scale = radius / rms(new double[]{pt[0] - centerX, pt[1] - centerY});
            pt[0] = centerX + scale * (pt[0] - centerX);
            pt[1] = centerY + scale * (pt[1] - centerY);
        }

        private double perpendicular(double[] subject, double guessTheta) {
            BrentOptimizer simplexOptimizer = new BrentOptimizer(TOL, TOL);
            try {
                UnivariatePointValuePair optimize = simplexOptimizer.optimize(
                        new UnivariateObjectiveFunction(theta -> {
                            double[] th = theta(theta);
                            double rms = rms(th);
                            if (rms >= 1) return (1000 * rms);
                            Circle intersecting = intersecting(subject, th);
                            if (null == intersecting) return 1e4;
                            return FastMath.abs((FastMath.PI / 2) - angle(intersecting));
                        }),
                        new MaxEval(1000),
                        GoalType.MINIMIZE,
                        new SearchInterval(-2 * FastMath.PI, 2 * FastMath.PI, guessTheta)
                );
                return optimize.getPoint();
            } catch (TooManyEvaluationsException e) {
                return guessTheta;
            }
        }

        private boolean isPerpendicular(Circle circle) {
            return circle != null && FastMath.abs((FastMath.PI / 2) - angle(circle)) < SPACIAL_CHECKS;
        }

        public double[] theta(double optimal) {
            double[] xy = new double[2];
            theta(optimal, xy);
            return xy;
        }

        public double[] reflect(double[] subject) {
            double targetDist = euclideanDistFromCircle(subject);
            if (targetDist < SPACIAL_CHECKS) return subject;
            PerpendicularResult perp = perpendicular(subject);
            double distanceFromMirror = poincareDist(subject, perp.intersection);
            UnivariateFunction fn = theta -> {
                double[] xy = perp.edge.theta(theta);
                double rms = rms(xy);
                if (rms > 1) return rms * 1000;
                double a = FastMath.abs(distanceFromMirror - poincareDist(xy, perp.intersection));
                double b = FastMath.abs(2 * distanceFromMirror - poincareDist(xy, subject));
                return a * a + b * b;
            };
            double theta1 = perp.edge.theta(subject);
            double theta2 = perp.edge.theta(perp.intersection);
            double[] reflected = reflect(perp.edge, fn, (2 * theta2 - theta1) % FastMath.PI);
            reflected = reflect_fallback(subject, perp, distanceFromMirror, fn, reflected);
            return reflected;
        }

        public void reflect(double[] subject, double[] reflect) {
            double targetDist = euclideanDistFromCircle(subject);
            if (targetDist < SPACIAL_CHECKS) {
                reflect[0] = subject[0];
                reflect[1] = subject[1];
                return;
            }
            if (Double.isNaN(reflect[0]) || Double.isNaN(reflect[1])) {
                double[] r = reflect(subject);
                reflect[0] = r[0];
                reflect[1] = r[1];
            } else {
                PerpendicularResult perp = perpendicular(subject, intersecting(subject, reflect));
                double distanceFromMirror = poincareDist(subject, perp.intersection);
                UnivariateFunction fn = theta -> {
                    double[] xy = perp.edge.theta(theta);
                    double rms = rms(xy);
                    if (rms > 1) return rms * 1000;
                    double a = FastMath.abs(distanceFromMirror - poincareDist(xy, perp.intersection));
                    double b = FastMath.abs(2 * distanceFromMirror - poincareDist(xy, subject));
                    return a * a + b * b;
                };
                double[] reflected = reflect(perp.edge, fn, perp.edge.theta(reflect) % FastMath.PI);
                reflected = reflect_fallback(subject, perp, distanceFromMirror, fn, reflected);
                reflect[0] = reflected[0];
                reflect[1] = reflected[1];
            }
        }

        private double[] reflect_fallback(double[] subject, PerpendicularResult perp, double distanceFromMirror, UnivariateFunction fn, double[] reflected) {
            if(!perp.isReflection(distanceFromMirror, subject, reflected)) {
                reflected = reflect(perp.edge, fn, perp.edge.theta(perp.intersection));
            }
            if(!perp.isReflection(distanceFromMirror, subject, reflected)) {
                reflected = reflect(perp.edge, fn, perp.edge.theta(new double[]{ 0,0 }));
            }
            if(!perp.isReflection(distanceFromMirror, subject, reflected)) {
                for (double[] pointAtInfinity : UNIT_CIRCLE.intersect(perp.edge)) {
                    reflected = reflect(perp.edge, fn, perp.edge.theta(new double[]{
                            pointAtInfinity[0] * (1-SPACIAL_CHECKS), pointAtInfinity[1] * (1-SPACIAL_CHECKS)
                    }));
                    if(perp.isReflection(distanceFromMirror, subject, reflected)) {
                        break;
                    }
                }
            }
            int loopCount = 0;
            while (!perp.isReflection(distanceFromMirror, subject, reflected)) {
                if (loopCount++ > 100) {
                    throw new AssertionError();
                }
                reflected = reflect(perp.edge, fn, (FastMath.random() - 0.5) * 2 * FastMath.PI);
            }
            return reflected;
        }

        private double[] reflect(Circle perpCircle, UnivariateFunction fn, double guess) {
            try {
                return perpCircle.theta(new BrentOptimizer(TOL, TOL).optimize(
                        new UnivariateObjectiveFunction(fn),
                        new MaxEval(1000),
                        GoalType.MINIMIZE,
                        new SearchInterval(-FastMath.PI, FastMath.PI, guess)
                ).getPoint());
            } catch (TooManyEvaluationsException e) {
                return new double[] {Double.NaN, Double.NaN};
            }
        }

        @NotNull
        public double[] getPoincareGenCoords() {
            double rms = rms(new double[]{this.centerX, this.centerY});
            return new double[]{
                    this.centerX * (rms - 1) / rms,
                    this.centerY * (rms - 1) / rms};
        }

        public double theta(double[] xy) {
            return -FastMath.atan2((xy[0] - centerX), (xy[1] - centerY));
        }

        public void theta(double theta, double[] returnValue) {
            returnValue[0] = centerX - FastMath.sin(theta) * radius;
            returnValue[1] = centerY + FastMath.cos(theta) * radius;
        }

        public boolean within(double[] xy) {
            double v = euclideanDistFromCenter(xy);
            return v <= radius;
        }

        public double euclideanDistFromCenter(double[] xy) {
            return rms(new double[]{xy[0] - centerX, xy[1] - centerY});
        }

        public double euclideanDistFromCircle(double[] xy) {
            return FastMath.abs(rms(new double[]{xy[0] - centerX, xy[1] - centerY}) - radius);
        }

        public double angle(Circle right) {
            double d = euclideanDistFromCenter(new double[]{right.centerX, right.centerY});
            if (d > radius + right.radius) {
                return Double.NaN;
            }

            return FastMath.PI - (FastMath.acos((radius * radius + right.radius * right.radius - d * d) / (2 * radius * right.radius)));
        }

        @Override
        public String toString() {
            return "Circle{" +
                    "centerX=" + centerX +
                    ", centerY=" + centerY +
                    ", radius=" + radius +
                    '}';
        }

    }

    private static double rms(double[] v) {
        return FastMath.sqrt(Arrays.stream(v).map(x -> x * x).reduce((a, b) -> a + b).getAsDouble());
    }

    public static class Polygon {
        public final double[][] vertices;
        public final Circle[] edges;

        public Polygon(double[][] vertices, Circle[] edges) {
            this.vertices = vertices;
            this.edges = edges;
            if (vertices.length != edges.length) {
                throw new IllegalArgumentException();
            }
            DoubleSummaryStatistics angleStats = IntStream.range(0, edges.length).mapToDouble(i -> edges[i].angle(edges[(i + 1) % edges.length])).summaryStatistics();
            if (!((angleStats.getMax() - angleStats.getMin()) < SPACIAL_CHECKS)) {
                throw new IllegalArgumentException();
            }
            for (int i = 0; i < edges.length; i++) {
                int j = (i + 1) % edges.length;
                if (edges[i].euclideanDistFromCircle(vertices[j]) > SPACIAL_CHECKS) {
                    throw new IllegalArgumentException();
                }
                if (edges[i].euclideanDistFromCircle(vertices[i]) > SPACIAL_CHECKS) {
                    throw new IllegalArgumentException();
                }
            }
        }

        public Polygon(double[][] vertices) {
            this.vertices = vertices;
            this.edges = IntStream.range(0, this.vertices.length)
                    .mapToObj(i -> intersecting(vertices[i], vertices[(i + 1) % vertices.length]))
                    .toArray(Circle[]::new);
        }

        private double[][] filter(double[][] pixelCoords) {
            for (Circle edge : this.edges) {
                DoubleSummaryStatistics distBounds = Arrays.stream(this.vertices)
                        .mapToDouble(x -> edge.euclideanDistFromCenter(x))
                        .summaryStatistics();
                pixelCoords = Arrays.stream(pixelCoords).filter(x -> {
                    double distFromCenter = edge.euclideanDistFromCenter(x);
                    if (distBounds.getMin() > distFromCenter) return false;
                    if (distBounds.getMax() < distFromCenter) return false;
                    return true;
                }).toArray(double[][]::new);
            }
            return pixelCoords;
        }

        public Polygon reflect(Circle edge) {
            return new Polygon(Arrays.stream(this.vertices)
                    .map(edge::reflect)
                    .toArray(double[][]::new));
        }
    }

    public static class PerpendicularResult {
        public final Circle edge;
        public final double[] intersection;

        public PerpendicularResult(Circle edge, double[] intersection) {
            this.edge = edge;
            this.intersection = intersection;
        }

        private boolean isReflection(double distanceFromMirror, double[] subject, double[] reflected) {
            if(null==reflected) return false;
            if(Arrays.stream(reflected).anyMatch(Double::isNaN)) return false;
            double tol = 1e-2 * distanceFromMirror;
            return FastMath.abs(poincareDist(subject, reflected) - 2 * distanceFromMirror) < tol && FastMath.abs(poincareDist(this.intersection, reflected) - distanceFromMirror) < tol;
        }
    }

    public static double poincareDist(double[] u, double[] v) {
        double ur = rms(u);
        double vr = rms(v);
        double uv = rms(new double[]{u[0] - v[0], u[1] - v[1]});
        double sigma1 = 1 + 2 * (uv * uv) / ((1 - ur * ur) * (1 - vr * vr));
        return FastMath.log(sigma1 + FastMath.sqrt(sigma1 * sigma1 - 1.0));
    }


    public static Circle intersecting(double[] xy1, double[] xy2) {
        Circle circle = intersecting(xy1, xy2, -2);
        if (circle.intersects(xy1) && circle.intersects(xy2)) {
            return circle;
        }
        circle = intersecting(xy1, xy2, 2);
        if (circle.intersects(xy1) && circle.intersects(xy2)) {
            return circle;
        }
        double scale = 1;
        while (scale < SPACIAL_INF) {
            int loopCount = 0;
            while (loopCount++ < 10) {
                circle = intersecting(xy1, xy2, (FastMath.random() - .5) * scale);
                if (circle.intersects(xy1) && circle.intersects(xy2)) {
                    return circle;
                }
            }
            scale *= 1e1;
        }
        return circle;
    }

    private static Circle intersecting(double[] a, double[] b, double guessX) {
        double midX = (a[0] + b[0]) / 2;
        double midY = (a[1] + b[1]) / 2;
        double slope = -(a[0] - b[0]) / (a[1] - b[1]);
        if (Math.abs(a[0] - b[0]) + Math.abs(a[1] - b[1]) < SPACIAL_CHECKS)
            return new Circle(new double[]{a[0], a[1]}, 0);
        if ((a[1] == b[1]) || Math.abs(slope) > 1e4) {
            Circle intersecting = intersecting(
                    new double[]{a[1], a[0]},
                    new double[]{b[1], b[0]},
                    guessX);
            return new Circle(new double[]{intersecting.centerY, intersecting.centerX}, intersecting.radius);
        }
        BrentOptimizer simplexOptimizer = new BrentOptimizer(TOL, TOL);
        UnivariateFunction fn = x -> {
            double y = (x - midX) * slope + midY;
            double rms = rms(new double[]{x, y});
            if (rms < 1) return 1000 / rms;
            Circle c = new Circle(new double[]{x, y});
            double d1 = c.euclideanDistFromCircle(a);
            double d2 = c.euclideanDistFromCircle(b);
            return FastMath.pow(d1, 2) + FastMath.pow(d2, 2);
        };
        double optimalX = simplexOptimizer.optimize(
                new UnivariateObjectiveFunction(fn),
                new MaxEval(1000),
                GoalType.MINIMIZE,
                new SearchInterval(-SPACIAL_INF, SPACIAL_INF, guessX)
        ).getPoint();
        double optimalY = (optimalX - midX) * slope + midY;
        return new Circle(new double[]{optimalX, optimalY});
    }


    public static Polygon regularPolygon(int p, int q) {
        double sweep_angle = 2 * FastMath.PI / p;
        double interior_angle = 2 * FastMath.PI / q;
        double projected_radius = FastMath.pow((1 - FastMath.pow(FastMath.sin(sweep_angle / 2) / FastMath.sin(((FastMath.PI / 2) + (interior_angle / 2))), 2)), -0.5);
        double poly_radius = projected_radius * FastMath.sin((FastMath.PI - (sweep_angle + interior_angle)) / 2) / FastMath.sin((FastMath.PI + interior_angle) / 2);
        return new Polygon(
                IntStream.range(0, p).mapToObj(n -> new double[]{
                        poly_radius * FastMath.cos(2 * FastMath.PI * n / p),
                        poly_radius * FastMath.sin(2 * FastMath.PI * n / p)
                }).toArray(double[][]::new),
                IntStream.range(0, p).mapToObj(n -> new Circle(new double[]{
                        projected_radius * FastMath.cos(2 * FastMath.PI * (n + 0.5) / p),
                        projected_radius * FastMath.sin(2 * FastMath.PI * (n + 0.5) / p)
                })).toArray(Circle[]::new)
        );
    }

    public static class TilingResult {
        private final int[] pixelMap;
        private final BufferedImage paint;

        public TilingResult(int[] pixelMap, BufferedImage paint) {
            this.pixelMap = pixelMap;
            this.paint = paint;
        }

        public int[] getPixelMap() {
            return pixelMap;
        }

        public BufferedImage getPaint() {
            return paint;
        }
    }
}
