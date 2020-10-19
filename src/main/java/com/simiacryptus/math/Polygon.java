package com.simiacryptus.math;

import org.apache.commons.math3.util.FastMath;

import java.util.Arrays;
import java.util.DoubleSummaryStatistics;
import java.util.stream.IntStream;

public class Polygon {
    public final double[][] vertices;
    public final PoincareCircle[] edges;

    public Polygon(double[][] vertices, PoincareCircle[] edges) {
        this.vertices = vertices;
        this.edges = edges;
        assert vertices.length == edges.length;
        DoubleSummaryStatistics angleStats = IntStream.range(0, edges.length).mapToDouble(i -> edges[i].angle(edges[(i + 1) % edges.length])).summaryStatistics();
        assert (angleStats.getMax() - angleStats.getMin()) < PoincareDisk.SPACIAL_CHECKS;
        for (int i = 0; i < edges.length; i++) {
            int j = (i + 1) % edges.length;
            assert !(edges[i].euclideanDistFromCircle(vertices[j]) > PoincareDisk.SPACIAL_CHECKS);
            assert !(edges[i].euclideanDistFromCircle(vertices[i]) > PoincareDisk.SPACIAL_CHECKS);
        }
    }

    public Polygon(double[][] vertices) {
        this.vertices = vertices;
        this.edges = IntStream.range(0, this.vertices.length)
                .mapToObj(i -> Circle.intersecting(vertices[i], vertices[(i + 1) % vertices.length]).poincare())
                .toArray(PoincareCircle[]::new);
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
                IntStream.range(0, p).mapToObj(n -> new PoincareCircle(new double[]{
                        projected_radius * FastMath.cos(2 * FastMath.PI * (n + 0.5) / p),
                        projected_radius * FastMath.sin(2 * FastMath.PI * (n + 0.5) / p)
                })).toArray(PoincareCircle[]::new)
        );
    }

    double[][] filter(double[][] pixelCoords) {
        for (PoincareCircle edge : this.edges) {
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

    public Polygon reflect(PoincareCircle edge) {
        return new Polygon(Arrays.stream(this.vertices)
                .map(edge::reflect)
                .toArray(double[][]::new));
    }
}
