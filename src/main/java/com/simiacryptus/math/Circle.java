package com.simiacryptus.math;

import org.apache.commons.math3.util.FastMath;

import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class Circle {
    public static final Circle UNIT_CIRCLE = new Circle(new Point(0, 0), 1);
    public final Point center;
    public final double radius;

    public Circle(Point center, double radius) {
        this.center = center;
        this.radius = radius;
    }

    public Point inverse(Point pt) {
        double dx = pt.x - center.x;
        double dy = pt.y - center.y;
        double r = new Point(dx, dy).sumSq();
        return new Point(center.x + dx / (r), center.y + dy / (r));
    }

    public static Circle intersecting(Point pt1, Point pt2, Point pt3) {
        if (Math.min(Geometry.dist(pt1, pt2), Geometry.dist(pt3, pt2)) < 1e-8) return new Circle(pt1, 0);
        double x1 = pt1.x;
        double y1 = pt1.y;
        double x2 = pt2.x;
        double y2 = pt2.y;
        double x3 = pt3.x;
        double y3 = pt3.y;
        double a = x1 * (y2 - y3) - y1 * (x2 - x3) + x2 * y3 - x3 * y2;
        double b = (x1 * x1 + y1 * y1) * (y3 - y2) + (x2 * x2 + y2 * y2) * (y1 - y3) + (x3 * x3 + y3 * y3) * (y2 - y1);
        double c = (x1 * x1 + y1 * y1) * (x2 - x3) + (x2 * x2 + y2 * y2) * (x3 - x1) + (x3 * x3 + y3 * y3) * (x1 - x2);
        double d = (x1 * x1 + y1 * y1) * (x3 * y2 - x2 * y3) + (x2 * x2 + y2 * y2) * (x1 * y3 - x3 * y1) + (x3 * x3 + y3 * y3) * (x2 * y1 - x1 * y2);
        double r = FastMath.sqrt((b * b + c * c - 4 * a * d) / (4 * a * a));
        return new Circle(new Point(-b / (2 * a), -c / (2 * a)), r);
    }

    public List<Point> intersect(Circle right) {
        double x0 = center.x;
        double y0 = center.y;
        double r0 = radius;
        double x1 = right.center.x;
        double y1 = right.center.y;
        double r1 = right.radius;
        double d = FastMath.sqrt((x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0));
        double a = (r0 * r0 - r1 * r1 + d * d) / (2 * d);
        double h = FastMath.sqrt(r0 * r0 - a * a);
        double x2 = x0 + a * (x1 - x0) / d;
        double y2 = y0 + a * (y1 - y0) / d;
        return Stream.of(
                new Point(
                        x2 + h * (y1 - y0) / d,
                        y2 - h * (x1 - x0) / d
                ), new Point(
                        x2 - h * (y1 - y0) / d,
                        y2 + h * (x1 - x0) / d
                )).collect(Collectors.toList());
    }

    public Point theta(double theta) {
        return new Point(
                center.x - FastMath.sin(theta) * radius,
                center.y + FastMath.cos(theta) * radius
        );
    }

    public double theta(Point xy) {
        return -FastMath.atan2((xy.x - center.x), (xy.y - center.y));
    }

    public boolean within(Point xy) {
        double v = euclideanDistFromCenter(xy);
        return v <= radius;
    }

    public double euclideanDistFromCenter(Point xy) {
        return Geometry.rms(xy.x - center.x, xy.y - center.y);
    }

    public double euclideanDistFromCircle(Point xy) {
        return FastMath.abs(Geometry.rms(xy.x - center.x, xy.y - center.y) - radius);
    }

    public double angle(Circle right) {
        double d = euclideanDistFromCenter(new Point(right.center.x, right.center.y));
        if (d > radius + right.radius) {
            return Double.NaN;
        }
        return FastMath.PI - (FastMath.acos((radius * radius + right.radius * right.radius - d * d) / (2 * radius * right.radius)));
    }

    @Override
    public String toString() {
        return "Circle{" +
                "centerX=" + center.x +
                ", centerY=" + center.y +
                ", radius=" + radius +
                '}';
    }

    public PoincareCircle asPoincareCircle() {
        Point center = this.center;
        if (center.rms() < 1.0) {
            center = center().scale(1.0 / center().rms());
        }
        while (center.rms() < 1.0) {
            center = center().scale(1.0 + 1e-6);
        }
        return new PoincareCircle(new Point(center.x, center.y));
    }

    public Point center() {
        return new Point(center.x, center.y);
    }

}
