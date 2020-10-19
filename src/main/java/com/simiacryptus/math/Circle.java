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

import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class Circle {
    public static final Circle UNIT_CIRCLE = new Circle(new double[]{0, 0}, 1);
    public final double centerX;
    public final double centerY;
    public final double radius;

    public Circle(double[] centerXY) {
        this(centerXY, FastMath.sqrt((centerXY[0] * centerXY[0] + centerXY[1] * centerXY[1]) - 1));
    }

    public Circle(double[] centerXY, double radius) {
        this.centerX = centerXY[0];
        this.centerY = centerXY[1];
        this.radius = radius;
    }

    public double[] inverse(double[] pt) {
        double dx = pt[0] - centerX;
        double dy = pt[1] - centerY;
        double r = PoincareDisk.rms(new double[]{dx, dy});
        return new double[]{centerX + dx / (r*r), centerY + dy / (r*r)};
    }

    public static PoincareCircle intersecting(double[] xy1, double[] xy2) {
        if(PoincareDisk.rms(xy1) > PoincareDisk.rms(xy2)) {
            return PoincareCircle.intersecting(xy1, xy2, UNIT_CIRCLE.inverse(xy2));
        } else {
            return PoincareCircle.intersecting(xy1, xy2, UNIT_CIRCLE.inverse(xy1));
        }
    }

    @NotNull
    double[] cycle_theta_bounds(double[] subject, double[] theta_bounds) {
        double theta = theta(subject);
        if (theta < theta_bounds[0]) {
            theta_bounds = new double[]{
                    theta_bounds[0] - FastMath.PI * 2,
                    theta_bounds[1] - FastMath.PI * 2
            };
        } else if (theta > theta_bounds[1]) {
            theta_bounds = new double[]{
                    theta_bounds[0] + FastMath.PI * 2,
                    theta_bounds[1] + FastMath.PI * 2
            };
        }
        assert !(theta_bounds[0] >= theta_bounds[1]);
        return theta_bounds;
    }

    @NotNull
    public double[] theta_bounds(Circle circle) {
        List<double[]> infinities = circle.intersect(this);
        assert 2 == infinities.size();
        double[] theta_bounds = infinities.stream().mapToDouble(this::theta).sorted().toArray();
        synchronized (theta_bounds) {
            while (theta_bounds[0] > theta_bounds[1]) {
                theta_bounds = infinities.stream().mapToDouble(this::theta).sorted().toArray();
            }
        }
        if (theta_bounds[1] - theta_bounds[0] > FastMath.PI) {
            theta_bounds = new double[]{
                    theta_bounds[1] - FastMath.PI * 2,
                    theta_bounds[0]
            };
            if (theta_bounds[0] < 2 * FastMath.PI) {
                theta_bounds = new double[]{
                        theta_bounds[0] + FastMath.PI * 2,
                        theta_bounds[1] + FastMath.PI * 2
                };
            }
        }
        return theta_bounds;
    }

    public boolean intersects(double[] xy) {
        return euclideanDistFromCircle(xy) < PoincareDisk.SPACIAL_CHECKS;
    }

    public List<double[]> intersect(Circle right) {
        double x0 = centerX;
        double y0 = centerY;
        double r0 = radius;
        double x1 = right.centerX;
        double y1 = right.centerY;
        double r1 = right.radius;
        double d = FastMath.sqrt((x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0));
        double a = (r0 * r0 - r1 * r1 + d * d) / (2 * d);
        double h = FastMath.sqrt(r0 * r0 - a * a);
        double x2 = x0 + a * (x1 - x0) / d;
        double y2 = y0 + a * (y1 - y0) / d;
        return Stream.of(
                new double[]{
                        x2 + h * (y1 - y0) / d,
                        y2 - h * (x1 - x0) / d
                }, new double[]{
                        x2 - h * (y1 - y0) / d,
                        y2 + h * (x1 - x0) / d
                }).collect(Collectors.toList());
    }

    boolean isPerpendicular(Circle circle) {
        return circle != null && FastMath.abs((FastMath.PI / 2) - angle(circle)) < PoincareDisk.SPACIAL_CHECKS;
    }

    public double[] theta(double optimal) {
        double[] xy = new double[2];
        theta(optimal, xy);
        return xy;
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
        return PoincareDisk.rms(new double[]{xy[0] - centerX, xy[1] - centerY});
    }

    public double euclideanDistFromCircle(double[] xy) {
        return FastMath.abs(PoincareDisk.rms(new double[]{xy[0] - centerX, xy[1] - centerY}) - radius);
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

    public PoincareCircle poincare() {
        return new PoincareCircle(new double[]{centerX, centerY});
    }
}
