package com.simiacryptus.math;

import org.apache.commons.math3.util.FastMath;
import org.jetbrains.annotations.NotNull;

import static java.lang.Math.PI;

public class PoincareCircle extends Circle {

    public final SpacialTransform reflection;

    public PoincareCircle(Point centerXY) {
        super(centerXY, calcRadius(centerXY));
        double theta = UNIT_CIRCLE.theta(center())+PI/2;
        double offset = center().rms() - radius;
        SpacialTransform rawReflection = SpacialTransform.rotate(-theta)
                .then(SpacialTransform.mobiusTranslate(-offset, 0))
                .then(SpacialTransform.scale(-1, 1))
                .then(SpacialTransform.mobiusTranslate(offset, 0))
                .then(SpacialTransform.rotate(theta));
        this.reflection = pt -> {
            Point reflect = rawReflection.apply(pt);
//            double dist = euclideanDistFromCircle(pt);
//            if(dist > 1e-8 || Geometry.rms(pt) > (1-1e-6)) {
//                PoincareCircle perpindicular = intersecting(reflect, pt);
//                double angle = angle(perpindicular);
//                List<Point> midpoint = perpindicular.intersect(this).stream().filter(UNIT_CIRCLE::within).collect(Collectors.toList());
//                double distA = poincareDist(reflect, midpoint.get(0));
//                double distB = poincareDist(pt, midpoint.get(0));
//                if (Math.abs((PI / 2) - angle) >= 1e-6) {
//                    throw new AssertionError();
//                }
//                if (Math.abs(distA - distB) >= 1e-6) {
//                    throw new AssertionError();
//                }
//            }
            return reflect;
        };
    }

    public static double calcRadius(Point centerXY) {
        double sqrt = FastMath.sqrt((centerXY.x * centerXY.x + centerXY.y * centerXY.y) - 1);
        if(Double.isFinite(sqrt)) {
            return sqrt;
        } else {
            return 0;
        }
    }

    public static double poincareDist(Point u, Point v) {
        double x = u.x - v.x;
        double y = u.y - v.y;
        double sigma1 = 1 + 2 * (x*x+y*y) / ((1 - u.sumSq()) * (1 - v.sumSq()));
        return FastMath.log(sigma1 + FastMath.sqrt(sigma1 * sigma1 - 1.0));
    }

    public static PoincareCircle intersecting(Point xy1, Point xy2) {
        if(Math.min(xy1.rms(), xy2.rms()) > (1-1e-6)) {
            throw new RuntimeException();
        }
        if(xy1.rms() > xy2.rms()) {
            return intersecting(xy1, xy2, UNIT_CIRCLE.inverse(xy2)).asPoincareCircle();
        } else {
            return intersecting(xy1, xy2, UNIT_CIRCLE.inverse(xy1)).asPoincareCircle();
        }
    }

    public Point center() {
        return new Point(center.x, center.y);
    }

    public Point reflect(Point subject) {
        if (subject.rms() >= 1) return subject;
        return reflection.apply(subject);
    }

    @NotNull
    public Point getPoincareGenCoords() {
        double rms = center().rms();
        return new Point(
                center.x * (rms - 1) / rms,
                center.y * (rms - 1) / rms);
    }

    public double poincareDistFromCircle(Point xy) {
        return poincareDist(reflect(xy), xy) / 2;
    }

}
