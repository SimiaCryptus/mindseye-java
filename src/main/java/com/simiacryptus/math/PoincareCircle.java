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

public class PoincareCircle extends Circle {
    final double[] bounds;

    public PoincareCircle(double[] centerXY) {
        super(centerXY);
        this.bounds = cycle_theta_bounds(new double[]{0, 0}, theta_bounds(Circle.UNIT_CIRCLE));
    }

    static PoincareCircle intersecting(double[] pt1, double[] pt2, double[] pt3) {
        double x1 = pt1[0];
        double y1 = pt1[1];
        double x2 = pt2[0];
        double y2 = pt2[1];
        double x3 = pt3[0];
        double y3 = pt3[1];
        double a = x1*(y2-y3)-y1*(x2-x3)+x2*y3-x3*y2;
        double b = (x1*x1+y1*y1)*(y3-y2)+(x2*x2+y2*y2)*(y1-y3)+(x3*x3+y3*y3)*(y2-y1);
        double c = (x1*x1+y1*y1)*(x2-x3)+(x2*x2+y2*y2)*(x3-x1)+(x3*x3+y3*y3)*(x1-x2);
        double d = (x1*x1+y1*y1)*(x3*y2-x2*y3)+(x2*x2+y2*y2)*(x1*y3-x3*y1)+(x3*x3+y3*y3)*(x2*y1-x1*y2);
        double r = FastMath.sqrt((b * b + c * c - 4 * a * d) / (4 * a * a));
        PoincareCircle circle = new PoincareCircle(new double[]{-b / (2 * a), -c / (2 * a)});
//        assert circle.euclideanDistFromCircle(pt1) < PoincareDisk.SPACIAL_CHECKS;
//        assert circle.euclideanDistFromCircle(pt2) < PoincareDisk.SPACIAL_CHECKS;
//        assert circle.euclideanDistFromCircle(pt3) < PoincareDisk.SPACIAL_CHECKS;
        return circle;
    }


    public PerpendicularResult perpendicular(double[] point) {
        assert !(euclideanDistFromCircle(point) < PoincareDisk.SPACIAL_CHECKS);
        return finalPerpendicular(point, cycle_theta_bounds(point, theta_bounds(UNIT_CIRCLE)), theta(point));
    }

    public PerpendicularResult perpendicular(double[] point, PoincareCircle guess) {
        assert !(euclideanDistFromCircle(point) < PoincareDisk.SPACIAL_CHECKS);
        @NotNull double[] theta_bounds = cycle_theta_bounds(point, theta_bounds(UNIT_CIRCLE));
        List<double[]> intersections = intersect(guess).stream().filter(pt -> {
            double theta = theta(pt);
            if (theta < theta_bounds[0]) return false;
            if (theta > theta_bounds[1]) return false;
            return true;
        }).collect(Collectors.toList());
        double[][] interfacePoints = intersections.stream().sorted(Comparator.comparingDouble(pt -> PoincareDisk.rms(new double[]{pt[0] - point[0], pt[1] - point[1]}))).toArray(double[][]::new);
        return finalPerpendicular(point, theta_bounds, Arrays.stream(interfacePoints).mapToDouble(this::theta).toArray());
    }

    private PerpendicularResult finalPerpendicular(double[] point, @NotNull double[] theta_bounds, double... theta_guesses) {
        double[] interfacePoint = null;
        PoincareCircle circle = null;
        for (double theta_guess : theta_guesses) {
            interfacePoint = theta(optimizeSurfaceTheta(point, theta_guess, theta_bounds));
            circle = intersecting(point, interfacePoint);
            if (isPerpendicular(circle)) {
                return new PerpendicularResult(circle, interfacePoint);
            }
        }
        int attempts = 0;
        while (!isPerpendicular(circle)) {
            if (attempts++ > 10) {
                break;
            }
            interfacePoint = theta(optimizeSurfaceTheta(point, (FastMath.random() * (theta_bounds[1] - theta_bounds[0])) + theta_bounds[0], theta_bounds));
            circle = intersecting(point, interfacePoint);
        }
        return new PerpendicularResult(circle, interfacePoint);
    }

    private double optimizeSurfaceTheta(double[] subject, double guessTheta, @NotNull double[] theta_bounds) {
        if (guessTheta < theta_bounds[0]) return theta_bounds[0];
        if (guessTheta > theta_bounds[1]) return theta_bounds[1];
        BrentOptimizer simplexOptimizer = new BrentOptimizer(PoincareDisk.TOL, PoincareDisk.TOL);
        try {
            UnivariatePointValuePair optimize = simplexOptimizer.optimize(
                    new UnivariateObjectiveFunction(theta -> {
                        double[] th = theta(theta);
                        double rms = PoincareDisk.rms(th);
                        if (rms >= 1) return (1000 * rms);
                        Circle intersecting = intersecting(subject, th);
                        if (null == intersecting) return 1e4;
                        return FastMath.abs((FastMath.PI / 2) - angle(intersecting));
                    }),
                    new MaxEval(100),
                    GoalType.MINIMIZE,
                    new SearchInterval(theta_bounds[0], theta_bounds[1], guessTheta)
            );
            return optimize.getPoint();
        } catch (TooManyEvaluationsException e) {
            return guessTheta;
        }
    }

    public double[] reflect(double[] subject) {
        if (PoincareDisk.rms(subject) >= 1) return subject;
        double targetDist = euclideanDistFromCircle(subject);
        if (targetDist < PoincareDisk.SPACIAL_CHECKS) return subject;
        PerpendicularResult perp = perpendicular(subject);
        double distanceFromMirror = PoincareDisk.poincareDist(subject, perp.intersection);
        if (!Double.isFinite(distanceFromMirror)) return subject;
        double[] reflected = new double[]{Double.NaN, Double.NaN};
        reflected = reflect_fallback(subject, perp, distanceFromMirror, reflected);
        return reflected;
    }

    public void reflect(double[] subject, double[] reflect) {
        if (!Arrays.stream(reflect).allMatch(Double::isFinite)) {
            double[] r = reflect(subject);
            reflect[0] = r[0];
            reflect[1] = r[1];
            return;
        }
        assert !(PoincareDisk.rms(subject) >= 1);
        double targetDist = euclideanDistFromCircle(subject);
        if (targetDist < PoincareDisk.SPACIAL_CHECKS) {
            reflect[0] = subject[0];
            reflect[1] = subject[1];
            return;
        }
        PerpendicularResult perp = perpendicular(subject, intersecting(subject, reflect));
        double distanceFromMirror = PoincareDisk.poincareDist(subject, perp.intersection);
        UnivariateFunction fn = reflectionFitness(perp, subject, distanceFromMirror);
        double[] reflected = reflect(perp, fn,
                new double[]{Double.NaN, Double.NaN},
                perp.edge.cycle_theta_bounds(subject, perp.edge.theta_bounds(UNIT_CIRCLE)),
                perp.edge.theta(reflect) % FastMath.PI);
        reflected = reflect_fallback(subject, perp, distanceFromMirror, reflected);
        reflect[0] = reflected[0];
        reflect[1] = reflected[1];
    }

    @NotNull
    private UnivariateFunction reflectionFitness(PerpendicularResult perp, double[] subject, double distanceFromMirror) {
        return theta -> {
            double[] xy = perp.edge.theta(theta);
            double rms = PoincareDisk.rms(xy);
            if (rms > 1) return rms * 1000;
            double a = FastMath.abs(distanceFromMirror - PoincareDisk.poincareDist(xy, perp.intersection));
            double b = FastMath.abs(2 * distanceFromMirror - PoincareDisk.poincareDist(xy, subject));
            return a * a + b * b;
        };
    }

    private double[] reflect_fallback(double[] subject, PerpendicularResult perp, double distanceFromMirror, double[] reflected) {
        if (!perp.isReflection(distanceFromMirror, subject, reflected)) {
            UnivariateFunction fn = reflectionFitness(perp, subject, distanceFromMirror);
            double theta_intersect = perp.edge.theta(perp.intersection);
            double[] theta_bounds = perp.edge.cycle_theta_bounds(subject, perp.edge.theta_bounds(UNIT_CIRCLE));
            double theta_subject = perp.edge.theta(subject);
            while (theta_intersect < theta_bounds[0]) {
                theta_intersect += 2 * FastMath.PI;
            }
            while (theta_intersect > theta_bounds[1]) {
                theta_intersect -= 2 * FastMath.PI;
            }
            double guess_theta;
            double l = theta_intersect - theta_bounds[0];
            double r = theta_bounds[1] - theta_intersect;
            if (l < 0) {
                l = 0;
            }
            if (r < 0) {
                r = 0;
            }
            if ((l + r) < PoincareDisk.SPACIAL_CHECKS) {
                return Arrays.copyOf(subject, subject.length);
            }
            if (theta_intersect > theta_subject) {
                guess_theta = theta_intersect + (theta_intersect - theta_subject) * (r / l);
            } else {
                guess_theta = theta_intersect - (theta_subject - theta_intersect) * (l / r);
            }
            if (Double.isInfinite(distanceFromMirror) || distanceFromMirror > 30) return perp.edge.theta(guess_theta);
            reflected = reflect(perp, fn, reflected, theta_bounds, guess_theta);
        }
        return reflected;
    }

    private double[] reflect(PerpendicularResult perp, UnivariateFunction fn, double[] reflected, double[] theta_bounds, double guess_theta) {
        if (guess_theta < theta_bounds[0]) {
            return reflected;
        }
        if (guess_theta > theta_bounds[1]) {
            return reflected;
        }
        try {
            double[] result = perp.edge.theta(new BrentOptimizer(PoincareDisk.TOL, PoincareDisk.TOL).optimize(
                    new UnivariateObjectiveFunction(fn),
                    new MaxEval(100),
                    GoalType.MINIMIZE,
                    new SearchInterval(theta_bounds[0], theta_bounds[1], guess_theta)
            ).getPoint());
            assert !(PoincareDisk.rms(result) >= 1);
            return result;
        } catch (TooManyEvaluationsException e) {
            return reflected;
        }
    }

    @NotNull
    public double[] getPoincareGenCoords() {
        double rms = PoincareDisk.rms(new double[]{this.centerX, this.centerY});
        return new double[]{
                this.centerX * (rms - 1) / rms,
                this.centerY * (rms - 1) / rms};
    }

}
