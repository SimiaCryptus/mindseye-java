package com.simiacryptus.math;

import org.apache.commons.math3.util.FastMath;

import java.util.Arrays;

public class PerpendicularResult {
    public final PoincareCircle edge;
    public final double[] intersection;

    public PerpendicularResult(PoincareCircle edge, double[] intersection) {
        this.edge = edge;
        this.intersection = intersection;
    }

    public boolean isReflection(double distanceFromMirror, double[] subject, double[] reflected) {
        if (null == reflected) return false;
        if (Arrays.stream(reflected).anyMatch(Double::isNaN)) return false;
        double tol = 1e-2 * distanceFromMirror;
        return FastMath.abs(PoincareDisk.poincareDist(subject, reflected) - 2 * distanceFromMirror) < tol && FastMath.abs(PoincareDisk.poincareDist(this.intersection, reflected) - distanceFromMirror) < tol;
    }
}
