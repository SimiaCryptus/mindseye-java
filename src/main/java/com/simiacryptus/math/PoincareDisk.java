package com.simiacryptus.math;

import org.apache.commons.math3.util.FastMath;

import java.util.Arrays;
import java.util.Random;

public class PoincareDisk {

    public static final double TOL = 1e-8;
    public static final double SPACIAL_CHECKS = 1e-8;

    static final Random random = new Random();

    static double rms(double[] v) {
        return FastMath.sqrt(Arrays.stream(v).map(x -> x * x).reduce((a, b) -> a + b).getAsDouble());
    }

    public static double poincareDist(double[] u, double[] v) {
        double ur = rms(u);
        double vr = rms(v);
        double uv = rms(new double[]{u[0] - v[0], u[1] - v[1]});
        double sigma1 = 1 + 2 * (uv * uv) / ((1 - ur * ur) * (1 - vr * vr));
        return FastMath.log(sigma1 + FastMath.sqrt(sigma1 * sigma1 - 1.0));
    }

}