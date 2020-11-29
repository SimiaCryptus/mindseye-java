package com.simiacryptus.math;

import org.apache.commons.math3.util.FastMath;

import java.util.Arrays;
import java.util.Random;

public class Geometry {

    static final Random random = new Random();

    static double rms(double[] v) {
        return FastMath.sqrt(sumSq(v));
    }

    static double rms(double x, double y) {
        return FastMath.sqrt(sumSq(x,y));
    }

    public static double sumSq(double... values) {
        double s = 0;
        for (double x : values) s += x * x;
        return s;
    }

    public static double sumSq(double x, double y) {
        double s = 0;
        s += x * x;
        s += y * y;
        return s;
    }

    public static double dist(Point a, Point b) {
        return rms(a.x - b.x, a.y - b.y);
    }
}
