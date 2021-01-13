package com.simiacryptus.math;

public class Point {
    public final double x;
    public final double y;

    public Point(double x, double y) {
        if (!Double.isFinite(x)) {
            throw new IllegalArgumentException();
        }
        if (!Double.isFinite(y)) {
            throw new IllegalArgumentException();
        }
        this.x = x;
        this.y = y;
    }

    public double rms() {
        return Geometry.rms(this.x, this.y);
    }

    public double theta() {
        return Math.sin(scale(1.0/rms()).x);
    }

    public double sumSq() {
        return Geometry.sumSq(this.x, this.y);
    }

    public Point add(Point r) {
        return new Point(x + r.x, y + r.y);
    }

    public double dist(Point r) {
        return r.scale(-1).add(this).rms();
    }

    public Point scale(double f) {
        return new Point(x*f,y*f);
    }
    public Point rotate(double f) {
        return new Point(
                x*Math.cos(f) - y * Math.sin(f),
                x*Math.sin(f) + y * Math.cos(f));
    }
}
