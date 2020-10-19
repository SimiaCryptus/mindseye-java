package com.simiacryptus.math;

import java.awt.image.BufferedImage;

public class TilingResult {
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
