package com.simiacryptus.mindseye.layers.java;

import com.simiacryptus.math.PoincareDisk;

import javax.annotation.Nonnull;

public class ImgIndexMapViewLayer extends ImgViewLayerBase {
    private final int[] pixelMap;
    private final int sizeX;
    private final int sizeY;

    public ImgIndexMapViewLayer(PoincareDisk.Raster raster, int[] pixelMap) {
        this.sizeX = raster.sizeX;
        this.sizeY = raster.sizeY;
        this.pixelMap = pixelMap;
    }

    @Override
    protected @Nonnull double[] coordinateMapping(@Nonnull double... xy) {
        final PoincareDisk.Raster raster = new PoincareDisk.Raster(sizeX, sizeY);
        int[] ints = {(int) xy[0], (int) xy[1]};
        int i = raster.toIndex(ints);
        int j = pixelMap[i];
        if (j < 0) return new double[]{0.0, 0.0};
        raster.fromIndex(j, ints);
        return new double[]{ints[0], ints[1]};
    }
}
