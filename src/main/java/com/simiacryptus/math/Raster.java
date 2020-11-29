package com.simiacryptus.math;

import com.simiacryptus.mindseye.layers.java.ImgIndexMapViewLayer;
import com.simiacryptus.mindseye.util.ImageUtil;
import org.jetbrains.annotations.NotNull;

import javax.annotation.Nonnull;
import java.awt.image.BufferedImage;
import java.awt.image.WritableRaster;
import java.util.Arrays;
import java.util.List;
import java.util.function.UnaryOperator;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public class Raster {
    public final int sizeX;
    public final int sizeY;
    private boolean filterCircle = true;

    public Raster(int sizeX, int sizeY) {
        this.sizeX = sizeX;
        this.sizeY = sizeY;
    }

    public @Nonnull BufferedImage resize(BufferedImage image) {
        return ImageUtil.resize(image, this.sizeX, this.sizeY);
    }

    @NotNull
    public ImgIndexMapViewLayer toLayer(UnaryOperator<Point> transform) {
        return new ImgIndexMapViewLayer(this, buildPixelMap(transform));
    }

    @NotNull
    public int[] buildPixelMap(UnaryOperator<Point> transform) {
        int[] pixelMap = new int[this.sizeX * this.sizeY];
        Arrays.fill(pixelMap, -2);
        allPixels().parallel().forEach(sourcePoint -> {
            int sourceIndex = toIndex(convertCoords(sourcePoint));
            pixelMap[sourceIndex] = -1;
        });
        assert Arrays.stream(pixelMap).allMatch(x->x!=-2);
        for (int pixel = 0; pixel < pixelMap.length; pixel++) pixelMap[pixel] = pixel;
        allPixels().parallel().forEach(sourcePoint -> {
            fillPixel(pixelMap, sourcePoint, transform);
        });
        return pixelMap;
    }

    private void fillPixel(int[] pixelMap, Point sourcePoint, UnaryOperator<Point> transform) {
        int sourceIndex = toIndex(convertCoords(sourcePoint));
        if(sourcePoint.rms() < 1 || !isFilterCircle()) {
            Point destPt = transform.apply(sourcePoint);
            if(destPt != null) {
                int targetIndex = toIndex(convertCoords(destPt));
                pixelMap[sourceIndex] = targetIndex;
            } else {
                pixelMap[sourceIndex] = -1;
            }
        } else {
            pixelMap[sourceIndex] = -1;
        }
    }

    @NotNull
    public BufferedImage view(int[] pixelMap, BufferedImage image) {
        BufferedImage imgview = getImage();
        IntStream.range(0, pixelMap.length).forEach(i -> {
            int j = pixelMap[i];
            if (j >= 0) {
                int[] xy1 = fromIndex(i);
                int[] xy2 = fromIndex(j);
                int[] buffer = new int[3];
                image.getRaster().getPixel(xy2[0], xy2[1], buffer);
                imgview.getRaster().setPixel(xy1[0], xy1[1], buffer);
            }
        });
        return imgview;
    }

    @NotNull
    public BufferedImage paint(int[] pixels, int[] color) {
        BufferedImage image = getImage();
        paint(pixels, color, image.getRaster());
        return image;
    }

    @NotNull
    public BufferedImage getImage() {
        return new BufferedImage(this.sizeX, this.sizeY, BufferedImage.TYPE_INT_RGB);
    }

    void paint(int[] pixels, int[] color, WritableRaster raster) {
        int[] buffer = new int[3];
        List<int[]> pixelsToWrite = Arrays.stream(pixels).mapToObj(this::fromIndex).filter(xy -> {
            raster.getPixel(xy[0], xy[1], buffer);
            return Arrays.stream(buffer).sum() == 0;
        }).collect(Collectors.toList());
        pixelsToWrite.forEach(xy -> {
            raster.setPixel(xy[0], xy[1], color);
        });
    }

    @NotNull
    public Point convertCoords(int[] xy) {
        Point point = new Point(
                (2.0 * xy[0] / sizeX) - 1,
                (2.0 * xy[1] / sizeY) - 1
        );
        assert Arrays.equals(xy, convertCoords(point));
        return point;
    }

    @NotNull
    public int[] convertCoords(Point xy) {
//        assert xy.x >= -1;
//        assert xy.x <= 1;
//        assert xy.y >= -1;
//        assert xy.y <= 1;
        int[] ints = {
                Math.max(0, Math.min(sizeX-1, (int) Math.round((sizeX / 2.0) * (xy.x + 1)))),
                Math.max(0, Math.min(sizeY-1, (int) Math.round((sizeY / 2.0) * (xy.y + 1))))
        };
        return ints;
    }

    @NotNull
    public Stream<Point> allPixels() {
        return IntStream.range(0, sizeX*sizeY).mapToObj(this::fromIndex).map(this::convertCoords);
    }

    public int toIndex(int... xy) {
        if (xy[0] < 0) {
            throw new AssertionError();
        }
        assert xy[1] >= 0;
        assert xy[0] < sizeX;
        assert xy[1] < sizeY;
        int index = xy[0] * sizeY + xy[1];
        assert index >= 0;
        assert index < (sizeX * sizeY);
        return index;
    }

    public int[] fromIndex(int i) {
        if(i < 0) throw new IllegalArgumentException();
        if(i >= sizeX*sizeY) throw new IllegalArgumentException();
        int[] ints = { i / sizeY, i % sizeY };
        assert toIndex(ints) == i : String.format("%d -> %s -> %d", i, Arrays.toString(ints), toIndex(ints));
        return ints;
    }

    public int[] pixels(HyperbolicPolygon polygon) {
        Point[] allpixels = allPixels().toArray(Point[]::new);
        Point[] filtered = polygon.filter(allpixels);
        return Arrays.stream(filtered)
                .map(this::convertCoords)
                .mapToInt(this::toIndex)
                .distinct().sorted().toArray();
    }

    public boolean isFilterCircle() {
        return filterCircle;
    }

    public Raster setFilterCircle(boolean filterCircle) {
        this.filterCircle = filterCircle;
        return this;
    }
}
