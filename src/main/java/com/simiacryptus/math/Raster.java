package com.simiacryptus.math;

import org.apache.commons.math3.util.FastMath;
import org.jetbrains.annotations.NotNull;

import java.awt.image.BufferedImage;
import java.awt.image.WritableRaster;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public class Raster {
    public final int sizeX;
    public final int sizeY;

    public Raster(int sizeX, int sizeY) {
        this.sizeX = sizeX;
        this.sizeY = sizeY;
    }

    private void paintall(Polygon polygon, BufferedImage paint, int n, int[] pixelMap) {
        Arrays.stream(polygon.edges).parallel().forEach(edge -> {
            Polygon reflected = polygon.reflect(edge);
            int[] color = IntStream.generate(() -> PoincareDisk.random.nextInt(200) + 50).limit(3).toArray();
            int[] pixels = pixels(reflected);
            double minFill = 0.25;
            if (paint(pixels, paint, color, minFill) && n > 0) {
                double[] reflect = new double[]{Double.NaN, Double.NaN};
                int skipped = 0;
                for (int pixel : pixels) {
                    if (pixelMap[pixel] != -1) {
                        if (skipped++ > ((1 - minFill) * pixels.length)) {
                            break;
                        } else {
                            continue;
                        }
                    }
                    double[] coords = convertCoords(fromIndex(pixel));
                    if (!(PoincareDisk.rms(coords) < 1)) continue;
                    edge.reflect(coords, reflect);
                    int toIndex = toIndex(convertCoords(reflect));
                    if (toIndex != pixel) {
                        if (pixelMap[toIndex] != -1) {
                            pixelMap[pixel] = toIndex;
                            if (pixelMap[toIndex] != toIndex) {
                                toIndex = pixelMap[toIndex];
                            }
                        }
                        pixelMap[pixel] = toIndex;
                    }
                }
                this.paintall(reflected, paint, n - 1, pixelMap);
            }
        });
    }

    public TilingResult pixelMap(Polygon polygon, int iterations) {
        int[] pixelMap = new int[this.sizeX * this.sizeY];
        Arrays.fill(pixelMap, -1);
        int[] pixels = pixels(polygon);
        for (int pixel : pixels) pixelMap[pixel] = pixel;
        BufferedImage paint = paint(pixels);
        this.paintall(polygon, paint, iterations, pixelMap);
        TilingResult tilingResult = new TilingResult(pixelMap, paint);
        return tilingResult;
    }

    @NotNull
    public BufferedImage view(int[] pixelMap, BufferedImage image) {
        BufferedImage imgview = new BufferedImage(this.sizeX, this.sizeY, BufferedImage.TYPE_INT_RGB);
        IntStream.range(0, pixelMap.length).forEach(i -> {
            int j = pixelMap[i];
            if (j >= 0) {
                @NotNull int[] xy1 = fromIndex(i);
                @NotNull int[] xy2 = fromIndex(j);
                int[] buffer = new int[3];
                image.getRaster().getPixel(xy2[0], xy2[1], buffer);
                imgview.getRaster().setPixel(xy1[0], xy1[1], buffer);
            }
        });
        return imgview;
    }

    @NotNull
    public BufferedImage paint(int[] pixels) {
        BufferedImage image = new BufferedImage(this.sizeX, this.sizeY, BufferedImage.TYPE_INT_RGB);
        paint(pixels, image, new int[]{255, 255, 255}, 0.0);
        return image;
    }

    public boolean paint(int[] pixels, BufferedImage image, int[] color, double minFill) {
        WritableRaster raster = image.getRaster();
        int[] buffer = new int[3];
        List<@NotNull int[]> pixelsToWrite = Arrays.stream(pixels).mapToObj(this::fromIndex).filter(xy -> {
            raster.getPixel(xy[0], xy[1], buffer);
            return Arrays.stream(buffer).sum() == 0;
        }).collect(Collectors.toList());
        if (pixelsToWrite.size() > (pixels.length * minFill)) {
            pixelsToWrite.forEach(xy -> {
                raster.setPixel(xy[0], xy[1], color);
            });
            System.out.printf("Painting poly with %d / %d pixels (%.3f)%n", pixelsToWrite.size(), pixels.length, (double) pixelsToWrite.size() /  pixels.length);
            return true;
        } else {
            System.out.printf("Ignoring poly with %d / %d pixels (%.3f)%n", pixelsToWrite.size(), pixels.length, (double) pixelsToWrite.size() /  pixels.length);
            return false;
        }
    }

    @NotNull
    public Stream<int[]> toPixels(Stream<double[]> stream) {
        return stream
                .map(this::convertCoords)
                .mapToInt(this::toIndex)
                .distinct().sorted()
                .mapToObj(this::fromIndex);
    }

    @NotNull
    public double[] convertCoords(int[] xy) {
        return new double[]{
                (2.0 * xy[0] / sizeX) - 1,
                (2.0 * xy[1] / sizeY) - 1
        };
    }

    @NotNull
    public int[] convertCoords(double[] xy) {
        return new int[]{
                (int) ((sizeX / 2) * (xy[0] + 1)),
                (int) ((sizeY / 2) * (xy[1] + 1))
        };
    }

    @NotNull
    public int[] fromIndex(int i) {
        int[] xy = new int[2];
        fromIndex(i, xy);
        return xy;
    }

    @NotNull
    public Stream<double[]> allPixels() {
        return IntStream.range(0, sizeX * 2).mapToObj(x -> x)
                .flatMap(x -> IntStream.range(0, sizeY * 2)
                        .mapToObj(y -> {
                            int[] xy = new int[]{x, y};
                            return new double[]{
                                    (1.0 * xy[0] / sizeX) - 1,
                                    (1.0 * xy[1] / sizeY) - 1
                            };
                        })
                        .filter(v -> PoincareDisk.rms(v) <= 1));
    }

    public int toIndex(int[] xy) {
        return xy[0] + sizeX * xy[1];
    }

    public void fromIndex(int i, int[] xy) {
        xy[0] = FastMath.max(0, FastMath.min(sizeX - 1, i % sizeX));
        xy[1] = FastMath.max(0, FastMath.min(sizeY - 1, i / sizeX));
    }

    public int[] pixels(Polygon polygon) {
        double[][] allpixels = allPixels().toArray(double[][]::new);
        double[][] filtered = polygon.filter(allpixels);
        return toPixels(Arrays.stream(filtered))
                .mapToInt(this::toIndex)
                .distinct().sorted().toArray();
    }
}
