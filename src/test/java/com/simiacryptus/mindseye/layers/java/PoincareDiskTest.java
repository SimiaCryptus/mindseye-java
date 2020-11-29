/*
 * Copyright (c) 2019 by Andrew Charneski.
 *
 * The author licenses this file to you under the
 * Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance
 * with the License.  You may obtain a copy
 * of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package com.simiacryptus.mindseye.layers.java;

import com.simiacryptus.math.Point;
import com.simiacryptus.math.*;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.util.ImageUtil;
import org.jetbrains.annotations.NotNull;
import org.junit.jupiter.api.Test;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.function.UnaryOperator;

import static com.simiacryptus.math.Circle.UNIT_CIRCLE;

public class PoincareDiskTest {

    @Test
    public void test() throws IOException {
        //new PoincareGeometry.Circle()
        Circle circle;
        for (double theta = -3; theta < 3; theta += 0.5) {
            Point thetaTest = UNIT_CIRCLE.theta(theta);
            System.out.printf("%s -> %s -> %s -> %s%n", theta, thetaTest, UNIT_CIRCLE.theta(thetaTest), UNIT_CIRCLE.theta(UNIT_CIRCLE.theta(thetaTest)));
        }
        circle = PoincareCircle.intersecting(new Point(0, .25), new Point(.5, .5));
        @NotNull Point poincareGenCoords = circle.asPoincareCircle().getPoincareGenCoords();
        System.out.println("Unit Angle: " + circle.angle(UNIT_CIRCLE));
        System.out.printf("d1=%s %s%n", circle.euclideanDistFromCenter(new Point(0, .25)), circle.euclideanDistFromCircle(new Point(0, .25)));
        System.out.printf("d2=%s %s%n", circle.euclideanDistFromCenter(new Point(.5, .5)), circle.euclideanDistFromCircle(new Point(.5, .5)));

    }

    @Test
    public void testPolygon() throws IOException {
        HyperbolicPolygon polygon = HyperbolicPolygon.regularPolygon(3, 20);
        Raster raster = new Raster(800, 800);
        HyperbolicTiling tiling = new HyperbolicTiling(polygon).expand(4);
        BufferedImage paint = raster.getImage();
        int[] pixelMap = tiling.buildPixelMap(paint, raster);
        show(paint);
        BufferedImage testImage = ImageUtil.resize(ImageUtil.getImage("file:///C:/Users/andre/code/all-projects/report/HyperbolicTexture/8abdf685-f6ef-4b86-b7d4-b27f03bddd44/etc/image_277b19524a6e2d.jpg"), raster.sizeX, raster.sizeY);
        show(new ImgIndexMapViewLayer(raster, pixelMap).eval(Tensor.fromRGB(testImage)).getData().get(0).toRgbImage());
    }

    @Test
    public void finishImage() throws IOException {
        BufferedImage image = ImageUtil.getImage("file:///C:/Users/andre/code/all-projects/report/HyperbolicTexture/bd074ae8-5e7d-48e2-a74f-a60aae1c474f/etc/image_8a5a91d97620a8e6.jpg");
        HyperbolicPolygon polygon = HyperbolicPolygon.regularPolygon(4,6);
        show(polygon.process_poincare(image, 2));
//        int size = (int) (image.getWidth() * 1.5);
//        image = ImageUtil.resize(image, size, size);
        //show(polygon.process_poincare_zoom(image, 2, x->(2*x)/(1+x*x)));
        //show(polygon.process_klien(image, 2, 1.0));
    }

    @Test
    public void finishImages() throws IOException {
        HyperbolicPolygon polygon = HyperbolicPolygon.regularPolygon(5,6);
        int width = -1;
        int superscaling = 2;
        Raster raster = null;
        int[] pixelMap = null;
        for (File file : new File("C:\\Users\\andre\\Downloads\\hyperbolic_reprocess\\5_6").listFiles()) {
            BufferedImage image = ImageIO.read(file);
            if(width != image.getWidth()) {
                width = image.getWidth();
                raster = new Raster(width * superscaling, width * superscaling);
                pixelMap = new HyperbolicTiling(polygon).expand(3).buildPixelMap(null, raster);
            }
            BufferedImage result = ImageUtil.resize(new ImgIndexMapViewLayer(raster, pixelMap).eval(Tensor.fromRGB(raster.resize(image))).getData().get(0).toRgbImage(), width, width);
            ImageIO.write(result, "jpg", new File(file.getParentFile(), "resampled_" + file.getName()));
        }
    }

    @Test
    public void testKlien() throws IOException {
        Raster raster = new Raster(1200, 1200);
        UnaryOperator<Point> transform = new HyperbolicTiling(HyperbolicPolygon.regularPolygon(4, 6)).expand(3).klien();
        BufferedImage testImage = ImageUtil.resize(ImageUtil.getImage("file:///C:/Users/andre/code/all-projects/report/HyperbolicTexture/8abdf685-f6ef-4b86-b7d4-b27f03bddd44/etc/image_277b19524a6e2d.jpg"), raster.sizeX, raster.sizeY);
        show(raster.toLayer(transform).eval(Tensor.fromRGB(testImage)).getData().get(0).toRgbImage());

    }

    public static void show(BufferedImage image) throws IOException {
        File tempFile = File.createTempFile("testPoly_", ".png");
        ImageIO.write(image, "png", tempFile);
        Desktop.getDesktop().open(tempFile);
    }

}
