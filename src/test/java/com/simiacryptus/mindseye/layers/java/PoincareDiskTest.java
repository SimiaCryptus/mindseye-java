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

import com.simiacryptus.math.PoincareDisk;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.util.ImageUtil;
import org.jetbrains.annotations.NotNull;
import org.junit.jupiter.api.Test;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;

import static com.simiacryptus.math.PoincareDisk.UNIT_CIRCLE;

public class PoincareDiskTest {
    @Test
    public void test() throws IOException {
        //new PoincareGeometry.Circle()
        PoincareDisk.Circle circle;
        for (double theta = -3; theta < 3; theta += 0.5) {
            double[] thetaTest = UNIT_CIRCLE.theta(theta);
            System.out.printf("%s -> %s -> %s -> %s%n", theta, Arrays.toString(thetaTest), UNIT_CIRCLE.theta(thetaTest), Arrays.toString(UNIT_CIRCLE.theta(UNIT_CIRCLE.theta(thetaTest))));
        }
        circle = PoincareDisk.intersecting(new double[]{0, .25}, new double[]{.5, .5});
        @NotNull double[] poincareGenCoords = circle.getPoincareGenCoords();
        System.out.println("Unit Angle: " + circle.angle(UNIT_CIRCLE));
        System.out.printf("d1=%s %s%n", circle.euclideanDistFromCenter(new double[]{0, .25}), circle.euclideanDistFromCircle(new double[]{0, .25}));
        System.out.printf("d2=%s %s%n", circle.euclideanDistFromCenter(new double[]{.5, .5}), circle.euclideanDistFromCircle(new double[]{.5, .5}));

        PoincareDisk.Circle perp = circle.perpendicular(new double[]{0, .4}).edge;
        System.out.println(perp.toString());
        System.out.println("Unit Angle: " + perp.angle(UNIT_CIRCLE));
        System.out.printf("angle=%s%n", perp.angle(circle));
        System.out.printf("d1=%s %s%n", perp.euclideanDistFromCenter(new double[]{0, .4}), perp.euclideanDistFromCircle(new double[]{0, .4}));
        System.out.printf("d2=%s %s%n", circle.euclideanDistFromCenter(new double[]{0, .4}), circle.euclideanDistFromCircle(new double[]{0, .4}));

        double[] reflect = circle.reflect(new double[]{0, .4});
        System.out.println(Arrays.toString(reflect));
        System.out.printf("d1=%s %s%n", perp.euclideanDistFromCenter(reflect), perp.euclideanDistFromCircle(reflect));
        System.out.printf("d2=%s %s%n", circle.euclideanDistFromCenter(reflect), circle.euclideanDistFromCircle(reflect));
    }

    @Test
    public void testPolygon() throws IOException {
        PoincareDisk.Polygon polygon = PoincareDisk.regularPolygon(4, 6);
        System.out.println("Test Interior Angle: " + ((180 / Math.PI) * polygon.edges[0].angle(polygon.edges[1])));
        System.out.println("Test Vertex Agreement: " + polygon.edges[0].euclideanDistFromCircle(polygon.vertices[0]));
        System.out.println("Test Vertex Agreement: " + polygon.edges[0].euclideanDistFromCircle(polygon.vertices[1]));
        System.out.println("Test Vertex Agreement: " + polygon.edges[1].euclideanDistFromCircle(polygon.vertices[1]));
        PoincareDisk.Raster raster = new PoincareDisk.Raster(800, 800);
        int[] pixelMap = pixelMap(polygon, raster);
        BufferedImage testImage = ImageUtil.resize(ImageUtil.getImage("file:///C:/Users/andre/Pictures/texture_sources/shutterstock_248374732_centered.jpg"), raster.sizeX, raster.sizeY);
        show(raster.view(pixelMap, testImage));
        show(new ImgIndexMapViewLayer(raster, pixelMap).eval(Tensor.fromRGB(testImage)).getData().get(0).toRgbImage());

    }

    private int[] pixelMap(PoincareDisk.Polygon polygon, PoincareDisk.Raster raster) throws IOException {
        PoincareDisk.TilingResult tilingResult = raster.pixelMap(polygon, 4);
        show(tilingResult.getPaint());
        return tilingResult.getPixelMap();
    }

    public static void show(BufferedImage image) throws IOException {
        File tempFile = File.createTempFile("testPoly_", ".png");
        ImageIO.write(image, "png", tempFile);
        Desktop.getDesktop().open(tempFile);
    }

}
