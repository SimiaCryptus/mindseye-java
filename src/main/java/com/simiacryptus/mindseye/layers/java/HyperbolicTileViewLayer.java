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

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.lang.DataSerializer;
import com.simiacryptus.ref.wrappers.RefArrays;
import org.jetbrains.annotations.NotNull;

import javax.annotation.Nonnull;
import java.util.Arrays;
import java.util.Map;
import java.util.stream.Stream;

/**
 * The type Img view layer.
 */
@SuppressWarnings("serial")
public class HyperbolicTileViewLayer extends ImgViewLayerBase {

    private int sizeX;
    private int sizeY;
    private int p;
    private int q;
    private int[] pixelMap;

    /**
     * Instantiates a new Img view layer.
     *
     * @param sizeX the size x
     * @param sizeY the size y
     */
    public HyperbolicTileViewLayer(final int sizeX, final int sizeY, final int p, final int q) {
        super();
        this.setSizeX(sizeX);
        this.setSizeY(sizeY);
        this.p = p;
        this.q = q;
        this.pixelMap = buildMap();
    }

    private int[] buildMap() {
        int[] map = new int[sizeX * sizeY];

        return map;
    }

    /**
     * Instantiates a new Img view layer.
     *
     * @param json the json
     */
    protected HyperbolicTileViewLayer(@Nonnull final JsonObject json) {
        super(json);
        setSizeX(json.getAsJsonPrimitive("sizeX").getAsInt());
        setSizeY(json.getAsJsonPrimitive("sizeY").getAsInt());
        p = (json.getAsJsonPrimitive("p").getAsInt());
        q = (json.getAsJsonPrimitive("q").getAsInt());
        this.pixelMap = buildMap();
    }

    @Nonnull
    public int[] getViewDimensions(@Nonnull int @NotNull [] inputDims) {
        int[] destinationDimensions = new int[]{getSizeX(), getSizeY(), inputDims[2]};
        int[] offset = new int[]{getSizeX(), getSizeY(), 0};
        @Nonnull final int[] viewDim = new int[3];
        RefArrays.parallelSetAll(viewDim, i -> isWrap() ? destinationDimensions[i]
                : Math.min(inputDims[i], destinationDimensions[i] + offset[i]) - Math.max(offset[i], 0));
        if (null != channelSelector)
            viewDim[2] = channelSelector.length;
        return viewDim;
    }

    /**
     * Gets offset x.
     *
     * @return the offset x
     */
    public int getSizeX() {
        return sizeX;
    }

    /**
     * Sets offset x.
     *
     * @param sizeX the offset x
     */
    public void setSizeX(int sizeX) {
        this.sizeX = sizeX;
    }

    /**
     * Gets offset y.
     *
     * @return the offset y
     */
    public int getSizeY() {
        return sizeY;
    }

    /**
     * Sets offset y.
     *
     * @param sizeY the offset y
     */
    public void setSizeY(int sizeY) {
        this.sizeY = sizeY;
    }

    /**
     * From json img view layer.
     *
     * @param json the json
     * @param rs   the rs
     * @return the img view layer
     */
    @Nonnull
    @SuppressWarnings("unused")
    public static HyperbolicTileViewLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
        return new HyperbolicTileViewLayer(json);
    }

    @Nonnull
    @Override
    public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
        @Nonnull final JsonObject json = super.getJsonStub();
        json.addProperty("sizeX", getSizeX());
        json.addProperty("sizeY", getSizeY());
        json.addProperty("p", p);
        json.addProperty("q", q);
        if (null != getChannelSelector()) {
            JsonArray _channelPermutationFilter = new JsonArray();
            for (int i : getChannelSelector()) {
                _channelPermutationFilter.add(i);
            }
            json.add("channelSelector", _channelPermutationFilter);
        }
        return json;
    }

    /**
     * Coordinate mapping double [ ].
     *
     * @param xy the xy
     * @return the double [ ]
     */
    @Override
    @Nonnull
    protected double[] coordinateMapping(@Nonnull double... xy) {
        if (xy[0] < 0 || xy[0] >= sizeX) return xy;
        if (xy[1] < 0 || xy[1] >= sizeY) return xy;
        int targetIdx = pixelMap[(int) (xy[0] * sizeY + xy[1])];
        @Nonnull int[] ints = coordinateMapping((int) xy[0], (int) xy[0]);
        return new double[]{
                (double) ints[0],
                (double) ints[1]
        };
    }

    @Nonnull
    protected int[] coordinateMapping(@Nonnull int... xy) {
        int targetIdx = pixelMap[xy[0] * sizeY + xy[1]];
        return new int[]{targetIdx / sizeY, targetIdx % sizeY};
    }

}
