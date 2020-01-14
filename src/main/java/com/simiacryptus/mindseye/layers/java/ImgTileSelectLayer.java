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

import com.google.gson.JsonObject;
import com.google.gson.JsonPrimitive;
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.lang.ReferenceCounting;
import com.simiacryptus.ref.wrappers.*;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.Map;
import java.util.UUID;
import java.util.function.Consumer;
import java.util.function.IntFunction;

@SuppressWarnings("serial")
public class ImgTileSelectLayer extends LayerBase {

  private final boolean toroidal;
  private final int sizeX;
  private final int sizeY;
  private final int positionX;
  private final int positionY;

  public ImgTileSelectLayer(final int sizeX, final int sizeY, final int positionX, final int positionY) {
    this(sizeX, sizeY, positionX, positionY, false);
  }

  public ImgTileSelectLayer(final int sizeX, final int sizeY, final int positionX, final int positionY,
                            final boolean toroidal) {
    super();
    this.sizeX = sizeX;
    this.sizeY = sizeY;
    this.positionX = positionX;
    this.positionY = positionY;
    this.toroidal = toroidal;
  }

  protected ImgTileSelectLayer(@Nonnull final JsonObject json) {
    super(json);
    sizeX = json.getAsJsonPrimitive("sizeX").getAsInt();
    sizeY = json.getAsJsonPrimitive("sizeY").getAsInt();
    positionX = json.getAsJsonPrimitive("positionX").getAsInt();
    positionY = json.getAsJsonPrimitive("positionY").getAsInt();
    JsonPrimitive toroidal = json.getAsJsonPrimitive("toroidal");
    this.toroidal = null != toroidal && toroidal.getAsBoolean();
  }

  @Nonnull
  public static void copy(@Nonnull final Tensor inputData, @Nonnull final Tensor outputData, final int posX,
                          final int posY, final boolean toroidal) {
    @Nonnull final int[] inDim = inputData.getDimensions();
    @Nonnull final int[] outDim = outputData.getDimensions();
    assert 3 == inDim.length;
    assert 3 == outDim.length;
    assert inDim[2] == outDim[2] : RefArrays.toString(inDim) + "; " + RefArrays.toString(outDim);
    outputData.coordStream(true).forEach(RefUtil.wrapInterface((Consumer<? super Coordinate>) (c) -> {
      int x = c.getCoords()[0] + posX;
      int y = c.getCoords()[1] + posY;
      int z = c.getCoords()[2];
      int width = inputData.getDimensions()[0];
      int height = inputData.getDimensions()[1];
      if (toroidal) {
        while (x < 0)
          x += width;
        x %= width;
        while (y < 0)
          y += height;
        y %= height;
      }
      double value;
      if (x < 0) {
        value = 0.0;
      } else if (x >= width) {
        value = 0.0;
      } else if (y < 0) {
        value = 0.0;
      } else if (y >= height) {
        value = 0.0;
      } else {
        value = inputData.get(x, y, z);
      }
      RefUtil.freeRef(outputData.set(c, value));
    }, outputData, inputData));
  }

  @Nonnull
  @SuppressWarnings("unused")
  public static ImgTileSelectLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new ImgTileSelectLayer(json);
  }

  @Nonnull
  public static Tensor[] toTiles(@Nonnull final NotebookOutput log, @Nonnull final Tensor canvas, final int width, final int height,
                                 final int strideX, final int strideY, final int offsetX, final int offsetY) {

    @Nonnull final int[] inputDims = canvas.getDimensions();
    int cols = (int) (Math.ceil((inputDims[0] - width - offsetX) * 1.0 / strideX) + 1);
    int rows = (int) (Math.ceil((inputDims[1] - height - offsetY) * 1.0 / strideY) + 1);
    log.p(RefString.format(
        "Partition %s x %s png with %s x %s tile size into %s x %s grid with stride %s x %s offset %s x %s",
        inputDims[0], inputDims[1], width, height, cols, rows, strideX, strideY, offsetX, offsetY));
    Tensor[] tiles = new Tensor[rows * cols];
    int index = 0;
    for (int row = 0; row < rows; row++) {
      for (int col = 0; col < cols; col++) {
        int positionX = col * strideX + offsetX;
        int positionY = row * strideY + offsetY;
        ImgTileSelectLayer tileSelectLayer = new ImgTileSelectLayer(width, height, positionX, positionY,
            offsetX < 0 || offsetY < 0);
        Result temp_14_0005 = tileSelectLayer.eval(canvas.addRef());
        assert temp_14_0005 != null;
        TensorList temp_14_0006 = temp_14_0005.getData();
        Tensor temp_14_0001 = temp_14_0006.get(0);
        temp_14_0006.freeRef();
        temp_14_0005.freeRef();
        if (null != tiles[index++])
          tiles[index++].freeRef();
        tiles[index++] = temp_14_0001.addRef();
        temp_14_0001.freeRef();
        tileSelectLayer.freeRef();
      }
    }
    canvas.freeRef();
    return tiles;
  }

  @Nonnull
  public static ImgTileSelectLayer[] tileSelectors(@Nonnull final NotebookOutput log, @Nonnull final Tensor canvas, final int width,
                                                   final int height, final int strideX, final int strideY, final int offsetX, final int offsetY) {

    @Nonnull final int[] inputDims = canvas.getDimensions();
    canvas.freeRef();
    int cols = (int) (Math.ceil((inputDims[0] - width - offsetX) * 1.0 / strideX) + 1);
    int rows = (int) (Math.ceil((inputDims[1] - height - offsetY) * 1.0 / strideY) + 1);
    log.p(RefString.format(
        "Partition %s x %s png with %s x %s tile size into %s x %s grid with stride %s x %s offset %s x %s",
        inputDims[0], inputDims[1], width, height, cols, rows, strideX, strideY, offsetX, offsetY));
    ImgTileSelectLayer[] tiles = new ImgTileSelectLayer[rows * cols];
    int index = 0;
    for (int row = 0; row < rows; row++) {
      for (int col = 0; col < cols; col++) {
        int positionX = col * strideX + offsetX;
        int positionY = row * strideY + offsetY;
        ImgTileSelectLayer tileSelectLayer = new ImgTileSelectLayer(width, height, positionX, positionY,
            offsetX < 0 || offsetY < 0);
        ImgTileSelectLayer temp_14_0002 = tileSelectLayer.addRef();
        if (null != tiles[index++])
          tiles[index++].freeRef();
        tiles[index++] = temp_14_0002.addRef();
        temp_14_0002.freeRef();
        tileSelectLayer.freeRef();
      }
    }
    return tiles;
  }

  @Nullable
  public static @SuppressWarnings("unused")
  ImgTileSelectLayer[] addRefs(@Nullable ImgTileSelectLayer[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ImgTileSelectLayer::addRef)
        .toArray((x) -> new ImgTileSelectLayer[x]);
  }

  @Nullable
  public static @SuppressWarnings("unused")
  ImgTileSelectLayer[][] addRefs(@Nullable ImgTileSelectLayer[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ImgTileSelectLayer::addRefs)
        .toArray((x) -> new ImgTileSelectLayer[x][]);
  }

  @Nonnull
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    final Result input = inObj[0].addRef();
    ReferenceCounting.freeRefs(inObj);
    final TensorList batch = input.getData();
    @Nonnull final int[] inputDims = batch.getDimensions();
    assert 3 == inputDims.length;
    @Nonnull final int[] dimOut = getViewDimensions(inputDims, new int[]{sizeX, sizeY, inputDims[2]},
        new int[]{positionX, positionY, 0});
    try {
      try {
        return new Result(new TensorArray(RefIntStream.range(0, batch.length())
            .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) dataIndex -> {
              @Nonnull final Tensor outputData = new Tensor(dimOut);
              Tensor inputData = batch.get(dataIndex);
              copy(inputData.addRef(), outputData.addRef(),
                  positionX, positionY, toroidal);
              inputData.freeRef();
              return outputData;
            }, batch.addRef())).toArray(i -> new Tensor[i])), new Result.Accumulator() {
          {
          }

          @Override
          public void accept(@Nullable DeltaSet<UUID> buffer, @Nonnull TensorList error) {
            if (input.isAlive()) {
              @Nonnull
              TensorArray tensorArray = new TensorArray(RefIntStream.range(0, error.length())
                  .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) dataIndex -> {
                    @Nullable final Tensor err = error.get(dataIndex);
                    @Nonnull final Tensor passback = new Tensor(inputDims);
                    copy(err.addRef(), passback.addRef(), -positionX,
                        -positionY, toroidal);
                    err.freeRef();
                    return passback;
                  }, error.addRef())).toArray(i -> new Tensor[i]));
              input.accumulate(buffer == null ? null : buffer.addRef(), tensorArray);
            }
            error.freeRef();
            if (null != buffer)
              buffer.freeRef();
          }

          public @SuppressWarnings("unused")
          void _free() {
          }
        }) {

          {
          }

          @Override
          public boolean isAlive() {
            return input.isAlive() || !isFrozen();
          }

          public void _free() {
          }
        };
      } finally {
        batch.freeRef();
      }
    } finally {
      input.freeRef();
    }
  }

  @Nonnull
  public int[] getViewDimensions(int[] sourceDimensions, int[] destinationDimensions, int[] offset) {
    @Nonnull final int[] viewDim = new int[3];
    RefArrays.parallelSetAll(viewDim, i -> toroidal ? (destinationDimensions[i])
        : (Math.min(sourceDimensions[i], destinationDimensions[i] + offset[i]) - Math.max(offset[i], 0)));
    return viewDim;
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    json.addProperty("sizeX", sizeX);
    json.addProperty("sizeY", sizeY);
    json.addProperty("positionX", positionX);
    json.addProperty("positionY", positionY);
    json.addProperty("toroidal", toroidal);
    return json;
  }

  @Nonnull
  @Override
  public RefList<double[]> state() {
    return new RefArrayList<>();
  }

  public @SuppressWarnings("unused")
  void _free() {
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  ImgTileSelectLayer addRef() {
    return (ImgTileSelectLayer) super.addRef();
  }

}
