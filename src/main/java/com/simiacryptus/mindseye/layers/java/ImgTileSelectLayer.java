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

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.*;
import java.util.stream.IntStream;
import com.simiacryptus.ref.wrappers.RefIntStream;

@SuppressWarnings("serial")
public @com.simiacryptus.ref.lang.RefAware class ImgTileSelectLayer extends LayerBase {

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
    @Nonnull
    final int[] inDim = inputData.getDimensions();
    @Nonnull
    final int[] outDim = outputData.getDimensions();
    assert 3 == inDim.length;
    assert 3 == outDim.length;
    assert inDim[2] == outDim[2] : com.simiacryptus.ref.wrappers.RefArrays.toString(inDim) + "; "
        + com.simiacryptus.ref.wrappers.RefArrays.toString(outDim);
    outputData.coordStream(true).forEach((c) -> {
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
      outputData.set(c, value);
    });
  }

  @SuppressWarnings("unused")
  public static ImgTileSelectLayer fromJson(@Nonnull final JsonObject json,
      com.simiacryptus.ref.wrappers.RefMap<CharSequence, byte[]> rs) {
    return new ImgTileSelectLayer(json);
  }

  @Nonnull
  public static Tensor[] toTiles(final NotebookOutput log, final Tensor canvas, final int width, final int height,
      final int strideX, final int strideY, final int offsetX, final int offsetY) {

    @Nonnull
    final int[] inputDims = canvas.getDimensions();
    int cols = (int) (Math.ceil((inputDims[0] - width - offsetX) * 1.0 / strideX) + 1);
    int rows = (int) (Math.ceil((inputDims[1] - height - offsetY) * 1.0 / strideY) + 1);
    log.p(String.format(
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
        tiles[index++] = tileSelectLayer.eval(canvas).getData().get(0);
      }
    }
    return tiles;
  }

  @Nonnull
  public static ImgTileSelectLayer[] tileSelectors(final NotebookOutput log, final Tensor canvas, final int width,
      final int height, final int strideX, final int strideY, final int offsetX, final int offsetY) {

    @Nonnull
    final int[] inputDims = canvas.getDimensions();
    int cols = (int) (Math.ceil((inputDims[0] - width - offsetX) * 1.0 / strideX) + 1);
    int rows = (int) (Math.ceil((inputDims[1] - height - offsetY) * 1.0 / strideY) + 1);
    log.p(String.format(
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
        tiles[index++] = tileSelectLayer;
      }
    }
    return tiles;
  }

  @Nonnull
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    final Result input = inObj[0];
    final TensorList batch = input.getData();
    @Nonnull
    final int[] inputDims = batch.getDimensions();
    assert 3 == inputDims.length;
    @Nonnull
    final int[] dimOut = getViewDimensions(inputDims, new int[] { sizeX, sizeY, inputDims[2] },
        new int[] { positionX, positionY, 0 });
    return new Result(
        new TensorArray(com.simiacryptus.ref.wrappers.RefIntStream.range(0, batch.length()).mapToObj(dataIndex -> {
          @Nonnull
          final Tensor outputData = new Tensor(dimOut);
          Tensor inputData = batch.get(dataIndex);
          copy(inputData, outputData, positionX, positionY, toroidal);
          return outputData;
        }).toArray(i -> new Tensor[i])), (@Nonnull final DeltaSet<UUID> buffer, @Nonnull final TensorList error) -> {
          if (input.isAlive()) {
            @Nonnull
            TensorArray tensorArray = new TensorArray(
                com.simiacryptus.ref.wrappers.RefIntStream.range(0, error.length()).mapToObj(dataIndex -> {
                  @Nullable
                  final Tensor err = error.get(dataIndex);
                  @Nonnull
                  final Tensor passback = new Tensor(inputDims);
                  copy(err, passback, -positionX, -positionY, toroidal);
                  return passback;
                }).toArray(i -> new Tensor[i]));
            input.accumulate(buffer, tensorArray);
          }
        }) {

      @Override
      public boolean isAlive() {
        return input.isAlive() || !isFrozen();
      }

      public void _free() {
      }
    };
  }

  @Nonnull
  public int[] getViewDimensions(int[] sourceDimensions, int[] destinationDimensions, int[] offset) {
    @Nonnull
    final int[] viewDim = new int[3];
    com.simiacryptus.ref.wrappers.RefArrays.parallelSetAll(viewDim, i -> toroidal ? (destinationDimensions[i])
        : (Math.min(sourceDimensions[i], destinationDimensions[i] + offset[i]) - Math.max(offset[i], 0)));
    return viewDim;
  }

  @Nonnull
  @Override
  public JsonObject getJson(com.simiacryptus.ref.wrappers.RefMap<CharSequence, byte[]> resources,
      DataSerializer dataSerializer) {
    @Nonnull
    final JsonObject json = super.getJsonStub();
    json.addProperty("sizeX", sizeX);
    json.addProperty("sizeY", sizeY);
    json.addProperty("positionX", positionX);
    json.addProperty("positionY", positionY);
    json.addProperty("toroidal", toroidal);
    return json;
  }

  @Nonnull
  @Override
  public com.simiacryptus.ref.wrappers.RefList<double[]> state() {
    return new com.simiacryptus.ref.wrappers.RefArrayList<>();
  }

  public @SuppressWarnings("unused") void _free() {
  }

  public @Override @SuppressWarnings("unused") ImgTileSelectLayer addRef() {
    return (ImgTileSelectLayer) super.addRef();
  }

  public static @SuppressWarnings("unused") ImgTileSelectLayer[] addRefs(ImgTileSelectLayer[] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(ImgTileSelectLayer::addRef)
        .toArray((x) -> new ImgTileSelectLayer[x]);
  }

  public static @SuppressWarnings("unused") ImgTileSelectLayer[][] addRefs(ImgTileSelectLayer[][] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(ImgTileSelectLayer::addRefs)
        .toArray((x) -> new ImgTileSelectLayer[x][]);
  }

}
